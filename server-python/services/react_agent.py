# 作者：yangkunpeng1
# 日期：2026-07-23
"""
轻量 ReAct：有限步 Thought → Action → Observation。

工具：
- read_memory：读对话摘要 / 硬事实 / 最近原文
- search_chat：检索被压缩出窗口的旧对话向量
- search_doc：向量检索文档
- finish：结束循环，进入最终回答
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence


ChatFn = Callable[[list[dict], int], str]
SearchFn = Callable[[str], list[dict]]

_VALID_ACTIONS = ("read_memory", "search_chat", "search_doc", "finish")
_ACTION_RE = re.compile(
    r"Action\s*[:：]\s*(read_memory|search_chat|search_doc|finish)\s*"
    r"(?:Action\s*Input\s*[:：]\s*(.*))?",
    re.IGNORECASE | re.DOTALL,
)
_DOC_HINT_RE = re.compile(
    r"(订单|合同|条款|金额|单价|总价|页码|第\s*\d+\s*页|比例|税率|日期|交付|"
    r"编号|单号|发票|附件|定义|多少|几号|哪一章|哪一节)",
    re.IGNORECASE,
)
_MEMORY_HINT_RE = re.compile(
    r"(之前|刚才|上次|我们聊|聊过|说过|回顾|总结一下对话|对话里|你记得)",
    re.IGNORECASE,
)


def react_max_steps() -> int:
    raw = os.environ.get("SMARTDOC_REACT_MAX_STEPS", "2")
    try:
        n = int(raw)
    except ValueError:
        n = 2
    return max(1, min(6, n))


def looks_like_doc_query(question: str) -> bool:
    return bool(_DOC_HINT_RE.search(question or ""))


def looks_like_memory_query(question: str) -> bool:
    return bool(_MEMORY_HINT_RE.search(question or ""))


def format_memory_observation(
    *,
    memory_summary: str,
    memory_facts: str,
    recent_messages: Sequence[dict],
) -> str:
    parts: list[str] = []
    facts = (memory_facts or "").strip()
    summary = (memory_summary or "").strip()
    recent_lines: list[str] = []
    for m in recent_messages or []:
        role = "用户" if m.get("role") == "user" else "助手"
        content = (m.get("content") or "").strip()
        if content:
            recent_lines.append(f"{role}：{content}")
    if facts:
        parts.append("【硬事实】\n" + facts)
    if summary:
        parts.append("【对话摘要】\n" + summary)
    if recent_lines:
        parts.append("【最近对话】\n" + "\n".join(recent_lines))
    if not parts:
        return "（记忆为空：尚无摘要、硬事实或最近对话）"
    return "\n\n".join(parts)


def format_search_observation(
    chunks: Sequence[dict],
    *,
    empty_msg: str = "（未检索到相关文档片段）",
    max_chars: int = 2400,
) -> str:
    if not chunks:
        return empty_msg
    blocks: list[str] = []
    used = 0
    for i, c in enumerate(chunks, 1):
        t = (c.get("text") or "").strip()
        if not t:
            continue
        score = c.get("score")
        page = c.get("page")
        source = c.get("source") or c.get("kind") or ""
        head = f"【片段{i}】"
        if source:
            head += f" source={source}"
        if page:
            head += f" page={page}"
        if score is not None:
            head += f" score={score}"
        block = f"{head}\n{t}"
        if used + len(block) > max_chars and blocks:
            break
        blocks.append(block)
        used += len(block) + 2
    return "\n\n".join(blocks) if blocks else empty_msg


def parse_react_decision(text: str) -> tuple[str, str, str]:
    """
    解析模型输出为 (thought, action, action_input)。
    优先 JSON，其次 ReAct 文本行；失败默认 finish。
    """
    raw = (text or "").strip()
    if not raw:
        return "", "finish", ""

    candidates = [raw]
    brace = raw.find("{")
    if brace >= 0:
        end = raw.rfind("}")
        if end > brace:
            candidates.insert(0, raw[brace : end + 1])

    for cand in candidates:
        try:
            data = json.loads(cand)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        action = str(data.get("action") or data.get("Action") or "").strip().lower()
        if action not in _VALID_ACTIONS:
            continue
        thought = str(data.get("thought") or data.get("Thought") or "").strip()
        action_input = str(
            data.get("action_input")
            or data.get("Action Input")
            or data.get("input")
            or ""
        ).strip()
        return thought, action, action_input

    thought = ""
    tm = re.search(r"Thought\s*[:：]\s*(.+?)(?=\n\s*Action\s*[:：]|$)", raw, re.I | re.S)
    if tm:
        thought = tm.group(1).strip()
    m = _ACTION_RE.search(raw)
    if m:
        action = m.group(1).strip().lower()
        action_input = (m.group(2) or "").strip()
        action_input = re.split(r"\n\s*(Thought|Action)\s*[:：]", action_input, maxsplit=1)[0].strip()
        return thought, action, action_input

    low = raw.lower()
    for name in _VALID_ACTIONS:
        if name in low:
            return thought, name, ""
    return thought, "finish", ""


def _planner_prompt(
    question: str,
    *,
    step: int,
    max_steps: int,
    trace_text: str,
    searched_doc: bool,
    searched_chat: bool,
    read_memory_done: bool,
    chat_search_available: bool,
) -> list[dict]:
    chat_tool_line = (
        "- search_chat：检索本会话里被压缩出窗口的旧对话原文"
        "（适合回忆某次说过的细节；Action Input 填检索问句）\n"
        if chat_search_available
        else ""
    )
    chat_rule = (
        "3. 摘要/硬事实不够、需要对话原话细节时用 search_chat；同一检索词不要重复。\n"
        if chat_search_available
        else "3. （当前会话无旧对话向量库，勿选 search_chat）\n"
    )
    rules = (
        "你是文档问答 ReAct 调度器。根据用户问题与已有 Observation，决定下一步工具。\n"
        "可用工具：\n"
        "- read_memory：查看对话摘要、硬事实、最近对话（适合「之前大致聊了啥」）\n"
        f"{chat_tool_line}"
        "- search_doc：在文档向量库检索（适合订单号、金额、条款、日期等文档事实；"
        "Action Input 填检索问句）\n"
        "- finish：信息已够，结束工具循环去回答\n\n"
        "判断「够不够」：对照用户问题，看 Observation 是否已含可直接作答的要点；"
        "缺实体/数字/原话就继续查，已覆盖就 finish；不要为了搜而搜。\n\n"
        "硬规则：\n"
        "1. 涉及文档实体/数字/条款/页码时，至少 search_doc 一次，不能只靠记忆瞎答。\n"
        "2. 纯对话回顾：优先 read_memory；若还需细节再用 search_chat，不必强行 search_doc。\n"
        f"{chat_rule}"
        "4. 信息够了立刻 finish。\n"
        f"5. 当前第 {step}/{max_steps} 步，步数用尽前尽量 finish。\n\n"
        "只输出一个 JSON 对象，不要其它文字：\n"
        '{"thought":"简短理由","action":"read_memory|search_chat|search_doc|finish","action_input":"..."}\n'
    )
    status = (
        f"已读记忆：{'是' if read_memory_done else '否'}；"
        f"已搜旧对话：{'是' if searched_chat else '否'}；"
        f"已搜文档：{'是' if searched_doc else '否'}；"
        f"旧对话库可用：{'是' if chat_search_available else '否'}。\n"
    )
    user = (
        f"【用户问题】\n{question}\n\n"
        f"【状态】\n{status}\n"
        f"【已有步骤】\n{trace_text or '（尚无）'}\n\n"
        "下一步："
    )
    return [
        {"role": "system", "content": rules},
        {"role": "user", "content": user},
    ]


@dataclass
class ReactResult:
    chunks: list[dict] = field(default_factory=list)
    chat_chunks: list[dict] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    chat_queries: list[str] = field(default_factory=list)
    used_memory: bool = False
    memory_observation: str = ""
    finish_reason: str = "finish"
    trace: list[dict[str, Any]] = field(default_factory=list)


def _merge_chunks(dst: list[dict], src: Sequence[dict]) -> None:
    existing_keys = {((c.get("text") or "")[:80]).strip() for c in dst}
    for c in src:
        key = ((c.get("text") or "")[:80]).strip()
        if key and key not in existing_keys:
            dst.append(c)
            existing_keys.add(key)


def run_react(
    question: str,
    *,
    memory_summary: str,
    memory_facts: str,
    recent_messages: Sequence[dict],
    chat_fn: ChatFn,
    search_fn: SearchFn,
    search_chat_fn: SearchFn | None = None,
    rewrite_fn: Callable[[str], str] | None = None,
    max_steps: int | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> ReactResult:
    """
    执行有限步 ReAct。
    明确文档题 / 纯回顾题走快路径，避免多次规划 LLM。
    """
    from services.conversation_memory import needs_question_rewrite

    q = (question or "").strip()
    steps = react_max_steps() if max_steps is None else max(1, int(max_steps))
    result = ReactResult()
    seen_doc_queries: set[str] = set()
    seen_chat_queries: set[str] = set()
    searched_doc = False
    searched_chat = False
    read_memory_done = False
    chat_available = search_chat_fn is not None

    def cancelled() -> bool:
        return bool(cancel_check and cancel_check())

    def maybe_rewrite(query: str) -> str:
        if not rewrite_fn:
            return query
        if query.strip() != q and query.strip():
            return query
        if not needs_question_rewrite(q):
            return query
        try:
            rewritten = (rewrite_fn(q) or "").strip()
            return rewritten or query
        except Exception as e:
            print(f"[REACT] rewrite failed: {e}", flush=True)
            return query

    # 快路径：文档向问题（零规划 LLM）
    if looks_like_doc_query(q) and not looks_like_memory_query(q):
        query = maybe_rewrite(q)
        try:
            chunks = search_fn(query) or []
        except Exception as e:
            print(f"[REACT] fast_doc search failed: {e}", flush=True)
            chunks = []
        _merge_chunks(result.chunks, chunks)
        result.search_queries.append(query)
        result.trace.append(
            {
                "step": 1,
                "thought": "快路径：文档向问题直接检索，跳过规划",
                "action": "search_doc",
                "action_input": query,
                "observation": format_search_observation(chunks),
            }
        )
        result.finish_reason = "fast_doc"
        print(
            f"[REACT] fast_doc query={query[:60]!r} hits={len(result.chunks)}",
            flush=True,
        )
        return result

    # 快路径：纯回顾（零规划 LLM）
    if looks_like_memory_query(q) and not looks_like_doc_query(q):
        obs = format_memory_observation(
            memory_summary=memory_summary,
            memory_facts=memory_facts,
            recent_messages=recent_messages,
        )
        result.used_memory = True
        result.memory_observation = obs
        result.trace.append(
            {
                "step": 1,
                "thought": "快路径：回顾类先读记忆",
                "action": "read_memory",
                "action_input": "",
                "observation": obs,
            }
        )
        if chat_available:
            try:
                chunks = search_chat_fn(q) or []
            except Exception as e:
                print(f"[REACT] fast_memory search_chat failed: {e}", flush=True)
                chunks = []
            _merge_chunks(result.chat_chunks, chunks)
            result.chat_queries.append(q)
            result.trace.append(
                {
                    "step": 2,
                    "thought": "快路径：补检索旧对话",
                    "action": "search_chat",
                    "action_input": q,
                    "observation": format_search_observation(
                        chunks, empty_msg="（未检索到相关旧对话片段）"
                    ),
                }
            )
        result.finish_reason = "fast_memory"
        print(
            f"[REACT] fast_memory chat_hits={len(result.chat_chunks)}",
            flush=True,
        )
        return result

    force_doc_first = False

    for step in range(1, steps + 1):
        if cancelled():
            result.finish_reason = "cancelled"
            break

        if force_doc_first and not searched_doc and step == 1:
            thought = "问题像在问文档事实，先检索文档"
            action = "search_doc"
            action_input = q
        else:
            trace_text = "\n".join(
                f"{i}. Thought: {t.get('thought','')}\n"
                f"   Action: {t.get('action')}({t.get('action_input','')})\n"
                f"   Observation: {(t.get('observation') or '')[:400]}"
                for i, t in enumerate(result.trace, 1)
            )
            try:
                raw = chat_fn(
                    _planner_prompt(
                        q,
                        step=step,
                        max_steps=steps,
                        trace_text=trace_text,
                        searched_doc=searched_doc,
                        searched_chat=searched_chat,
                        read_memory_done=read_memory_done,
                        chat_search_available=chat_available,
                    ),
                    220,
                )
            except Exception as e:
                print(f"[REACT] planner failed: {e}", flush=True)
                thought, action, action_input = "", "finish", ""
            else:
                thought, action, action_input = parse_react_decision(raw)

            # 护栏：文档向问题尚未检索 → 强制 search_doc
            if looks_like_doc_query(q) and not searched_doc and action != "search_doc":
                thought = (thought + "；护栏：文档向问题必须先检索").strip("；")
                action = "search_doc"
                action_input = action_input or q

            # 不可用 search_chat 时降级
            if action == "search_chat" and not chat_available:
                if looks_like_memory_query(q):
                    action = "read_memory" if not read_memory_done else "finish"
                else:
                    action = "finish"
                thought = (thought + "；护栏：无旧对话库").strip("；")

            # 纯回顾误选 search_doc → 改 search_chat 或 finish
            if (
                action == "search_doc"
                and looks_like_memory_query(q)
                and not looks_like_doc_query(q)
            ):
                if chat_available and not searched_chat:
                    thought = (thought + "；护栏：回顾类改搜旧对话").strip("；")
                    action = "search_chat"
                    action_input = action_input or q
                elif read_memory_done:
                    thought = (thought + "；护栏：回顾类已读记忆，结束").strip("；")
                    action = "finish"
                    action_input = ""

        observation = ""
        if action == "read_memory":
            observation = format_memory_observation(
                memory_summary=memory_summary,
                memory_facts=memory_facts,
                recent_messages=recent_messages,
            )
            result.used_memory = True
            result.memory_observation = observation
            read_memory_done = True
        elif action == "search_chat":
            query = (action_input or "").strip() or q
            qkey = "".join(query.split()).lower()
            if qkey in seen_chat_queries:
                observation = f"（已用过相同旧对话检索词，跳过）query={query}"
            elif not chat_available:
                observation = "（旧对话检索不可用）"
            else:
                seen_chat_queries.add(qkey)
                try:
                    chunks = search_chat_fn(query) or []
                except Exception as e:
                    print(f"[REACT] search_chat failed: {e}", flush=True)
                    chunks = []
                searched_chat = True
                result.chat_queries.append(query)
                _merge_chunks(result.chat_chunks, chunks)
                observation = format_search_observation(
                    chunks,
                    empty_msg="（未检索到相关旧对话片段）",
                )
                if chunks:
                    result.used_memory = True
        elif action == "search_doc":
            query = maybe_rewrite((action_input or "").strip() or q)
            qkey = "".join(query.split()).lower()
            if qkey in seen_doc_queries:
                observation = f"（已用过相同检索词，跳过重复搜索）query={query}"
            else:
                seen_doc_queries.add(qkey)
                try:
                    chunks = search_fn(query) or []
                except Exception as e:
                    print(f"[REACT] search failed: {e}", flush=True)
                    chunks = []
                searched_doc = True
                result.search_queries.append(query)
                _merge_chunks(result.chunks, chunks)
                observation = format_search_observation(chunks)
        else:
            action = "finish"
            observation = "（结束工具循环）"

        result.trace.append(
            {
                "step": step,
                "thought": thought,
                "action": action,
                "action_input": action_input,
                "observation": observation,
            }
        )
        print(
            f"[REACT] step={step} action={action} "
            f"input={(action_input or '')[:60]!r} obs_chars={len(observation)}",
            flush=True,
        )

        if action == "finish":
            result.finish_reason = "finish"
            break

        # 已有文档命中则不再开下一轮规划 LLM
        if action == "search_doc" and result.chunks:
            result.finish_reason = "finish"
            break
    else:
        result.finish_reason = "max_steps"

    # 回顾类：从未读记忆则补读
    if looks_like_memory_query(q) and not read_memory_done:
        obs = format_memory_observation(
            memory_summary=memory_summary,
            memory_facts=memory_facts,
            recent_messages=recent_messages,
        )
        result.used_memory = True
        result.memory_observation = obs
        result.trace.append(
            {
                "step": len(result.trace) + 1,
                "thought": "补读记忆供回顾类问题作答",
                "action": "read_memory",
                "action_input": "",
                "observation": obs,
            }
        )

    return result
