# 作者：yangkunpeng1
# 日期：2026-07-23
"""
多轮对话记忆：最近 N 轮原文 + 叙述滚动摘要 + 硬事实清单 + 提问改写（供检索）。

硬事实（memory_facts）与叙述摘要分离：摘要可概括，事实只追加不改写，降低多轮压缩丢细节。
"""

from __future__ import annotations

import os
import re
from typing import Callable, Sequence


ChatFn = Callable[[list[dict], int], str]

_PINNED_RE = re.compile(r"^\[pinned\]\s*", re.IGNORECASE)


def history_keep_turns() -> int:
    """保留原文的轮数，默认 5，夹在 4～6。"""
    raw = os.environ.get("SMARTDOC_HISTORY_TURNS", "5")
    try:
        n = int(raw)
    except ValueError:
        n = 5
    return max(4, min(6, n))


def history_keep_messages() -> int:
    """按「一轮 ≈ user+assistant 两条」换算消息条数。"""
    return history_keep_turns() * 2


def normalize_history(history: Sequence[dict] | None) -> list[dict]:
    out: list[dict] = []
    for m in history or []:
        role = str((m or {}).get("role") or "").strip()
        content = str((m or {}).get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        out.append({"role": role, "content": content})
    return out


def split_history(
    history: Sequence[dict] | None,
    *,
    keep_messages: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """拆成 (更早需压缩, 最近原文)。"""
    hist = normalize_history(history)
    keep = history_keep_messages() if keep_messages is None else max(2, int(keep_messages))
    if len(hist) <= keep:
        return [], hist
    return hist[:-keep], hist[-keep:]


def format_turns_text(messages: Sequence[dict]) -> str:
    lines: list[str] = []
    for m in messages:
        role = "用户" if m.get("role") == "user" else "助手"
        lines.append(f"{role}：{m.get('content') or ''}")
    return "\n".join(lines)


def parse_fact_lines(text: str | None) -> list[str]:
    """把事实文本拆成行；去掉常见列表前缀。"""
    out: list[str] = []
    for raw in (text or "").splitlines():
        s = raw.strip()
        if not s:
            continue
        for p in ("- ", "• ", "* ", "· ", "– ", "— "):
            if s.startswith(p):
                s = s[len(p) :].strip()
                break
        if s:
            out.append(s)
    return out


def format_fact_lines(lines: Sequence[str]) -> str:
    return "\n".join(f"- {x.strip()}" for x in lines if (x or "").strip())


def _fact_key(line: str) -> str:
    s = _PINNED_RE.sub("", (line or "").strip())
    return "".join(s.split()).lower()


def is_pinned_fact(line: str) -> bool:
    return (line or "").strip().lower().startswith("[pinned]")


def merge_fact_lines(
    existing: str | None,
    new_lines: Sequence[str],
    *,
    max_chars: int = 2000,
) -> str:
    """
    追加合并事实行：同内容去重；新行为 pinned 时可升级旧行。
    超长时优先丢掉最旧的非 pinned。
    """
    out = parse_fact_lines(existing)
    key_to_idx = {_fact_key(x): i for i, x in enumerate(out)}

    for raw in new_lines:
        line = (raw or "").strip()
        if not line:
            continue
        for p in ("- ", "• ", "* ", "· "):
            if line.startswith(p):
                line = line[len(p) :].strip()
                break
        if not line:
            continue
        k = _fact_key(line)
        if not k:
            continue
        if k in key_to_idx:
            i = key_to_idx[k]
            old = out[i]
            if is_pinned_fact(line) and not is_pinned_fact(old):
                pinned = line if is_pinned_fact(line) else f"[pinned] {line}"
                out[i] = pinned
            continue
        out.append(line)
        key_to_idx[k] = len(out) - 1

    def packed(lines: Sequence[str]) -> str:
        return format_fact_lines(lines)

    text = packed(out)
    if len(text) <= max_chars:
        return text

    pinned = [x for x in out if is_pinned_fact(x)]
    normal = [x for x in out if not is_pinned_fact(x)]
    while normal and len(packed(pinned + normal)) > max_chars:
        normal.pop(0)
    text = packed(pinned + normal)
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def extract_facts_from_messages(
    messages: Sequence[dict],
    chat_fn: ChatFn,
) -> list[str]:
    """从对话中抽取硬事实行；失败返回空列表。"""
    older = normalize_history(messages)
    if not older:
        return []

    block = format_turns_text(older)
    prompt = (
        "从对话中抽取「硬事实」清单（可供后续多轮引用）。\n"
        "要求：\n"
        "- 只抽可核对信息：订单号/合同号、金额、日期、公司或人名、已确认方案、明确约定等；\n"
        "- 每条一行，优先「标签：值」；用户明确确认的在行首加 [pinned]；\n"
        "- 禁止概括成主题句，禁止寒暄；没有硬事实则只输出空行；\n"
        "- 只输出事实行，不要标题或解释。\n\n"
        f"【对话】\n{block}\n\n"
        "事实："
    )
    try:
        text = (chat_fn([{"role": "user", "content": prompt}], 300) or "").strip()
    except Exception as e:
        print(f"[MEMORY] extract_facts failed: {e}", flush=True)
        return []
    if not text or text in ("（无）", "(无)", "无", "没有", "无硬事实"):
        return []
    return parse_fact_lines(text)


def merge_memory_facts(
    existing: str,
    older_messages: Sequence[dict],
    chat_fn: ChatFn,
    *,
    max_chars: int = 2000,
) -> str:
    """从新增更早对话抽取硬事实，追加合并进已有清单。"""
    older = normalize_history(older_messages)
    if not older:
        return (existing or "").strip()

    new_lines = extract_facts_from_messages(older, chat_fn)
    if not new_lines:
        return merge_fact_lines(existing, [], max_chars=max_chars)
    return merge_fact_lines(existing, new_lines, max_chars=max_chars)


def merge_memory_summary(
    existing: str,
    older_messages: Sequence[dict],
    chat_fn: ChatFn,
    *,
    max_chars: int = 1200,
) -> str:
    """
    把「已有摘要 + 更早对话」压成新的滚动叙述摘要。
    硬事实（单号/金额/日期等）不要写进摘要——由 memory_facts 单独保管。
    chat_fn(messages, max_tokens) -> 文本
    """
    older = normalize_history(older_messages)
    if not older:
        return (existing or "").strip()

    prev = (existing or "").strip()
    block = format_turns_text(older)
    prompt = (
        "你是对话记忆压缩助手。请把「已有摘要」与「新增更早对话」合并成一段中文叙述摘要。\n"
        "要求：保留用户关注点、未决问题与关键指代；删除寒暄与重复；\n"
        "不要写入订单号、金额、日期、合同号等硬事实（另有事实清单保管）；\n"
        f"控制在 {max_chars} 字以内；只输出摘要正文。\n\n"
        f"【已有摘要】\n{prev or '（无）'}\n\n"
        f"【新增更早对话】\n{block}\n\n"
        "摘要："
    )
    try:
        text = (chat_fn([{"role": "user", "content": prompt}], 400) or "").strip()
    except Exception as e:
        print(f"[MEMORY] merge_summary failed: {e}", flush=True)
        fallback = (prev + "\n" + block).strip() if prev else block
        return fallback[:max_chars]

    if not text:
        fallback = (prev + "\n" + block).strip() if prev else block
        return fallback[:max_chars]
    return text[:max_chars]


def needs_question_rewrite(question: str) -> bool:
    """完整问句不必再调 LLM 改写；短问/指代追问才需要。"""
    q = (question or "").strip()
    if not q:
        return False
    if len(q) < 10:
        return True
    return bool(
        re.search(
            r"(那|这|它|他|她|该|上述|上面|刚才|之前|这个|那个|哪个|多少来着|呢\s*$|吗\s*$)",
            q,
        )
    )


def rewrite_question(
    question: str,
    *,
    memory_summary: str,
    recent_messages: Sequence[dict],
    chat_fn: ChatFn,
    memory_facts: str = "",
) -> str:
    """
    结合摘要、硬事实与最近原文，把追问改写成可独立检索的完整问句。
    失败时回退原问题。完整问句直接跳过（省一次 LLM）。
    """
    q = (question or "").strip()
    if not q:
        return q

    if not needs_question_rewrite(q):
        return q

    summary = (memory_summary or "").strip()
    facts = (memory_facts or "").strip()
    recent = normalize_history(recent_messages)
    if not summary and not facts and not recent:
        return q

    recent_txt = format_turns_text(recent) if recent else "（无）"
    prompt = (
        "根据对话摘要、已确认硬事实与最近对话，把「当前问题」改写成一句独立、完整、可检索的中文问句。\n"
        "要求：补全省略与指代（可引用硬事实中的单号/金额等）；不要回答问题；不要解释；只输出改写后的问句。\n\n"
        f"【对话摘要】\n{summary or '（无）'}\n\n"
        f"【已确认硬事实】\n{facts or '（无）'}\n\n"
        f"【最近对话】\n{recent_txt}\n\n"
        f"【当前问题】\n{q}\n\n"
        "改写问句："
    )
    try:
        text = (chat_fn([{"role": "user", "content": prompt}], 120) or "").strip()
    except Exception as e:
        print(f"[MEMORY] rewrite_question failed: {e}", flush=True)
        return q

    text = text.strip().strip("「」\"'")
    if not text or len(text) > 500:
        return q
    return text


def build_history_prompt_messages(
    memory_summary: str,
    recent_messages: Sequence[dict],
    memory_facts: str = "",
    *,
    for_final: bool = False,
    question: str = "",
) -> list[dict]:
    """
    拼进最终 LLM 的历史部分（不含本题文档与问题）。

    for_final=True：按问题检索相关事实/摘要句，再裁剪最近原文；非整表粘贴。
    """
    if for_final:
        q = (question or "").strip()
        if q:
            memory_facts = retrieve_relevant_facts(q, memory_facts)
            memory_summary = retrieve_relevant_summary(q, memory_summary)
        else:
            memory_summary = clip_text(memory_summary, final_summary_max_chars())
            memory_facts = clip_text(memory_facts, final_facts_max_chars())
        recent_messages = trim_recent_for_final(recent_messages)

    out: list[dict] = []
    facts = (memory_facts or "").strip()
    if facts:
        out.append(
            {
                "role": "system",
                "content": (
                    "以下是与当前问题相关的已确认硬事实（须优先采信，勿编造）：\n"
                    f"{facts}"
                ),
            }
        )
    summary = (memory_summary or "").strip()
    if summary:
        out.append(
            {
                "role": "system",
                "content": f"与当前问题相关的对话摘要：\n{summary}",
            }
        )
    for m in normalize_history(recent_messages):
        out.append({"role": m["role"], "content": m["content"]})
    return out


def clip_text(text: str | None, max_chars: int) -> str:
    s = (text or "").strip()
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    if max_chars <= 1:
        return "…"
    return s[: max_chars - 1].rstrip() + "…"


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        n = int(raw)
    except ValueError:
        n = default
    return max(lo, min(hi, n))


def final_summary_max_chars() -> int:
    """最终回答里摘要上限（存储仍可更长）。"""
    return _env_int("SMARTDOC_FINAL_SUMMARY_CHARS", 500, 100, 1200)


def final_facts_max_chars() -> int:
    return _env_int("SMARTDOC_FINAL_FACTS_CHARS", 600, 100, 2000)


def final_facts_top_k() -> int:
    return _env_int("SMARTDOC_FINAL_FACTS_TOP_K", 5, 1, 20)


def final_summary_top_sents() -> int:
    return _env_int("SMARTDOC_FINAL_SUMMARY_SENTS", 3, 1, 10)


def final_recent_messages() -> int:
    """最终回答保留的最近消息条数（约 2 轮 = 4 条）。"""
    return _env_int("SMARTDOC_FINAL_RECENT_MSGS", 4, 0, 10)


def final_msg_max_chars() -> int:
    """最终回答里单条最近消息截断。"""
    return _env_int("SMARTDOC_FINAL_MSG_CHARS", 200, 50, 800)


def final_doc_max_chars() -> int:
    """最终回答文档片段总字数上限。"""
    return _env_int("SMARTDOC_FINAL_DOC_CHARS", 3200, 500, 8000)


def final_chat_max_chars(*, with_doc: bool) -> int:
    """有文档时旧对话少带；纯记忆问答可稍多。"""
    default = 800 if with_doc else 1500
    name = "SMARTDOC_FINAL_CHAT_CHARS_WITH_DOC" if with_doc else "SMARTDOC_FINAL_CHAT_CHARS"
    return _env_int(name, default, 0, 4000)


def trim_recent_for_final(recent_messages: Sequence[dict] | None) -> list[dict]:
    keep = final_recent_messages()
    per = final_msg_max_chars()
    hist = normalize_history(recent_messages)
    if keep <= 0:
        return []
    sliced = hist[-keep:]
    out: list[dict] = []
    for m in sliced:
        out.append({"role": m["role"], "content": clip_text(m.get("content") or "", per)})
    return out


def trim_joined_blocks(text: str | None, max_chars: int) -> str:
    """按总长度裁剪已拼好的多段文本（尽量在段落边界切）。"""
    s = (text or "").strip()
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    cut = s[:max_chars]
    for sep in ("\n\n", "\n"):
        idx = cut.rfind(sep)
        if idx >= max_chars // 2:
            cut = cut[:idx]
            break
    return cut.rstrip() + "…"


def estimate_messages_chars(messages: Sequence[dict]) -> int:
    return sum(len(str(m.get("content") or "")) for m in messages)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    n = min(len(a), len(b))
    return float(sum(float(a[i]) * float(b[i]) for i in range(n)))


def _lexical_score(query: str, text: str) -> float:
    """无向量时的退化相关度：字符 bigram 重叠。"""
    q = "".join((query or "").lower().split())
    t = "".join((text or "").lower().split())
    if not q or not t:
        return 0.0
    if q in t or t in q:
        return 0.95
    q_bi = {q[i : i + 2] for i in range(len(q) - 1)} or {q}
    t_bi = {t[i : i + 2] for i in range(len(t) - 1)} or {t}
    inter = len(q_bi & t_bi)
    if not inter:
        return 0.0
    return inter / max(len(q_bi), 1)


def _score_texts(query: str, texts: Sequence[str]) -> list[float]:
    items = [str(x or "") for x in texts]
    if not items:
        return []
    # 条数不多时用字面相关度，避免每问都跑一遍 embedding（显著省首字延迟）
    if len(items) <= 24:
        return [_lexical_score(query, t) for t in items]
    try:
        from .embedder import embed_documents, embed_query

        qe = embed_query(query)
        de = embed_documents(items)
        return [_dot(qe, d) for d in de]
    except Exception as e:
        print(f"[MEMORY] embed score fallback lexical: {e}", flush=True)
        return [_lexical_score(query, t) for t in items]


def _split_summary_units(summary: str) -> list[str]:
    s = (summary or "").strip()
    if not s:
        return []
    # 先按换行，再按句号类切
    parts: list[str] = []
    for block in re.split(r"\n+", s):
        block = block.strip()
        if not block:
            continue
        bits = re.split(r"(?<=[。！？；!?])\s*", block)
        for b in bits:
            b = b.strip()
            if b:
                parts.append(b)
    return parts or [s]


def retrieve_relevant_facts(
    question: str,
    facts_text: str | None,
    *,
    top_k: int | None = None,
    max_chars: int | None = None,
) -> str:
    """
    按问题检索相关硬事实行；[pinned] 事实始终保留。
    返回 format 后的事实文本（可能为空）。
    """
    lines = parse_fact_lines(facts_text)
    if not lines:
        return ""
    q = (question or "").strip()
    k = final_facts_top_k() if top_k is None else max(1, int(top_k))
    cap = final_facts_max_chars() if max_chars is None else max(50, int(max_chars))

    pinned = [x for x in lines if is_pinned_fact(x)]
    normal = [x for x in lines if not is_pinned_fact(x)]

    if not q:
        chosen = pinned + normal[:k]
        return clip_text(format_fact_lines(chosen), cap)

    if not normal:
        return clip_text(format_fact_lines(pinned), cap)

    scores = _score_texts(q, normal)
    ranked = sorted(
        zip(normal, scores),
        key=lambda x: x[1],
        reverse=True,
    )
    # 至少取 top_k；极低分仍取最高的几条，避免全空
    picked = [t for t, _ in ranked[:k]]
    # 去重合并 pinned
    merged: list[str] = []
    seen: set[str] = set()
    for line in pinned + picked:
        key = _fact_key(line)
        if key in seen:
            continue
        seen.add(key)
        merged.append(line)
    return clip_text(format_fact_lines(merged), cap)


def retrieve_relevant_summary(
    question: str,
    summary: str | None,
    *,
    top_sents: int | None = None,
    max_chars: int | None = None,
) -> str:
    """按问题检索相关摘要句，再拼回短摘要。"""
    s = (summary or "").strip()
    if not s:
        return ""
    q = (question or "").strip()
    cap = final_summary_max_chars() if max_chars is None else max(50, int(max_chars))
    n = final_summary_top_sents() if top_sents is None else max(1, int(top_sents))

    units = _split_summary_units(s)
    if len(units) <= n or not q:
        return clip_text(s, cap)

    scores = _score_texts(q, units)
    ranked = sorted(range(len(units)), key=lambda i: scores[i], reverse=True)
    # 保持原文顺序，避免摘要读起来乱跳
    keep_idx = sorted(ranked[:n])
    text = "".join(units[i] if units[i].endswith(("。", "！", "？", ";", "；")) else units[i] + "。" for i in keep_idx)
    # 上面可能多重句号；简单清理
    text = re.sub(r"。{2,}", "。", text).strip()
    return clip_text(text, cap)
