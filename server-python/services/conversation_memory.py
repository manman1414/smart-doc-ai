# 作者：yangkunpeng1
# 日期：2026-07-22
"""
多轮对话记忆：最近 N 轮原文 + 更早内容滚动摘要 + 提问改写（供检索）。
"""

from __future__ import annotations

import os
from typing import Callable, Sequence


ChatFn = Callable[[list[dict], int], str]


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


def merge_memory_summary(
    existing: str,
    older_messages: Sequence[dict],
    chat_fn: ChatFn,
    *,
    max_chars: int = 1200,
) -> str:
    """
    把「已有摘要 + 更早对话」压成新的滚动摘要。
    chat_fn(messages, max_tokens) -> 文本
    """
    older = normalize_history(older_messages)
    if not older:
        return (existing or "").strip()

    prev = (existing or "").strip()
    block = format_turns_text(older)
    prompt = (
        "你是对话记忆压缩助手。请把「已有摘要」与「新增更早对话」合并成一段中文摘要。\n"
        "要求：保留用户关注点、已确认事实、未决问题与关键指代；删除寒暄与重复；"
        f"控制在 {max_chars} 字以内；只输出摘要正文。\n\n"
        f"【已有摘要】\n{prev or '（无）'}\n\n"
        f"【新增更早对话】\n{block}\n\n"
        "摘要："
    )
    try:
        text = (chat_fn([{"role": "user", "content": prompt}], 400) or "").strip()
    except Exception as e:
        print(f"[MEMORY] merge_summary failed: {e}", flush=True)
        # 降级：保留旧摘要 + 截断旧对话
        fallback = (prev + "\n" + block).strip() if prev else block
        return fallback[:max_chars]

    if not text:
        fallback = (prev + "\n" + block).strip() if prev else block
        return fallback[:max_chars]
    return text[:max_chars]


def rewrite_question(
    question: str,
    *,
    memory_summary: str,
    recent_messages: Sequence[dict],
    chat_fn: ChatFn,
) -> str:
    """
    结合摘要与最近原文，把追问改写成可独立检索的完整问句。
    失败时回退原问题。
    """
    q = (question or "").strip()
    if not q:
        return q

    summary = (memory_summary or "").strip()
    recent = normalize_history(recent_messages)
    if not summary and not recent:
        return q

    recent_txt = format_turns_text(recent) if recent else "（无）"
    prompt = (
        "根据对话摘要与最近对话，把「当前问题」改写成一句独立、完整、可检索的中文问句。\n"
        "要求：补全省略与指代；不要回答问题；不要解释；只输出改写后的问句。\n\n"
        f"【对话摘要】\n{summary or '（无）'}\n\n"
        f"【最近对话】\n{recent_txt}\n\n"
        f"【当前问题】\n{q}\n\n"
        "改写问句："
    )
    try:
        text = (chat_fn([{"role": "user", "content": prompt}], 120) or "").strip()
    except Exception as e:
        print(f"[MEMORY] rewrite_question failed: {e}", flush=True)
        return q

    # 去掉常见包裹
    text = text.strip().strip("「」\"'")
    if not text or len(text) > 500:
        return q
    return text


def build_history_prompt_messages(
    memory_summary: str,
    recent_messages: Sequence[dict],
) -> list[dict]:
    """拼进最终 LLM 的历史部分（不含本题文档与问题）。"""
    out: list[dict] = []
    summary = (memory_summary or "").strip()
    if summary:
        out.append(
            {
                "role": "system",
                "content": f"以下是更早多轮对话的压缩摘要，供连贯回答参考：\n{summary}",
            }
        )
    for m in normalize_history(recent_messages):
        out.append({"role": m["role"], "content": m["content"]})
    return out
