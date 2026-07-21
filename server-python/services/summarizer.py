# 作者：yangkunpeng1
# 日期：2026-07-21
"""
文档摘要：短文直接总结；长文分段摘要再合并（map-reduce）。
"""

from __future__ import annotations

import os
import re
from typing import Callable

CancelCheck = Callable[[], bool]
LlmComplete = Callable[[str], str]

_PAGE_SPLIT = re.compile(r"(?=【第\d+页】)")


def _section_chars() -> int:
    raw = os.environ.get("SMARTDOC_SUMMARY_SECTION_CHARS", "2500")
    try:
        return max(800, int(raw))
    except ValueError:
        return 2500


def _max_sections() -> int:
    raw = os.environ.get("SMARTDOC_SUMMARY_MAX_SECTIONS", "8")
    try:
        return max(1, min(20, int(raw)))
    except ValueError:
        return 8


def split_for_summary(
    text: str,
    section_size: int | None = None,
    max_sections: int | None = None,
) -> list[str]:
    """
    将长文切成若干段供摘要。
    优先按【第N页】切开再装箱；段数过多时均匀抽样，保证覆盖开头/中间/结尾。
    """
    raw = (text or "").strip()
    if not raw:
        return []

    size = section_size if section_size is not None else _section_chars()
    limit = max_sections if max_sections is not None else _max_sections()

    if len(raw) <= size:
        return [raw]

    parts = [p.strip() for p in _PAGE_SPLIT.split(raw) if p and p.strip()]
    if len(parts) <= 1:
        # 无页码标记：按定长切（尽量在换行处）
        parts = _hard_split_prefer_newline(raw, size)

    buckets: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for p in parts:
        add = len(p) + (1 if cur else 0)
        if cur and cur_len + add > size:
            buckets.append("\n".join(cur).strip())
            cur = [p]
            cur_len = len(p)
        else:
            cur.append(p)
            cur_len += add
    if cur:
        buckets.append("\n".join(cur).strip())

    buckets = [b for b in buckets if b]
    if not buckets:
        return [raw[:size]]

    if len(buckets) <= limit:
        return buckets

    # 均匀抽样
    n = limit
    if n == 1:
        return [buckets[0]]
    indices = [int(round(i * (len(buckets) - 1) / (n - 1))) for i in range(n)]
    uniq: list[int] = []
    for i in indices:
        if not uniq or i != uniq[-1]:
            uniq.append(i)
    return [buckets[i] for i in uniq]


def _hard_split_prefer_newline(text: str, size: int) -> list[str]:
    out: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            window = text[start:end]
            cut = window.rfind("\n\n")
            if cut < size // 3:
                cut = window.rfind("\n")
            if cut >= size // 3:
                end = start + cut + 1
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        start = end
    return out or [text[:size]]


def build_section_prompt(section: str, index: int, total: int) -> str:
    return (
        f"以下是文档的第 {index}/{total} 部分，请用中文概括要点（80字以内，不要开场白）：\n"
        f"---\n{section}\n---\n要点："
    )


def build_merge_prompt(partials: list[str]) -> str:
    numbered = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(partials))
    return (
        "下面是同一文档不同部分的要点，请合并成一份完整中文摘要（200字以内），"
        "覆盖主要结论与关键信息，不要重复罗列：\n"
        f"---\n{numbered}\n---\n摘要："
    )


def build_single_prompt(text: str) -> str:
    return (
        "请用中文简要总结以下文档的核心内容（200字以内）：\n"
        f"---\n{text}\n---\n摘要："
    )


def summarize_document(
    text: str,
    llm_complete: LlmComplete,
    cancel_check: CancelCheck | None = None,
) -> str:
    """
    对全文做摘要。短文一次完成；长文先分段再合并。
    llm_complete(prompt) -> 模型回复文本
    """
    raw = (text or "").strip()
    if not raw:
        return ""

    def cancelled() -> bool:
        return bool(cancel_check and cancel_check())

    sections = split_for_summary(raw)
    print(
        f"[SUMMARY] plan chars={len(raw)} sections={len(sections)} "
        f"sizes={[len(s) for s in sections]}",
        flush=True,
    )

    if len(sections) == 1:
        if cancelled():
            return ""
        return (llm_complete(build_single_prompt(sections[0])) or "").strip()

    partials: list[str] = []
    total = len(sections)
    for i, sec in enumerate(sections, start=1):
        if cancelled():
            return ""
        part = (llm_complete(build_section_prompt(sec, i, total)) or "").strip()
        if part:
            partials.append(part)
        print(f"[SUMMARY] section {i}/{total} ok chars={len(part)}", flush=True)

    if cancelled():
        return ""
    if not partials:
        return ""
    if len(partials) == 1:
        return partials[0]

    merged = (llm_complete(build_merge_prompt(partials)) or "").strip()
    print(f"[SUMMARY] merge done chars={len(merged)}", flush=True)
    return merged or "\n".join(partials)
