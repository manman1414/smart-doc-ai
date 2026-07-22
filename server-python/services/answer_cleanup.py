# 作者：yangkunpeng1
# 日期：2026-07-22
"""
回答去重：去掉完全重复的段落/行/连续相同句。

只删「一模一样」的复读，保留内容不同的要点。
"""

from __future__ import annotations

import re
import unicodedata

_PAGE_CITE = re.compile(
    r"[（(]第\s*\d+(?:\s*[、,，\-–—]\s*\d+)*\s*页[）)]"
)
_ZW = re.compile(r"[\u200b\u200c\u200d\ufeff\u00a0]")
_WS = re.compile(r"\s+")
_SENT_END = re.compile(r"(?<=[。！？；])")


def _norm_key(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = _ZW.sub("", s)
    s = _PAGE_CITE.sub("", s)
    s = _WS.sub("", s).strip()
    s = re.sub(r"^[\-\*\d\.、•]+", "", s)
    return s


def _collapse_consecutive_sentences(text: str) -> str:
    """去掉连续重复的句子（同一句连说两遍）。"""
    parts = _SENT_END.split(text or "")
    kept: list[str] = []
    prev_key = ""
    for p in parts:
        key = _norm_key(p)
        if key and key == prev_key:
            continue
        kept.append(p)
        if key:
            prev_key = key
    return "".join(kept)


def collapse_duplicate_lines(text: str) -> str:
    """
    1) 连续相同句去重
    2) 按行去重（忽略页码/空白/零宽字符差异）
    """
    if not (text or "").strip():
        return text or ""

    text = _collapse_consecutive_sentences(
        text.replace("\r\n", "\n").replace("\r", "\n")
    )

    kept: list[str] = []
    seen: set[str] = set()
    blank_run = 0
    for line in text.split("\n"):
        raw = line.rstrip()
        # 行内再清一次连续复句
        if raw.strip():
            raw = _collapse_consecutive_sentences(raw).rstrip()
        key = _norm_key(raw)
        if not key:
            blank_run += 1
            if blank_run <= 1:
                kept.append("")
            continue
        blank_run = 0
        if key in seen:
            continue
        seen.add(key)
        kept.append(raw)

    while kept and not kept[0].strip():
        kept.pop(0)
    while kept and not kept[-1].strip():
        kept.pop()
    return "\n".join(kept)


def is_highly_repetitive(text: str, *, min_lines: int = 4, dup_ratio: float = 0.25) -> bool:
    lines = [ln for ln in (text or "").splitlines() if _norm_key(ln)]
    if len(lines) < min_lines:
        return False
    keys = [_norm_key(ln) for ln in lines]
    unique = len(set(keys))
    return (len(keys) - unique) / max(len(keys), 1) >= dup_ratio
