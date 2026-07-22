# 作者：yangkunpeng1
# 日期：2026-07-21
"""
文本分块：

- structure：有表格/图内文字等结构标记时 → 结构单元 + 句子装箱
- recursive：普通 TXT、以及无结构标记的文字页 → 递归字符分块
- PDF 按页选型（auto）；碎块合并；overlap 不突破 chunk_size
"""

from __future__ import annotations

import os
import re
from typing import Iterable

# 递归字符分隔符：从粗到细（含中文）
_RECURSIVE_SEPARATORS = [
    "\n\n",
    "\n",
    "。",
    "！",
    "？",
    "；",
    ". ",
    "! ",
    "? ",
    "; ",
    "，",
    "、",
    ", ",
    " ",
    "",
]


def _chunk_size_default() -> int:
    raw = os.environ.get("SMARTDOC_CHUNK_SIZE", "300")
    try:
        return max(100, int(raw))
    except ValueError:
        return 300


def _chunk_overlap_default() -> int:
    """默认 overlap=50；可用 SMARTDOC_CHUNK_OVERLAP 覆盖。"""
    raw = os.environ.get("SMARTDOC_CHUNK_OVERLAP", "50")
    try:
        return max(0, int(raw))
    except ValueError:
        return 50


def _min_chunk_ratio() -> float:
    """过短块阈值：len < chunk_size * ratio 则并入相邻块。"""
    raw = os.environ.get("SMARTDOC_CHUNK_MIN_RATIO", "0.3")
    try:
        return min(1.0, max(0.0, float(raw)))
    except ValueError:
        return 0.3


def _strategy_default() -> str:
    """auto | structure | recursive；可用环境变量覆盖。"""
    return (os.environ.get("SMARTDOC_CHUNK_STRATEGY") or "auto").strip().lower()


_STRUCT_HEAD = re.compile(
    r"^(【图内文字\s*[^\】]*】|【表格\s*[^\】]*】|【图像表格\s*[^\】]*】)"
)
_FIGURE_HEAD = re.compile(r"^【图内文字\s*([^\】]*)】")
_SENT_SPLIT = re.compile(r"(?<=[。！？；.!?;])\s*")
_HAS_STRUCTURE = re.compile(
    r"【图内文字|【表格|【图像表格|^\s*\|.+\|\s*$",
    re.M,
)


def _parse_figure_id(title_or_text: str) -> str:
    """从【图内文字 …】标题解析稳定 ID；无则空串。"""
    first = (title_or_text or "").strip().split("\n", 1)[0].strip()
    m = _FIGURE_HEAD.match(first)
    if not m:
        return ""
    return (m.group(1) or "").strip()


def _figure_marker(figure_id: str) -> str:
    fid = (figure_id or "").strip()
    return f"【图内文字 {fid}】" if fid else "【图内文字】"


def _stamp_figure_chunks(
    pieces: list[dict],
    figure_id: str,
) -> list[dict]:
    """给同一张图拆出的每一段都打上同一 figure_id 标记（正文+元数据）。"""
    if not pieces:
        return []
    marker = _figure_marker(figure_id)
    fid = (figure_id or "").strip()
    out: list[dict] = []
    for p in pieces:
        body = (p.get("text") or "").strip()
        if not body:
            continue
        # 去掉旧标题，避免重复叠加
        if _FIGURE_HEAD.match(body.split("\n", 1)[0].strip()):
            rest = body.split("\n", 1)
            body = rest[1].strip() if len(rest) > 1 else ""
        if not body:
            continue
        item = {
            "text": f"{marker}\n{body}",
            "kind": "figure",
        }
        if fid:
            item["figure_id"] = fid
        out.append(item)
    return out


def resolve_strategy(strategy: str | None, sample_text: str = "", source_ext: str = "") -> str:
    """
    解析最终策略。
    - 显式 structure / recursive
    - auto：.txt → recursive；有结构标记 → structure；否则 recursive
    PDF 建议按页传入 sample_text，实现「有表/图的页 structure、纯文字页 recursive」。
    """
    s = (strategy or _strategy_default() or "auto").strip().lower()
    if s in ("structure", "recursive"):
        return s
    # auto
    ext = (source_ext or "").lower().lstrip(".")
    if ext == "txt":
        return "recursive"
    if sample_text and _HAS_STRUCTURE.search(sample_text):
        return "structure"
    return "recursive"


def _merge_tiny_chunks(
    chunks: list[str],
    chunk_size: int,
    min_ratio: float | None = None,
) -> list[str]:
    """把过短碎块并入前一块；末块过短则并入上一块。"""
    ratio = _min_chunk_ratio() if min_ratio is None else min_ratio
    if ratio <= 0 or not chunks:
        return [c for c in chunks if c and c.strip()]
    min_len = max(1, int(chunk_size * ratio))
    out: list[str] = []
    for c in chunks:
        c = (c or "").strip()
        if not c:
            continue
        if out and len(c) < min_len:
            out[-1] = (out[-1] + c).strip()
        else:
            out.append(c)
    if len(out) >= 2 and len(out[-1]) < min_len:
        tail = out.pop()
        out[-1] = (out[-1] + tail).strip()
    return out


def _merge_tiny_units(
    units: list[dict],
    chunk_size: int,
    min_ratio: float | None = None,
) -> list[dict]:
    """同 kind 的过短单元合并；不跨 table/figure/text；不同 figure_id 不合并。"""
    ratio = _min_chunk_ratio() if min_ratio is None else min_ratio
    if ratio <= 0 or not units:
        return units
    min_len = max(1, int(chunk_size * ratio))
    out: list[dict] = []
    for u in units:
        text = (u.get("text") or "").strip()
        if not text:
            continue
        kind = u.get("kind") or "text"
        fid = str(u.get("figure_id") or "").strip()
        item = {"text": text, "kind": kind}
        if fid:
            item["figure_id"] = fid
        same_fig = (out[-1].get("figure_id") or "") == fid if out else False
        if (
            out
            and len(text) < min_len
            and (out[-1].get("kind") or "text") == kind
            and (kind != "figure" or same_fig)
        ):
            out[-1]["text"] = (out[-1]["text"] + "\n" + text).strip()
        else:
            out.append(item)
    if (
        len(out) >= 2
        and len(out[-1]["text"]) < min_len
        and (out[-2].get("kind") or "text") == (out[-1].get("kind") or "text")
        and (
            (out[-1].get("kind") or "") != "figure"
            or (out[-2].get("figure_id") or "") == (out[-1].get("figure_id") or "")
        )
    ):
        tail = out.pop()
        out[-1]["text"] = (out[-1]["text"] + "\n" + tail["text"]).strip()
    return out


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
    strategy: str | None = None,
) -> list[str]:
    """兼容旧接口：只返回文本列表。"""
    size = chunk_size if chunk_size is not None else _chunk_size_default()
    ov = overlap if overlap is not None else _chunk_overlap_default()
    return [
        c["text"]
        for c in chunk_text_units(text, chunk_size=size, overlap=ov, strategy=strategy)
    ]


def chunk_text_units(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
    strategy: str | None = None,
    source_ext: str = "",
) -> list[dict]:
    """
    对单段文本分块。
    返回 [{"text", "kind": text|table|figure}, ...]
    """
    size = chunk_size if chunk_size is not None else _chunk_size_default()
    ov = overlap if overlap is not None else _chunk_overlap_default()
    raw = (text or "").strip()
    if not raw:
        return []
    mode = resolve_strategy(strategy, sample_text=raw, source_ext=source_ext)
    if mode == "recursive":
        units = [
            {"text": t, "kind": "text"}
            for t in chunk_text_recursive(raw, chunk_size=size, overlap=ov)
        ]
    else:
        units = _pack_units(
            list(_iter_structural_units(raw)),
            chunk_size=size,
            overlap=ov,
        )
    return _merge_tiny_units(units, size)


def chunk_text_recursive(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
    separators: list[str] | None = None,
) -> list[str]:
    """
    递归字符分块（RecursiveCharacterTextSplitter 同思路）。
    按分隔符从粗到细切分，超长块继续用更细分隔符。
    """
    size = chunk_size if chunk_size is not None else _chunk_size_default()
    ov = overlap if overlap is not None else _chunk_overlap_default()
    raw = (text or "").strip()
    if not raw:
        return []
    if len(raw) <= size:
        return [raw]
    seps = separators if separators is not None else list(_RECURSIVE_SEPARATORS)
    # 递归内部不做 overlap，避免层层叠加；最后统一回带（且不突破 size）
    parts = _recursive_split(raw, seps, size, overlap=0)
    parts = _apply_overlap(parts, ov, size) if ov > 0 else parts
    return _merge_tiny_chunks(parts, size)


def _recursive_split(
    text: str,
    separators: list[str],
    chunk_size: int,
    overlap: int,
) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    seps = list(separators) if separators else [""]
    separator = seps[0]
    rest = seps[1:] if len(seps) > 1 else [""]

    if separator == "":
        return _hard_cut_with_overlap(text, chunk_size, overlap)

    if separator not in text:
        return _recursive_split(text, rest, chunk_size, overlap)

    pieces = _split_keep_sep(text, separator)
    merged = _merge_pieces(pieces, chunk_size)
    final: list[str] = []
    for piece in merged:
        piece = piece.strip()
        if not piece:
            continue
        if len(piece) <= chunk_size:
            final.append(piece)
        else:
            final.extend(_recursive_split(piece, rest, chunk_size, overlap))
    return final


def _split_keep_sep(text: str, sep: str) -> list[str]:
    if not sep:
        return [text]
    parts = text.split(sep)
    if len(parts) == 1:
        return parts
    out: list[str] = []
    for i, p in enumerate(parts):
        if i < len(parts) - 1:
            out.append(p + sep)
        else:
            if p:
                out.append(p)
    return out


def _merge_pieces(pieces: list[str], chunk_size: int) -> list[str]:
    """把小片尽量合并到不超过 chunk_size。"""
    merged: list[str] = []
    cur = ""
    for p in pieces:
        if not p:
            continue
        if not cur:
            cur = p
            continue
        if len(cur) + len(p) <= chunk_size:
            cur = cur + p
        else:
            merged.append(cur)
            cur = p
    if cur:
        merged.append(cur)
    return merged


def _hard_cut_with_overlap(text: str, chunk_size: int, overlap: int) -> list[str]:
    step = max(1, chunk_size - max(0, overlap))
    out: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        out.append(text[start:end])
        if end >= len(text):
            break
        start += step
    return out


def _apply_overlap(chunks: list[str], overlap: int, chunk_size: int) -> list[str]:
    """
    在已切好的块之间做字符级 overlap，且保证 len(chunk) <= chunk_size。
    若当前块已接近上限，则少带或不带上一块尾巴。
    """
    if overlap <= 0 or len(chunks) <= 1:
        return chunks
    out = [chunks[0]]
    for i in range(1, len(chunks)):
        cur = chunks[i]
        prev = chunks[i - 1]
        if not cur:
            continue
        room = max(0, chunk_size - len(cur))
        take = min(overlap, room, len(prev))
        if take <= 0:
            out.append(cur)
            continue
        prev_tail = prev[-take:]
        if cur.startswith(prev_tail):
            out.append(cur)
        else:
            out.append((prev_tail + cur).strip())
    return out


def chunk_pages(
    pages: list[dict],
    chunk_size: int | None = None,
    overlap: int | None = None,
    strategy: str | None = None,
    source_ext: str = "",
) -> list[dict]:
    """
    按页分块，保留页码与类型。

    strategy: auto | structure | recursive
    source_ext: 如 txt / pdf，供 auto 判断
    auto 下按页选型：该页有结构标记 → structure，否则 recursive
    """
    size = chunk_size if chunk_size is not None else _chunk_size_default()
    ov = overlap if overlap is not None else _chunk_overlap_default()

    out: list[dict] = []
    for item in pages or []:
        try:
            page_no = int(item.get("page") or 1)
        except (TypeError, ValueError):
            page_no = 1
        page_text = (item.get("text") or "").strip()
        if not page_text:
            continue
        page_mode = resolve_strategy(
            strategy, sample_text=page_text, source_ext=source_ext
        )
        page_units = chunk_text_units(
            page_text,
            chunk_size=size,
            overlap=ov,
            strategy=page_mode,
            source_ext=source_ext,
        )
        for unit in page_units:
            item = {
                "text": unit["text"],
                "page": page_no,
                "kind": unit.get("kind") or "text",
                "strategy": page_mode,
            }
            fid = str(unit.get("figure_id") or "").strip()
            if not fid and item["kind"] == "figure":
                fid = _parse_figure_id(item["text"])
            # 旧格式「图内文字 1」补全为页级稳定 ID：p{page}-{n}
            if fid and re.fullmatch(r"\d+", fid):
                fid = f"p{page_no}-{fid}"
            if fid:
                item["figure_id"] = fid
                if item["kind"] == "figure":
                    stamped = _stamp_figure_chunks(
                        [{"text": item["text"], "kind": "figure"}], fid
                    )
                    if stamped:
                        item["text"] = stamped[0]["text"]
            out.append(item)
    return out


def _iter_structural_units(text: str) -> Iterable[dict]:
    """拆成结构单元：普通段落 / 表格 / 图内文字（带 figure_id）。"""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    buf: list[str] = []
    kind = "text"
    figure_id = ""
    i = 0
    n = len(lines)

    def flush():
        nonlocal buf, kind, figure_id
        body = "\n".join(buf).strip()
        prev_kind = kind
        prev_fid = figure_id
        buf = []
        kind = "text"
        figure_id = ""
        if body:
            item = {"text": body, "kind": prev_kind}
            if prev_kind == "figure":
                parsed = prev_fid or _parse_figure_id(body)
                if parsed:
                    item["figure_id"] = parsed
            yield item

    while i < n:
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("|"):
            yield from flush()
            table_lines = [line]
            i += 1
            while i < n and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            table = "\n".join(table_lines).strip()
            if table:
                yield {"text": table, "kind": "table"}
            continue

        m = _STRUCT_HEAD.match(stripped)
        if m:
            yield from flush()
            title = stripped
            if "图内" in title or "图像表格" in title:
                kind = "figure"
                figure_id = _parse_figure_id(title)
            else:
                kind = "table"
                figure_id = ""
            buf = [title]
            i += 1
            while i < n:
                ns = lines[i].strip()
                if _STRUCT_HEAD.match(ns) or ns.startswith("|"):
                    break
                if ns == "" and buf and buf[-1].strip() == "":
                    i += 1
                    break
                buf.append(lines[i])
                i += 1
            yield from flush()
            continue

        buf.append(line)
        i += 1

    yield from flush()


def _split_sentences(text: str) -> list[str]:
    t = (text or "").strip()
    if not t:
        return []
    rough: list[str] = []
    for para in re.split(r"\n+", t):
        para = para.strip()
        if not para:
            continue
        parts = [p.strip() for p in _SENT_SPLIT.split(para) if p and p.strip()]
        rough.extend(parts if parts else [para])
    return rough


def _pack_sentence_stream(
    sentences: list[str],
    *,
    kind: str,
    chunk_size: int,
    overlap: int,
    figure_id: str = "",
) -> list[dict]:
    """按句装箱。figure 带 figure_id 时预留标题长度，最后每段统一打标。"""
    fid = (figure_id or "").strip()
    marker = _figure_marker(fid) if (kind == "figure" and fid) else ""
    body_limit = chunk_size
    if marker:
        body_limit = max(32, chunk_size - len(marker) - 1)

    raw_chunks: list[dict] = []
    cur: list[str] = []
    cur_len = 0

    def flush():
        nonlocal cur, cur_len
        if not cur:
            return
        item = {"text": "\n".join(cur).strip(), "kind": kind}
        if fid and kind == "figure":
            item["figure_id"] = fid
        raw_chunks.append(item)
        cur = []
        cur_len = 0

    for sent in sentences:
        if not sent:
            continue
        if kind == "figure" and _FIGURE_HEAD.match(sent.strip()):
            if not fid:
                fid = _parse_figure_id(sent)
                marker = _figure_marker(fid) if fid else ""
                if marker:
                    body_limit = max(32, chunk_size - len(marker) - 1)
            continue
        if len(sent) > body_limit:
            flush()
            for piece in _hard_split(sent, body_limit):
                item = {"text": piece, "kind": kind}
                if fid and kind == "figure":
                    item["figure_id"] = fid
                raw_chunks.append(item)
            continue
        add = len(sent) + (1 if cur else 0)
        if cur and cur_len + add > body_limit:
            flush()
            if overlap > 0 and raw_chunks:
                prev_text = raw_chunks[-1]["text"]
                sep = "\n"
                room = max(0, body_limit - len(sent) - len(sep))
                take = min(overlap, room, len(prev_text))
                if take > 0:
                    carry = prev_text[-take:]
                    cur = [carry, sent]
                    cur_len = len(carry) + len(sep) + len(sent)
                else:
                    cur = [sent]
                    cur_len = len(sent)
            else:
                cur = [sent]
                cur_len = len(sent)
        else:
            cur.append(sent)
            cur_len = len("\n".join(cur))
    flush()

    if kind == "figure" and fid:
        return _stamp_figure_chunks(raw_chunks, fid)
    return raw_chunks


def _pack_units(
    units: list[dict],
    *,
    chunk_size: int,
    overlap: int,
) -> list[dict]:
    out: list[dict] = []
    for unit in units:
        utext = (unit.get("text") or "").strip()
        ukind = unit.get("kind") or "text"
        if not utext:
            continue

        if ukind == "table":
            if len(utext) <= chunk_size:
                out.append({"text": utext, "kind": "table"})
            else:
                for piece in _split_long_table(utext, chunk_size):
                    out.append({"text": piece, "kind": "table"})
            continue

        fid = str(unit.get("figure_id") or "").strip() or _parse_figure_id(utext)
        body = utext
        if ukind == "figure" and _FIGURE_HEAD.match(utext.split("\n", 1)[0].strip()):
            parts = utext.split("\n", 1)
            body = parts[1].strip() if len(parts) > 1 else ""
            if not fid:
                fid = _parse_figure_id(parts[0])
        if not body:
            continue
        sents = _split_sentences(body)
        out.extend(
            _pack_sentence_stream(
                sents,
                kind=ukind if ukind in ("text", "figure") else "text",
                chunk_size=chunk_size,
                overlap=overlap,
                figure_id=fid if ukind == "figure" else "",
            )
        )
    return out


def _hard_split(text: str, chunk_size: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            window = text[start:end]
            cut = max(
                window.rfind("，"),
                window.rfind("、"),
                window.rfind(","),
                window.rfind(" "),
            )
            if cut >= chunk_size // 3:
                end = start + cut + 1
        piece = text[start:end].strip()
        if piece:
            parts.append(piece)
        start = end
    return parts


def _split_long_table(table: str, chunk_size: int) -> list[str]:
    rows = table.splitlines()
    if not rows:
        return []
    header: list[str] = []
    body = rows
    if len(rows) >= 2 and re.match(r"^\|?\s*:?-{3,}", rows[1].strip()):
        header = rows[:2]
        body = rows[2:]
    elif rows[0].strip().startswith("|"):
        header = [rows[0]]
        body = rows[1:]

    pieces: list[str] = []
    cur = list(header)
    cur_len = sum(len(x) + 1 for x in cur)

    def emit():
        nonlocal cur, cur_len
        if len(cur) > len(header):
            pieces.append("\n".join(cur).strip())
        cur = list(header)
        cur_len = sum(len(x) + 1 for x in cur)

    for row in body:
        add = len(row) + 1
        if cur_len + add > chunk_size and len(cur) > len(header):
            emit()
        cur.append(row)
        cur_len += add
    if len(cur) > len(header) or (header and not body):
        text = "\n".join(cur).strip()
        if text:
            pieces.append(text)
    return [p for p in pieces if p]
