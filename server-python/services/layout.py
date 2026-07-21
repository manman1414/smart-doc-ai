# 作者：yangkunpeng1
# 日期：2026-07-21
"""
PDF 版面分析（基于 PyMuPDF 几何信息，无额外版面模型）。

- 文字块：数字层 text block
- 图片区：嵌入图 / image block
- 供解析层决定：整页 OCR / 仅数字层 / 图文混排（文字层 + 图片区 OCR）
"""

from __future__ import annotations

import re
from typing import Any

# 图片区域面积占页面积达到该比例才单独 OCR（忽略小图标）
MIN_IMAGE_REGION_RATIO = 0.04
# 单图接近整页时，由上层走「整页扫描 OCR」，不再按区域切
NEAR_FULL_PAGE_RATIO = 0.85


def _block_text(block: dict) -> str:
    parts: list[str] = []
    for line in block.get("lines") or []:
        spans = line.get("spans") or []
        line_txt = "".join(str(s.get("text") or "") for s in spans).strip()
        if line_txt:
            parts.append(line_txt)
    return "\n".join(parts).strip()


def _area_ratio(bbox: tuple | list, page_area: float) -> float:
    if page_area <= 1e-6 or not bbox or len(bbox) < 4:
        return 0.0
    x0, y0, x1, y1 = bbox[:4]
    a = abs(float(x1 - x0) * float(y1 - y0))
    return max(0.0, min(1.0, a / page_area))


def _merge_overlapping_regions(regions: list[dict], iou_thresh: float = 0.5) -> list[dict]:
    """粗略合并高度重叠的图片框，减少重复 OCR。"""
    if not regions:
        return []
    ordered = sorted(regions, key=lambda r: r.get("area_ratio", 0), reverse=True)
    kept: list[dict] = []

    def iou(a: list, b: list) -> float:
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        ix0, iy0 = max(ax0, bx0), max(ay0, by0)
        ix1, iy1 = min(ax1, bx1), min(ay1, by1)
        iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = abs((ax1 - ax0) * (ay1 - ay0))
        area_b = abs((bx1 - bx0) * (by1 - by0))
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    for reg in ordered:
        bbox = list(reg["bbox"])
        if any(iou(bbox, list(k["bbox"])) >= iou_thresh for k in kept):
            continue
        kept.append(reg)
    # 阅读顺序：先上后下、先左后右
    kept.sort(key=lambda r: (round(float(r["bbox"][1]) / 8.0), float(r["bbox"][0])))
    return kept


def analyze_page(page: Any) -> dict:
    """
    分析单页版面。

    返回:
      {
        "text_blocks": [{"bbox", "text", "y", "x", "kind"}],
        "image_regions": [{"bbox", "area_ratio", "y", "x", "kind"}],
        "text_chars": int,
        "max_image_ratio": float,
        "page_rect": (x0,y0,x1,y1),
      }
    """
    rect = page.rect
    page_area = abs(float(rect.width) * float(rect.height))
    page_rect = (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))

    text_blocks: list[dict] = []
    image_regions: list[dict] = []

    try:
        blocks = page.get_text("dict").get("blocks") or []
    except Exception:
        blocks = []

    for block in blocks:
        btype = block.get("type")
        bbox = block.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        if btype == 0:
            text = _block_text(block)
            if text:
                text_blocks.append(
                    {
                        "kind": "text",
                        "bbox": tuple(bbox),
                        "text": text,
                        "y": float(bbox[1]),
                        "x": float(bbox[0]),
                    }
                )
        elif btype == 1:
            ratio = _area_ratio(bbox, page_area)
            if ratio >= MIN_IMAGE_REGION_RATIO:
                image_regions.append(
                    {
                        "kind": "image",
                        "bbox": tuple(float(v) for v in bbox[:4]),
                        "area_ratio": ratio,
                        "y": float(bbox[1]),
                        "x": float(bbox[0]),
                    }
                )

    # get_image_info 补充（部分 PDF 不走 dict image block）
    try:
        infos = page.get_image_info(hashes=False) or []
    except Exception:
        infos = []
    for info in infos:
        if not isinstance(info, dict):
            continue
        bbox = info.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        ratio = _area_ratio(bbox, page_area)
        if ratio < MIN_IMAGE_REGION_RATIO:
            continue
        image_regions.append(
            {
                "kind": "image",
                "bbox": tuple(float(v) for v in bbox[:4]),
                "area_ratio": ratio,
                "y": float(bbox[1]),
                "x": float(bbox[0]),
            }
        )

    image_regions = _merge_overlapping_regions(image_regions)
    text_blocks.sort(key=lambda b: (round(b["y"] / 8.0), b["x"]))

    text_chars = sum(len(b["text"]) for b in text_blocks)
    max_image_ratio = max((r["area_ratio"] for r in image_regions), default=0.0)

    return {
        "text_blocks": text_blocks,
        "image_regions": image_regions,
        "text_chars": text_chars,
        "max_image_ratio": max_image_ratio,
        "page_rect": page_rect,
    }


def choose_strategy(
    digital_text: str,
    image_coverage: float,
    layout: dict,
    *,
    text_rich_chars: int = 80,
    text_sparse_chars: int = 40,
    image_cover_scan: float = 0.35,
    image_cover_empty: float = 0.15,
) -> tuple[str, str]:
    """
    选择解析策略: full_ocr | hybrid | digital
    返回 (strategy, reason)
    """
    text_len = len((digital_text or "").strip())
    cover = max(0.0, min(1.0, float(image_coverage or 0.0)))
    regions = layout.get("image_regions") or []
    max_img = float(layout.get("max_image_ratio") or 0.0)

    # 整页扫描：无字/少字 + 大图
    if text_len == 0 and cover >= image_cover_empty:
        return "full_ocr", f"整页扫描 text_len=0 cover={cover:.0%}"
    if text_len < text_sparse_chars and cover >= image_cover_scan:
        return "full_ocr", f"稀疏文字+高覆盖 text_len={text_len} cover={cover:.0%}"
    if text_len < text_sparse_chars and max_img >= NEAR_FULL_PAGE_RATIO:
        return "full_ocr", f"近整页大图 max_img={max_img:.0%}"

    # 图文混排：数字层够用，但仍有值得 OCR 的插图区域
    ocrable = [r for r in regions if r["area_ratio"] < NEAR_FULL_PAGE_RATIO]
    if text_len >= text_rich_chars and ocrable:
        return (
            "hybrid",
            f"图文混排 text_len={text_len} image_regions={len(ocrable)}",
        )
    # 数字层一般，但有中等插图
    if text_len > 0 and ocrable and max(r["area_ratio"] for r in ocrable) >= MIN_IMAGE_REGION_RATIO:
        return (
            "hybrid",
            f"有正文+插图 text_len={text_len} image_regions={len(ocrable)}",
        )

    if text_len > 0:
        return "digital", f"仅数字层 text_len={text_len} cover={cover:.0%}"
    if ocrable:
        return "hybrid", f"无数字层但有局部图 regions={len(ocrable)}"
    return "digital", f"无有效内容 text_len=0 cover={cover:.0%}"


def merge_reading_order(items: list[dict]) -> str:
    """按阅读顺序合并 {y,x,text} 块，并去掉重复行。"""
    usable = [it for it in items if (it.get("text") or "").strip()]
    usable.sort(
        key=lambda it: (
            round(float(it.get("y") or 0) / 8.0),
            float(it.get("x") or 0),
        )
    )
    raw = "\n\n".join(str(it["text"]).strip() for it in usable)
    return dedupe_lines(raw)


def dedupe_lines(text: str) -> str:
    """去掉页面内重复行（版面块 / OCR 常见重复）。"""
    if not text:
        return ""
    from difflib import SequenceMatcher

    def norm(s: str) -> str:
        return re.sub(r"\s+", "", s.strip())

    kept: list[str] = []
    seen_norm: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            if kept and kept[-1] != "":
                kept.append("")
            continue
        n = norm(s)
        if not n:
            continue
        if any(
            n == sn or n in sn or sn in n or SequenceMatcher(None, n, sn).ratio() >= 0.9
            for sn in seen_norm
        ):
            continue
        seen_norm.append(n)
        kept.append(s)
    # 压缩多余空行
    out: list[str] = []
    for line in kept:
        if line == "" and (not out or out[-1] == ""):
            continue
        out.append(line)
    return "\n".join(out).strip()
