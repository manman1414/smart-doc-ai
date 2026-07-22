# 作者：Auto
# 日期：2026-07-21
"""从云启智能 PDF 抽文本：先 pymupdf，空页则 RapidOCR（优先 1-35 + 目录页）。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PDF = Path(r"c:\Users\Administrator\Desktop\云启智能-中国企业级智能文档处理市场研究报告2026.pdf")
OUT = Path(__file__).with_name("_yunqi_pdf_extract.txt")
META = Path(__file__).with_name("_yunqi_pdf_extract_meta.txt")
PAGES_DIR = Path(__file__).with_name("_yunqi_pages")
DONE_FLAG = Path(__file__).with_name("_yunqi_extract_done.flag")

# 150dpi ≈ 150/72 scale
RENDER_SCALE = 150 / 72.0
MAX_OCR_PAGES = 35  # 1-indexed inclusive thorough OCR


def _import_fitz():
    try:
        import pymupdf as fitz

        return fitz
    except ImportError:
        import fitz  # type: ignore

        return fitz


def _looks_like_toc(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    keys = ("目录", "CONTENTS", "Contents", "目 录")
    return any(k in t for k in keys)


def _ocr_page(fitz_mod, doc, page_index: int, ocr_engine, save_png: bool) -> str:
    from services.parser import _ocr_result_to_text

    page = doc.load_page(page_index)
    pix = page.get_pixmap(matrix=fitz_mod.Matrix(RENDER_SCALE, RENDER_SCALE), alpha=False)
    if save_png:
        PAGES_DIR.mkdir(parents=True, exist_ok=True)
        png_path = PAGES_DIR / f"page_{page_index + 1:03d}.png"
        pix.save(str(png_path))

    import numpy as np

    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = img[:, :, :3]
    out = ocr_engine(img)
    return _ocr_result_to_text(out)


def main() -> None:
    t0 = time.time()
    fitz = _import_fitz()
    doc = fitz.open(str(PDF))
    page_count = doc.page_count

    digital: list[str] = []
    non_empty = 0
    for i in range(page_count):
        t = doc.load_page(i).get_text("text") or ""
        digital.append(t)
        if t.strip():
            non_empty += 1

    scanned = non_empty < max(3, page_count // 5)
    ocr_pages: list[int] = []
    backend = "pymupdf"
    parts: list[str] = []

    if not scanned:
        for i, t in enumerate(digital):
            parts.append(f"===== PAGE {i + 1} =====\n{t}")
    else:
        backend = "pymupdf+rapidocr"
        from services.parser import _get_ocr_engine

        engine = _get_ocr_engine()
        if engine is None:
            backend = "pymupdf(scanned,OCR_UNAVAILABLE)"
            for i, t in enumerate(digital):
                parts.append(f"===== PAGE {i + 1} =====\n{t}")
        else:
            # 先 OCR 1..MAX_OCR_PAGES；另扫可能目录页（数字文本里含目录，或前几页已 OCR）
            candidates = set(range(1, min(MAX_OCR_PAGES, page_count) + 1))
            # 也尝试 36..min(45) 中像目录的页：先用低成本数字文本判断；空则靠页码启发
            for i in range(page_count):
                if _looks_like_toc(digital[i]):
                    candidates.add(i + 1)
            # 常见目录位置：第 2-8 页（已在 1-35）

            # 渲染前 40 页 PNG（便于人工核对）；OCR 仍按 candidates
            render_n = min(40, page_count)
            final_texts = list(digital)

            for i in range(page_count):
                page_no = i + 1
                need_ocr = page_no in candidates
                save_png = page_no <= render_n
                if not need_ocr and not save_png:
                    parts.append(f"===== PAGE {page_no} =====\n{final_texts[i]}")
                    continue
                if need_ocr:
                    print(f"[YUNQI] OCR page {page_no}/{page_count} ...", flush=True)
                    try:
                        ot = _ocr_page(fitz, doc, i, engine, save_png=save_png)
                        final_texts[i] = ot
                        ocr_pages.append(page_no)
                        print(f"[YUNQI] OCR page {page_no} chars={len(ot)}", flush=True)
                    except Exception as e:
                        print(f"[YUNQI] OCR page {page_no} FAILED: {e}", flush=True)
                elif save_png:
                    try:
                        page = doc.load_page(i)
                        pix = page.get_pixmap(
                            matrix=fitz.Matrix(RENDER_SCALE, RENDER_SCALE), alpha=False
                        )
                        PAGES_DIR.mkdir(parents=True, exist_ok=True)
                        pix.save(str(PAGES_DIR / f"page_{page_no:03d}.png"))
                    except Exception as e:
                        print(f"[YUNQI] render page {page_no} FAILED: {e}", flush=True)
                parts.append(f"===== PAGE {page_no} =====\n{final_texts[i]}")

    doc.close()
    text = "\n\n".join(parts)
    OUT.write_text(text, encoding="utf-8")

    # 非空页统计
    filled = sum(1 for p in parts if len(p.split("\n", 1)[-1].strip()) > 0)
    elapsed = time.time() - t0
    meta = (
        f"written={OUT}\n"
        f"page_count={page_count}\n"
        f"total_chars={len(text)}\n"
        f"filled_pages={filled}\n"
        f"digital_non_empty={non_empty}\n"
        f"scanned={scanned}\n"
        f"backend={backend}\n"
        f"ocr_pages={ocr_pages}\n"
        f"ocr_page_count={len(ocr_pages)}\n"
        f"pages_dir={PAGES_DIR if PAGES_DIR.exists() else ''}\n"
        f"elapsed_sec={elapsed:.1f}\n"
        f"pdf={PDF}\n"
    )
    META.write_text(meta, encoding="utf-8")
    DONE_FLAG.write_text(f"ok {time.strftime('%Y-%m-%d %H:%M:%S')}\n{meta}", encoding="utf-8")
    print(meta, flush=True)


if __name__ == "__main__":
    main()
