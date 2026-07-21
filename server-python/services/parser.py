# 作者：yangkunpeng1
# 日期：2026-07-21
"""文档文本解析：PDF / TXT → 纯文本（表格结构化 + 扫描件 OCR + 按页进度）。"""

from __future__ import annotations

import os
import re
from html import unescape
from typing import Callable, Iterator, Optional

# 与历史约定一致：失败返回方括号错误串，供调用方识别
_ERROR_MARKERS = (
    "[错误：",
    "[PDF解析错误",
    "[TXT读取错误",
    "[不支持的文件类型",
)

SUPPORTED_EXTENSIONS = frozenset({"pdf", "txt"})

# —— OCR 判定（主流启发式：数字层字数 + 页面图像占比）——
_TEXT_RICH_CHARS = 80
_TEXT_SPARSE_CHARS = 40
_IMAGE_COVER_SCAN = 0.35
_IMAGE_COVER_EMPTY_PAGE = 0.15

# 页数上限（可用环境变量 SMARTDOC_MAX_PDF_PAGES 覆盖）
def _max_pdf_pages() -> int:
    raw = os.environ.get("SMARTDOC_MAX_PDF_PAGES", "120")
    try:
        n = int(raw)
        return max(1, n)
    except ValueError:
        return 120

ProgressCallback = Callable[[dict], None]
CancelCheck = Callable[[], bool]

_ocr_engine = None
_ocr_init_attempted = False
_ocr_backend = ""  # "rapidocr" | "legacy"

_table_engine = None
_table_init_attempted = False


def _plog(msg: str) -> None:
    """解析链路日志（看 Python 终端即可）。"""
    print(f"[PARSE] {msg}", flush=True)


def decide_ocr(digital_text: str, image_coverage: float) -> tuple[bool, str]:
    """
    判断单页是否需要 OCR，并返回原因（便于日志）。
    """
    text_len = len((digital_text or "").strip())
    cover = max(0.0, min(1.0, float(image_coverage or 0.0)))

    if text_len >= _TEXT_RICH_CHARS:
        return False, f"数字层充足 text_len={text_len}>={_TEXT_RICH_CHARS} cover={cover:.0%}"
    if text_len == 0 and cover >= _IMAGE_COVER_EMPTY_PAGE:
        return True, f"无字+有图 text_len=0 cover={cover:.0%}>={_IMAGE_COVER_EMPTY_PAGE:.0%}"
    if text_len < _TEXT_SPARSE_CHARS and cover >= _IMAGE_COVER_SCAN:
        return True, (
            f"稀疏文字+高图像占比 text_len={text_len}<{_TEXT_SPARSE_CHARS} "
            f"cover={cover:.0%}>={_IMAGE_COVER_SCAN:.0%}"
        )
    return False, f"不满足 OCR 条件 text_len={text_len} cover={cover:.0%}"


def should_run_ocr(digital_text: str, image_coverage: float) -> bool:
    need, _ = decide_ocr(digital_text, image_coverage)
    return need

def is_read_error(text: str | None) -> bool:
    """判断 read_file_content 返回值是否为解析失败。"""
    if not text or not text.strip():
        return True
    return text.startswith(_ERROR_MARKERS)


def _file_extension(original_name: str) -> str:
    if not original_name or "." not in original_name:
        return ""
    return original_name.rsplit(".", 1)[-1].lower().strip()


def table_to_markdown(table: list) -> str:
    """将二维表转为 Markdown，便于 RAG 检索。"""
    if not table:
        return ""
    rows: list[list[str]] = []
    for raw in table:
        if raw is None:
            continue
        cells = [("" if c is None else str(c).replace("\n", " ").strip()) for c in raw]
        if any(cells):
            rows.append(cells)
    if not rows:
        return ""

    width = max(len(r) for r in rows)
    norm = [r + [""] * (width - len(r)) for r in rows]
    header = norm[0]
    body = norm[1:] if len(norm) > 1 else []

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def html_table_to_markdown(html: str) -> str:
    """将 RapidTable 输出的 HTML <table> 转为 Markdown。"""
    if not html or "<table" not in html.lower():
        return ""
    rows: list[list[str]] = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.I | re.S):
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, flags=re.I | re.S)
        cleaned = [
            re.sub(r"<[^>]+>", "", unescape(c)).replace("\n", " ").strip()
            for c in cells
        ]
        if any(cleaned):
            rows.append(cleaned)
    return table_to_markdown(rows)


def dedupe_ocr_against_tables(ocr_text: str, tables_md: str) -> str:
    """
    去掉已出现在结构化表格中的 OCR 行，减少 RAG 重复计分。
    保留表格外的正文行（标题、说明等）。
    """
    if not (ocr_text or "").strip():
        return ""
    if not (tables_md or "").strip():
        return ocr_text.strip()

    cell_set: set[str] = set()
    for line in tables_md.splitlines():
        s = line.strip()
        if not s.startswith("|"):
            continue
        if re.match(r"^\|\s*:?-{3,}", s):
            continue
        for cell in s.strip("|").split("|"):
            c = cell.strip()
            if not c or set(c) <= {"-", ":"}:
                continue
            cell_set.add(c)
            cell_set.add(re.sub(r"\s+", "", c))

    if not cell_set:
        return ocr_text.strip()

    kept: list[str] = []
    dropped = 0
    for line in ocr_text.splitlines():
        s = line.strip()
        if not s:
            continue
        compact = re.sub(r"\s+", "", s)
        if s in cell_set or compact in cell_set:
            dropped += 1
            continue
        kept.append(s)
    _plog(f"OCR去重: 原{len(ocr_text.splitlines())}行 → 保留{len(kept)}行 / 去掉表格重复{dropped}行")
    return "\n".join(kept)


def _extract_tables_markdown(page) -> str:
    blocks: list[str] = []
    try:
        tables = page.extract_tables() or []
    except Exception:
        return ""
    for idx, table in enumerate(tables, start=1):
        md = table_to_markdown(table)
        if md:
            blocks.append(f"【表格 {idx}】\n{md}")
    return "\n\n".join(blocks)


def _get_ocr_engine():
    """懒加载 RapidOCR；失败时返回 None（数字 PDF 仍可用）。"""
    global _ocr_engine, _ocr_init_attempted, _ocr_backend
    if _ocr_init_attempted:
        return _ocr_engine
    _ocr_init_attempted = True

    # 优先新包 rapidocr（支持 Python 3.13+）；旧包 rapidocr-onnxruntime 已停维且不支持 3.13
    try:
        from rapidocr import RapidOCR

        _ocr_engine = RapidOCR()
        _ocr_backend = "rapidocr"
        print("[PARSER] RapidOCR 已加载 (rapidocr)", flush=True)
        return _ocr_engine
    except Exception as e1:
        print(f"[PARSER] rapidocr 不可用: {e1}", flush=True)

    try:
        from rapidocr_onnxruntime import RapidOCR

        _ocr_engine = RapidOCR()
        _ocr_backend = "legacy"
        print("[PARSER] RapidOCR 已加载 (rapidocr-onnxruntime)", flush=True)
        return _ocr_engine
    except Exception as e2:
        _ocr_engine = None
        _ocr_backend = ""
        print(f"[PARSER] RapidOCR 全部不可用，扫描件 OCR 将跳过: {e2}", flush=True)
    return _ocr_engine


def _get_table_engine():
    """懒加载 RapidTable（图像表格结构还原 → HTML）。"""
    global _table_engine, _table_init_attempted
    if _table_init_attempted:
        return _table_engine
    _table_init_attempted = True
    try:
        from rapid_table import ModelType, RapidTable, RapidTableInput

        # 中文表格优先 PP-Structure 中文模型；不可用再退 SLANETPLUS
        try:
            _table_engine = RapidTable(
                RapidTableInput(model_type=ModelType.PPSTRUCTURE_ZH)
            )
            print("[PARSER] RapidTable 已加载 (ppstructure_zh)", flush=True)
        except Exception:
            _table_engine = RapidTable(
                RapidTableInput(model_type=ModelType.SLANETPLUS)
            )
            print("[PARSER] RapidTable 已加载 (slanet_plus)", flush=True)
    except Exception as e:
        _table_engine = None
        print(f"[PARSER] RapidTable 不可用，图像表格将仅保留 OCR 纯文本: {e}", flush=True)
    return _table_engine


def _ocr_boxes_for_table(out) -> list | None:
    """把 RapidOCR 输出整理为 RapidTable 需要的 [boxes, txts, scores]。"""
    if out is None:
        return None

    boxes = getattr(out, "boxes", None)
    txts = getattr(out, "txts", None)
    scores = getattr(out, "scores", None)
    if boxes is not None and txts is not None:
        if scores is None:
            scores = tuple(1.0 for _ in txts)
        return [boxes, txts, scores]

    # 旧版 list: [box, text, score]
    result = out[0] if isinstance(out, tuple) else out
    if not result:
        return None
    b_list, t_list, s_list = [], [], []
    for item in result:
        if not item or len(item) < 2:
            continue
        b_list.append(item[0])
        t_list.append(item[1])
        s_list.append(item[2] if len(item) > 2 else 1.0)
    if not t_list:
        return None
    return [b_list, t_list, s_list]


def _structure_tables_from_image(img, ocr_out) -> str:
    """
    对页面图像做表格结构识别，返回 Markdown。
    依赖 RapidTable；失败则返回空串（上层仍保留 OCR 纯文本）。
    """
    table_engine = _get_table_engine()
    if table_engine is None:
        return ""
    ocr_results = _ocr_boxes_for_table(ocr_out)
    if not ocr_results:
        return ""
    try:
        result = table_engine(img, ocr_results=ocr_results)
    except TypeError:
        # 部分版本参数名不同
        try:
            result = table_engine(img, ocr_result=ocr_results)
        except Exception as e:
            print(f"[PARSER] RapidTable 调用失败: {e}", flush=True)
            return ""
    except Exception as e:
        print(f"[PARSER] RapidTable 调用失败: {e}", flush=True)
        return ""

    htmls: list[str] = []
    pred_htmls = getattr(result, "pred_htmls", None)
    if pred_htmls:
        htmls.extend([h for h in pred_htmls if h])
    else:
        single = getattr(result, "pred_html", None)
        if single:
            htmls.append(single)

    blocks: list[str] = []
    for idx, html in enumerate(htmls, start=1):
        md = html_table_to_markdown(html)
        if md:
            blocks.append(f"【图像表格 {idx}】\n{md}")
    if blocks:
        _plog(f"RapidTable 成功: {len(blocks)} 张表, markdown字符数={sum(len(b) for b in blocks)}")
    else:
        _plog("RapidTable 无有效表格输出")
    return "\n\n".join(blocks)


def _ocr_result_to_text(out) -> str:
    """兼容 rapidocr 新版 RapidOCROutput 与旧版 (list, elapse) 返回值。"""
    if out is None:
        return ""

    # 新版: RapidOCROutput.txts
    txts = getattr(out, "txts", None)
    if txts:
        return "\n".join(str(t).strip() for t in txts if t and str(t).strip())

    # 旧版: (result_list, elapse) 或直接 list，项为 [box, text, score]
    result = out[0] if isinstance(out, tuple) else out
    if not result:
        return ""
    lines = []
    for item in result:
        if not item or len(item) < 2:
            continue
        text = item[1]
        if text and str(text).strip():
            lines.append(str(text).strip())
    return "\n".join(lines)


def _import_pymupdf():
    """优先 pymupdf（官方包名）；兼容旧别名 fitz。注意：PyPI 上另有无关包也叫 fitz。"""
    try:
        import pymupdf as fitz
        return fitz
    except ImportError:
        import fitz  # type: ignore
        return fitz


def _pdf_exception_message(exc: Exception) -> str:
    """将打开/解析异常映射为可读错误。"""
    msg = str(exc).lower()
    name = type(exc).__name__.lower()
    blob = f"{name} {msg}"
    if any(
        k in blob
        for k in (
            "password",
            "encrypt",
            "encrypted",
            "authentication",
            "need password",
            "密码",
        )
    ):
        return "[PDF解析错误：PDF 已加密，请解除密码后再上传]"
    if any(k in blob for k in ("damaged", "corrupt", "invalid pdf", "xref", "broken")):
        return "[PDF解析错误：PDF 文件损坏或格式无效]"
    return f"[PDF解析错误: {exc}]"


def _open_pymupdf_doc(file_path: str):
    """
    打开 PDF 并处理加密。
    成功返回 (fitz_module, doc)；失败抛出带可读信息的 Exception。
    """
    fitz = _import_pymupdf()
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        raise RuntimeError(_pdf_exception_message(e)) from e

    try:
        if getattr(doc, "is_encrypted", False):
            # 空密码常见于仅权限加密；失败则视为需用户密码
            ok = False
            try:
                ok = bool(doc.authenticate(""))
            except Exception:
                ok = False
            if not ok:
                doc.close()
                raise RuntimeError("[PDF解析错误：PDF 已加密，请解除密码后再上传]")
        if doc.page_count <= 0:
            doc.close()
            raise RuntimeError("[PDF解析错误：PDF 无页面]")
        return fitz, doc
    except RuntimeError:
        raise
    except Exception as e:
        try:
            doc.close()
        except Exception:
            pass
        raise RuntimeError(_pdf_exception_message(e)) from e


def _page_image_coverage(page) -> float:
    """计算页内图像 bbox 面积占页面面积的比例（PyMuPDF page）。"""
    try:
        rect = page.rect
        page_area = abs(float(rect.width) * float(rect.height))
        if page_area <= 1e-6:
            return 0.0

        img_area = 0.0
        infos = []
        try:
            infos = page.get_image_info(hashes=False) or []
        except Exception:
            infos = []

        for info in infos:
            bbox = info.get("bbox") if isinstance(info, dict) else None
            if not bbox or len(bbox) < 4:
                continue
            x0, y0, x1, y1 = bbox[:4]
            img_area += abs(float(x1 - x0) * float(y1 - y0))

        try:
            blocks = page.get_text("dict").get("blocks") or []
            for block in blocks:
                if block.get("type") != 1:
                    continue
                b = block.get("bbox")
                if not b or len(b) < 4:
                    continue
                x0, y0, x1, y1 = b[:4]
                img_area += abs(float(x1 - x0) * float(y1 - y0))
        except Exception:
            pass

        return max(0.0, min(1.0, img_area / page_area))
    except Exception:
        return 0.0


def _coverages_from_doc(doc, page_count: int) -> list[float]:
    coverages = [0.0] * page_count
    n = min(page_count, doc.page_count)
    for i in range(n):
        coverages[i] = _page_image_coverage(doc.load_page(i))
    return coverages


def _render_page_image(fitz_mod, doc, page_index: int, scale: float = 2.0):
    """从已打开的 doc 渲染单页为 numpy RGB 图。"""
    import numpy as np

    if page_index < 0 or page_index >= doc.page_count:
        return None
    page = doc.load_page(page_index)
    pix = page.get_pixmap(matrix=fitz_mod.Matrix(scale, scale), alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = img[:, :, :3]
    return img


def _render_page_region(fitz_mod, page, bbox, scale: float = 2.5):
    """裁剪页面局部区域渲染为图（插图 OCR）。"""
    import numpy as np

    try:
        clip = fitz_mod.Rect(bbox)
        # 略扩边，避免切掉边缘字
        clip = fitz_mod.Rect(clip.x0 - 2, clip.y0 - 2, clip.x1 + 2, clip.y1 + 2)
        clip = clip & page.rect
        if clip.is_empty or clip.width < 2 or clip.height < 2:
            return None
        pix = page.get_pixmap(matrix=fitz_mod.Matrix(scale, scale), clip=clip, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = img[:, :, :3]
        return img
    except Exception as e:
        _plog(f"区域渲染失败 bbox={bbox}: {e}")
        return None


def _ocr_page_image(img) -> tuple[str, str]:
    """对页面图像 OCR + 表格结构；返回 (去重后纯文本, 表格Markdown)。"""
    engine = _get_ocr_engine()
    if engine is None or img is None:
        _plog("OCR跳过: 引擎或图像不可用")
        return "", ""
    try:
        _plog(f"RapidOCR 开始识别 image_shape={getattr(img, 'shape', None)}")
        out = engine(img)
        plain = _ocr_result_to_text(out)
        _plog(f"RapidOCR 完成: 文本{len(plain)}字 / {plain.count(chr(10))+1 if plain else 0}行")
        _plog("RapidTable 开始表格结构识别…")
        tables_md = _structure_tables_from_image(img, out)
        plain = dedupe_ocr_against_tables(plain, tables_md)
        return plain, tables_md
    except Exception as e:
        _plog(f"OCR 图像识别失败: {e}")
        print(f"[PARSER] OCR 图像识别失败: {e}", flush=True)
        return "", ""


def _merge_page_parts(
    digital_text: str,
    tables_md: str,
    ocr_text: str,
    *,
    prefer_ocr: bool = False,
) -> tuple[str, str]:
    """
    合并一页内容，返回 (文本, 模式标记 text|table|ocr|mixed)。
    prefer_ocr=True 表示判定为扫描页，优先采用 OCR 结果。
    """
    parts: list[str] = []
    modes: list[str] = []

    digital = (digital_text or "").strip()
    tables = (tables_md or "").strip()
    ocr = (ocr_text or "").strip()

    use_ocr = prefer_ocr and bool(ocr)

    if use_ocr:
        parts.append(ocr)
        modes.append("ocr")
    elif digital:
        parts.append(digital)
        modes.append("text")

    if tables:
        parts.append(tables)
        modes.append("table")

    if not parts and ocr:
        parts.append(ocr)
        modes.append("ocr")

    mode = "mixed" if len(set(modes)) > 1 else (modes[0] if modes else "empty")
    return "\n\n".join(parts).strip(), mode


def iter_parse_events(
    file_path: str,
    original_name: str,
    cancel_check: Optional[CancelCheck] = None,
) -> Iterator[dict]:
    """
    流式解析事件：
      {"type": "progress", "stage": "reading", "page", "total", "mode", "progress", "message"}
      {"type": "result", "text": "..."}
      {"type": "error", "message": "..."}
    """
    if not file_path:
        yield {"type": "error", "message": "[错误：文件路径为空]"}
        return
    if not os.path.isfile(file_path):
        yield {"type": "error", "message": f"[错误：文件不存在 {file_path}]"}
        return

    ext = _file_extension(original_name)
    if ext not in SUPPORTED_EXTENSIONS:
        label = f".{ext}" if ext else "(无扩展名)"
        yield {"type": "error", "message": f"[不支持的文件类型: {label}]"}
        return

    if ext == "txt":
        _plog(f"======= 开始解析 TXT name={original_name} =======")
        yield {
            "type": "progress",
            "stage": "reading",
            "page": 1,
            "total": 1,
            "mode": "text",
            "progress": 8,
            "message": "正在读取 TXT…",
        }
        if cancel_check and cancel_check():
            yield {"type": "error", "message": "[错误：请求已取消]"}
            return
        text = _read_txt(file_path)
        if is_read_error(text):
            _plog(f"TXT 失败: {text}")
            yield {"type": "error", "message": text}
            return
        _plog(f"TXT 完成 chars={len(text)}")
        # TXT 无真实页概念：page=0，避免问答里出现「第 1 页」
        yield {"type": "result", "text": text, "pages": [{"page": 0, "text": text}]}
        return

    # PDF：
    # 1) 优先用 PyMuPDF 单次打开（加密检查 / 页数上限 / 覆盖率 / OCR 渲染共用）
    # 2) 数字层+表格优先 pdfplumber；否则用已打开的 PyMuPDF 抽文本
    _plog(f"======= 开始解析 PDF name={original_name} path={file_path} =======")
    try:
        import pdfplumber  # noqa: F401

        has_pdfplumber = True
    except ImportError:
        has_pdfplumber = False
    _plog(f"后端: pdfplumber={'yes' if has_pdfplumber else 'no'}")

    fitz_mod = None
    fitz_doc = None
    try:
        try:
            fitz_mod, fitz_doc = _open_pymupdf_doc(file_path)
        except ImportError:
            fitz_mod, fitz_doc = None, None
        except RuntimeError as e:
            _plog(f"打开失败: {e}")
            yield {"type": "error", "message": str(e)}
            return
        except Exception as e:
            msg = _pdf_exception_message(e)
            _plog(f"打开异常: {msg}")
            yield {"type": "error", "message": msg}
            return

        if fitz_doc is None and not has_pdfplumber:
            yield {
                "type": "error",
                "message": "[PDF解析错误：未安装 pdfplumber 或 PyMuPDF，请执行 pip install pdfplumber PyMuPDF]",
            }
            return

        if fitz_doc is not None:
            total = fitz_doc.page_count
            _plog(f"PyMuPDF 打开成功 pages={total}")
        else:
            # 仅有 pdfplumber：先打开取页数（并识别加密）
            try:
                import pdfplumber

                with pdfplumber.open(file_path) as pdf:
                    total = len(pdf.pages)
            except Exception as e:
                yield {"type": "error", "message": _pdf_exception_message(e)}
                return

        if total <= 0:
            yield {"type": "error", "message": "[PDF解析错误：PDF 无页面]"}
            return

        max_pages = _max_pdf_pages()
        if total > max_pages:
            _plog(f"页数超限 total={total} max={max_pages}")
            yield {
                "type": "error",
                "message": (
                    f"[PDF解析错误：页数超过上限 {max_pages}（当前 {total} 页），"
                    f"请拆分后上传，或设置环境变量 SMARTDOC_MAX_PDF_PAGES]"
                ),
            }
            return

        coverages = (
            _coverages_from_doc(fitz_doc, total)
            if fitz_doc is not None
            else [0.0] * total
        )
        _plog(
            "图像覆盖率: "
            + ", ".join(f"p{i+1}={c:.0%}" for i, c in enumerate(coverages))
        )

        page_digitals: list[str] = []
        page_tables: list[str] = []

        if has_pdfplumber:
            import pdfplumber
            _plog("阶段A: pdfplumber 抽取数字层+表格")

            try:
                with pdfplumber.open(file_path) as pdf:
                    # pdfplumber 页数可能与 pymupdf 略有差异时取较小值
                    n = min(total, len(pdf.pages))
                    for i in range(n):
                        if cancel_check and cancel_check():
                            yield {"type": "error", "message": "[错误：请求已取消]"}
                            return
                        page = pdf.pages[i]
                        page_no = i + 1
                        pct = 5 + int(3 * page_no / total)
                        digital = page.extract_text() or ""
                        tables_md = _extract_tables_markdown(page)
                        page_digitals.append(digital)
                        page_tables.append(tables_md)
                        _plog(
                            f"P{page_no}/{total} 数字层: chars={len(digital.strip())} "
                            f"tables={'yes' if tables_md else 'no'}"
                        )
                        yield {
                            "type": "progress",
                            "stage": "reading",
                            "page": page_no,
                            "total": total,
                            "mode": "text" if digital.strip() else "empty",
                            "progress": pct,
                            "message": f"正在读取第 {page_no}/{total} 页…",
                        }
                    while len(page_digitals) < total:
                        page_digitals.append("")
                        page_tables.append("")
            except Exception as e:
                yield {"type": "error", "message": _pdf_exception_message(e)}
                return
        else:
            assert fitz_doc is not None
            _plog("阶段A: PyMuPDF 抽取数字层（无 pdfplumber）")
            for i in range(total):
                if cancel_check and cancel_check():
                    yield {"type": "error", "message": "[错误：请求已取消]"}
                    return
                page_no = i + 1
                pct = 5 + int(3 * page_no / total)
                digital = fitz_doc.load_page(i).get_text("text") or ""
                page_digitals.append(digital)
                page_tables.append("")
                _plog(f"P{page_no}/{total} 数字层: chars={len(digital.strip())}")
                yield {
                    "type": "progress",
                    "stage": "reading",
                    "page": page_no,
                    "total": total,
                    "mode": "text" if digital.strip() else "empty",
                    "progress": pct,
                    "message": f"正在读取第 {page_no}/{total} 页（PyMuPDF）…",
                }

        _plog("阶段B: 版面分析 + 按页策略（digital / hybrid / full_ocr）")
        from services.layout import (
            NEAR_FULL_PAGE_RATIO,
            analyze_page,
            choose_strategy,
            merge_reading_order,
        )

        pages_out: list[dict] = []
        for i in range(total):
            if cancel_check and cancel_check():
                yield {"type": "error", "message": "[错误：请求已取消]"}
                return

            page_no = i + 1
            pct = 8 + int(2 * page_no / total)
            digital = page_digitals[i]
            tables_md = page_tables[i]
            coverage = coverages[i] if i < len(coverages) else 0.0

            layout = {
                "text_blocks": [],
                "image_regions": [],
                "text_chars": 0,
                "max_image_ratio": 0.0,
            }
            fitz_page = None
            if fitz_doc is not None:
                try:
                    fitz_page = fitz_doc.load_page(i)
                    layout = analyze_page(fitz_page)
                except Exception as e:
                    _plog(f"P{page_no}/{total} 版面分析失败: {e}")

            strategy, reason = choose_strategy(
                digital,
                coverage,
                layout,
                text_rich_chars=_TEXT_RICH_CHARS,
                text_sparse_chars=_TEXT_SPARSE_CHARS,
                image_cover_scan=_IMAGE_COVER_SCAN,
                image_cover_empty=_IMAGE_COVER_EMPTY_PAGE,
            )
            _plog(
                f"P{page_no}/{total} 策略={strategy} | {reason} | "
                f"text_blocks={len(layout.get('text_blocks') or [])} "
                f"image_regions={len(layout.get('image_regions') or [])}"
            )

            ocr_text = ""
            ocr_tables_md = ""
            page_text = ""
            mode = "empty"

            if strategy == "full_ocr":
                yield {
                    "type": "progress",
                    "stage": "reading",
                    "page": page_no,
                    "total": total,
                    "mode": "ocr",
                    "progress": pct,
                    "image_coverage": round(coverage, 3),
                    "message": (
                        f"正在整页 OCR 第 {page_no}/{total} 页"
                        f"（图像占比 {coverage:.0%}）…"
                    ),
                }
                if cancel_check and cancel_check():
                    yield {"type": "error", "message": "[错误：请求已取消]"}
                    return
                if fitz_mod is not None and fitz_doc is not None:
                    img = _render_page_image(fitz_mod, fitz_doc, i)
                    ocr_text, ocr_tables_md = _ocr_page_image(img)
                    _plog(
                        f"P{page_no}/{total} 整页OCR: text={len(ocr_text)}字 "
                        f"img_table={'yes' if ocr_tables_md else 'no'}"
                    )
                combined_tables = "\n\n".join(
                    x for x in (tables_md.strip(), ocr_tables_md.strip()) if x
                )
                page_text, mode = _merge_page_parts(
                    digital, combined_tables, ocr_text, prefer_ocr=True
                )

            elif strategy == "hybrid" and fitz_mod is not None and fitz_page is not None:
                yield {
                    "type": "progress",
                    "stage": "reading",
                    "page": page_no,
                    "total": total,
                    "mode": "hybrid",
                    "progress": pct,
                    "image_coverage": round(coverage, 3),
                    "message": f"正在版面解析第 {page_no}/{total} 页（文字+插图OCR）…",
                }
                items: list[dict] = []
                # 文字块（优先版面块；没有则整页数字层当作一块）
                text_blocks = layout.get("text_blocks") or []
                if text_blocks:
                    for b in text_blocks:
                        items.append(
                            {
                                "y": b["y"],
                                "x": b["x"],
                                "text": b["text"],
                                "kind": "text",
                            }
                        )
                elif digital.strip():
                    items.append({"y": 0.0, "x": 0.0, "text": digital.strip(), "kind": "text"})

                regions = [
                    r
                    for r in (layout.get("image_regions") or [])
                    if float(r.get("area_ratio") or 0) < NEAR_FULL_PAGE_RATIO
                ]
                region_tables: list[str] = []
                for ri, reg in enumerate(regions, start=1):
                    if cancel_check and cancel_check():
                        yield {"type": "error", "message": "[错误：请求已取消]"}
                        return
                    _plog(
                        f"P{page_no}/{total} 插图OCR #{ri} "
                        f"area={float(reg['area_ratio']):.0%} bbox={reg['bbox']}"
                    )
                    img = _render_page_region(fitz_mod, fitz_page, reg["bbox"])
                    plain, tables = _ocr_page_image(img)
                    if tables:
                        region_tables.append(tables)
                    if plain.strip():
                        items.append(
                            {
                                "y": float(reg["y"]),
                                "x": float(reg["x"]),
                                "text": f"【图内文字 {ri}】\n{plain.strip()}",
                                "kind": "ocr",
                            }
                        )

                page_text = merge_reading_order(items)
                ocr_tables_md = "\n\n".join(region_tables)
                combined_tables = "\n\n".join(
                    x for x in (tables_md.strip(), ocr_tables_md.strip()) if x
                )
                if combined_tables:
                    page_text = (
                        f"{page_text}\n\n{combined_tables}".strip()
                        if page_text
                        else combined_tables
                    )
                kinds = {it.get("kind") for it in items}
                if "text" in kinds and "ocr" in kinds:
                    mode = "mixed"
                elif "ocr" in kinds:
                    mode = "ocr"
                elif page_text:
                    mode = "text"
                else:
                    mode = "empty"
                if combined_tables and mode in ("text", "ocr", "mixed"):
                    mode = "mixed" if mode != "empty" else "table"
                _plog(
                    f"P{page_no}/{total} 混排完成 mode={mode} out_chars={len(page_text)} "
                    f"blocks={len(items)} img_ocr={len(regions)}"
                )

            else:
                # digital only（或无法做区域 OCR 时的回退）
                if strategy == "hybrid":
                    _plog(f"P{page_no}/{total} hybrid 回退为 digital（无 PyMuPDF 页对象）")
                combined_tables = tables_md.strip()
                page_text, mode = _merge_page_parts(
                    digital, combined_tables, "", prefer_ocr=False
                )
                _plog(
                    f"P{page_no}/{total} 数字层完成 mode={mode} out_chars={len(page_text)}"
                )

            if page_text:
                pages_out.append({"page": page_no, "text": page_text})

            yield {
                "type": "progress",
                "stage": "reading",
                "page": page_no,
                "total": total,
                "mode": mode,
                "progress": pct,
                "image_coverage": round(coverage, 3),
                "ocr": strategy in ("full_ocr", "hybrid"),
                "strategy": strategy,
                "message": f"正在解析第 {page_no}/{total} 页（{mode}/{strategy}）",
            }

        if not pages_out:
            _plog("失败: 全部页无有效文本")
            yield {
                "type": "error",
                "message": "[PDF解析错误：文档中没有可提取的文字（含 OCR）]",
            }
            return

        full = "\n\n".join(
            f"【第{p['page']}页】\n{p['text']}" for p in pages_out
        )
        _plog(f"======= 解析完成 pages={len(pages_out)} total_chars={len(full)} =======")
        yield {"type": "result", "text": full, "pages": pages_out}
    except Exception as e:
        _plog(f"异常: {e}")
        yield {"type": "error", "message": _pdf_exception_message(e)}
    finally:
        if fitz_doc is not None:
            try:
                fitz_doc.close()
            except Exception:
                pass


def _read_txt(file_path: str) -> str:
    encodings = ("utf-8-sig", "utf-8", "gb18030", "gbk")
    last_error: Exception | None = None

    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                content = f.read()
            # 防御：部分编码组合仍可能残留 BOM
            if content.startswith("\ufeff"):
                content = content.lstrip("\ufeff")
            if content.strip():
                return content
            return "[TXT读取错误：文件内容为空]"
        except UnicodeDecodeError as e:
            last_error = e
            continue
        except Exception as e:
            return f"[TXT读取错误: {e}]"

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        if content.startswith("\ufeff"):
            content = content.lstrip("\ufeff")
        if content.strip():
            return content
        return "[TXT读取错误：文件内容为空]"
    except Exception as e:
        detail = last_error or e
        return f"[TXT读取错误: {detail}]"


def read_file_content(
    file_path: str,
    original_name: str,
    progress_cb: Optional[ProgressCallback] = None,
    cancel_check: Optional[CancelCheck] = None,
) -> str:
    """
    从本地路径读取文档纯文本。

    失败时返回以约定前缀开头的错误字符串。
    progress_cb 可选，接收 progress 事件字典。
    """
    result_text = ""
    for event in iter_parse_events(file_path, original_name, cancel_check=cancel_check):
        et = event.get("type")
        if et == "progress":
            if progress_cb:
                progress_cb(event)
        elif et == "error":
            return str(event.get("message") or "[错误：解析失败]")
        elif et == "result":
            result_text = str(event.get("text") or "")
    if not result_text.strip():
        return "[错误：未解析到文本]"
    return result_text
