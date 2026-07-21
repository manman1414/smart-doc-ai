# 作者：yangkunpeng1
# 日期：2026-07-21
"""生成 Web 端 OCR 联调用样例 PDF（用系统字体画图，避免中文空白页）。"""

from __future__ import annotations

import io
from pathlib import Path

import pymupdf as fitz
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent.parent / "sample_pdfs"

# Windows 常见中文字体（按优先级）
_FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\msyh.ttf"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\simsun.ttc"),
    Path(r"C:\Windows\Fonts\simkai.ttf"),
]


def _pick_font(size: int = 36) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for p in _FONT_CANDIDATES:
        if p.is_file():
            try:
                return ImageFont.truetype(str(p), size=size)
            except Exception:
                continue
    print("WARN: 未找到中文字体，中文可能仍显示异常")
    return ImageFont.load_default()


def _a4_image(bg: str = "white") -> Image.Image:
    # A4 @ ~150dpi，OCR 更友好
    return Image.new("RGB", (1240, 1754), bg)


def _draw_lines(
    img: Image.Image,
    lines: list[str],
    *,
    font_size: int = 36,
    x: int = 80,
    y0: int = 100,
    line_gap: int = 56,
    fill: str = "black",
) -> None:
    draw = ImageDraw.Draw(img)
    font = _pick_font(font_size)
    y = y0
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += line_gap


def _draw_table(
    img: Image.Image,
    headers: list[str],
    rows: list[list[str]],
    *,
    origin: tuple[int, int] = (80, 220),
    col_w: int = 280,
    row_h: int = 64,
) -> None:
    draw = ImageDraw.Draw(img)
    font = _pick_font(32)
    data = [headers] + rows
    x0, y0 = origin
    cols = len(headers)
    for r, row in enumerate(data):
        for c, cell in enumerate(row):
            x = x0 + c * col_w
            y = y0 + r * row_h
            rect = [x, y, x + col_w, y + row_h]
            draw.rectangle(rect, outline="black", width=2)
            draw.text((x + 16, y + 14), str(cell), font=font, fill="black")


def _image_to_pdf(img: Image.Image, name: str) -> Path:
    """整页贴图 → 无文字层，强制走 OCR。"""
    OUT.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png = buf.getvalue()

    doc = fitz.open()
    # 与图片比例接近的页面
    w, h = 595, 842
    page = doc.new_page(width=w, height=h)
    page.insert_image(page.rect, stream=png)

    path = OUT / name
    doc.save(path)
    doc.close()
    print(f"ok  {path}  ({path.stat().st_size} bytes)")
    return path


def make_ocr_text() -> None:
    img = _a4_image()
    _draw_lines(
        img,
        [
            "SmartDoc OCR 文字测试",
            "",
            "本页是整页图片，没有可选中文字层。",
            "用于验证：扫描件判定 + RapidOCR。",
            "",
            "甲方：示例科技有限公司",
            "乙方：智能文档测试方",
            "合同金额：人民币壹拾万元整",
            "交付周期：三十个自然日",
            "",
            "关键词：扫描识别、本地大模型、文档问答",
        ],
        font_size=40,
        line_gap=58,
    )
    _image_to_pdf(img, "01_ocr_text.pdf")


def make_ocr_table() -> None:
    img = _a4_image()
    _draw_lines(
        img,
        [
            "SmartDoc OCR 表格测试",
            "下面是画在图片上的表格，应走 OCR + RapidTable。",
        ],
        font_size=36,
        y0=80,
        line_gap=52,
    )
    _draw_table(
        img,
        ["产品", "数量", "单价"],
        [
            ["路由器", "2", "299"],
            ["交换机", "1", "899"],
            ["网线", "10", "25"],
            ["防火墙", "1", "3999"],
        ],
        origin=(80, 240),
    )
    _draw_lines(
        img,
        ["备注：请确认能否问答「交换机单价是多少」。"],
        font_size=32,
        y0=560,
    )
    _image_to_pdf(img, "02_ocr_table.pdf")


def make_ocr_long() -> None:
    img = _a4_image()
    _draw_lines(
        img,
        [
            "智能文档问答系统验收说明",
            "",
            "一、测试目的",
            "验证上传扫描 PDF 后，系统能否正确 OCR，",
            "并将内容写入向量库，支持多轮追问。",
            "",
            "二、验收标准",
            "1. 进度条出现 OCR / 表格识别提示",
            "2. 摘要包含本文关键信息",
            "3. 提问「验收标准有几条」能答出要点",
            "",
            "三、注意",
            "本页故意做成图片页，请勿用复制文字验证。",
        ],
        font_size=34,
        line_gap=50,
    )
    _image_to_pdf(img, "03_ocr_long_doc.pdf")


def make_mixed() -> None:
    """第1页数字层（英文，避免缺字），第2页扫描中文。"""
    OUT.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()

    p1 = doc.new_page()
    p1.insert_text(
        (72, 72),
        "Mixed PDF - Page 1 (digital text layer)\n\n"
        "This page has a real text layer and should NOT need OCR.\n"
        "Page 2 is a full-page scan image for OCR.",
        fontsize=14,
        fontname="helv",
    )

    img = _a4_image()
    _draw_lines(
        img,
        [
            "混合 PDF 第2页（扫描）",
            "这一页是图片，应当触发 OCR。",
            "验证同一文件内按页切换解析策略。",
            "暗号：蓝桥春雪",
        ],
        font_size=38,
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    p2 = doc.new_page()
    p2.insert_image(p2.rect, stream=buf.getvalue())

    path = OUT / "05_mixed_digital_and_ocr.pdf"
    doc.save(path)
    doc.close()
    print(f"ok  {path}  ({path.stat().st_size} bytes)")


def _cjk_fontfile() -> str | None:
    for p in _FONT_CANDIDATES:
        if p.is_file():
            return str(p)
    return None


def make_digital_text() -> None:
    """可抽取文字层（中文靠系统字体嵌入）。"""
    fontfile = _cjk_fontfile()
    doc = fitz.open()
    page = doc.new_page()
    if fontfile:
        page.insert_font(fontname="cjk", fontfile=fontfile)
        fontname = "cjk"
    else:
        fontname = "helv"
    lines = [
        "SmartDoc 数字文字 PDF",
        "本页有真实文字层，不应触发 OCR。",
        "甲方：示例科技有限公司",
        "乙方：智能文档测试方",
        "用于验证：pdfplumber / PyMuPDF 数字层解析。",
    ]
    y = 72
    for line in lines:
        page.insert_text((72, y), line, fontsize=14, fontname=fontname)
        y += 28
    path = OUT / "06_digital_text.pdf"
    OUT.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    doc.close()
    print(f"ok  {path}  ({path.stat().st_size} bytes)")


def make_digital_table() -> None:
    """数字层表格（画框 + 嵌入中文字体）。"""
    fontfile = _cjk_fontfile()
    doc = fitz.open()
    page = doc.new_page()
    if fontfile:
        page.insert_font(fontname="cjk", fontfile=fontfile)
        fontname = "cjk"
    else:
        fontname = "helv"
    page.insert_text((72, 56), "SmartDoc 数字表格 PDF", fontsize=16, fontname=fontname)
    rows = [
        ["姓名", "部门", "金额"],
        ["张三", "研发", "1000"],
        ["李四", "产品", "2000"],
        ["王五", "运营", "1500"],
    ]
    x0, y0, cw, rh = 72, 100, 120, 28
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            x = x0 + c * cw
            y = y0 + r * rh
            rect = fitz.Rect(x, y, x + cw, y + rh)
            page.draw_rect(rect, color=(0, 0, 0), width=0.6)
            page.insert_textbox(
                fitz.Rect(x + 4, y + 6, x + cw - 4, y + rh - 4),
                cell,
                fontsize=11,
                fontname=fontname,
            )
    page.insert_text(
        (72, 240),
        "用于验证：数字层表格 → Markdown（不问 OCR）。",
        fontsize=12,
        fontname=fontname,
    )
    path = OUT / "07_digital_table.pdf"
    OUT.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    doc.close()
    print(f"ok  {path}  ({path.stat().st_size} bytes)")


def make_encrypted() -> None:
    """加密 PDF：上传应明确提示需要密码。密码 test123"""
    fontfile = _cjk_fontfile()
    doc = fitz.open()
    page = doc.new_page()
    if fontfile:
        page.insert_font(fontname="cjk", fontfile=fontfile)
        fontname = "cjk"
    else:
        fontname = "helv"
    page.insert_text(
        (72, 72),
        "加密 PDF 测试页：若解析提示已加密即通过。",
        fontsize=14,
        fontname=fontname,
    )
    page.insert_text(
        (72, 110),
        "Password protected sample for SmartDoc.",
        fontsize=12,
        fontname="helv",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "08_encrypted_need_password.pdf"
    doc.save(
        path,
        encryption=fitz.PDF_ENCRYPT_AES_256,
        user_pw="test123",
        owner_pw="owner123",
    )
    doc.close()
    print(f"ok  {path}  ({path.stat().st_size} bytes)  密码: test123")


def make_hybrid_text_and_image() -> None:
    """同一页：上方数字层文字 + 下方嵌入图片（图内有字，测版面 hybrid）。"""
    OUT.mkdir(parents=True, exist_ok=True)
    fontfile = _cjk_fontfile()
    doc = fitz.open()
    page = doc.new_page()
    if fontfile:
        page.insert_font(fontname="cjk", fontfile=fontfile)
        fontname = "cjk"
    else:
        fontname = "helv"

    # 足够长的数字层，避免被判成整页 OCR
    body = (
        "图文混排测试页。本段是可复制的数字层文字，不应整页OCR。"
        "系统应保留本段，并对下方插图单独做区域OCR。"
        "暗号甲：青云直上。"
    )
    page.insert_text((72, 72), body, fontsize=12, fontname=fontname)

    # 画一张带字的图嵌入页面下半部
    img = Image.new("RGB", (900, 420), "white")
    draw = ImageDraw.Draw(img)
    font = _pick_font(40)
    draw.rectangle([20, 20, 880, 400], outline="black", width=3)
    draw.text((50, 80), "插图区域文字（应走区域OCR）", font=font, fill="black")
    draw.text((50, 160), "暗号乙：紫气东来", font=font, fill="black")
    draw.text((50, 240), "交换机单价：899", font=font, fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    rect = fitz.Rect(72, 280, 520, 520)
    page.insert_image(rect, stream=buf.getvalue())

    path = OUT / "09_hybrid_text_and_image.pdf"
    doc.save(path)
    doc.close()
    print(f"ok  {path}  ({path.stat().st_size} bytes)")


def main() -> None:
    font = _pick_font(24)
    print("font:", getattr(font, "path", font))

    # OCR
    make_ocr_text()
    make_ocr_table()
    make_ocr_long()
    make_mixed()
    # 非 OCR / 边界
    make_digital_text()
    make_digital_table()
    make_encrypted()
    make_hybrid_text_and_image()

    print(f"\n输出目录: {OUT}")
    print(
        """
========== Web 测试清单 ==========
01_ocr_text.pdf              整页 OCR 文字
02_ocr_table.pdf             整页 OCR + 图像表格
03_ocr_long_doc.pdf          稍长扫描正文
05_mixed_digital_and_ocr.pdf 跨页混合
06_digital_text.pdf          纯数字层
07_digital_table.pdf         数字表格
08_encrypted_need_password.pdf  加密（应报错；密码 test123）
09_hybrid_text_and_image.pdf 同页图文混排（策略=hybrid）
  预期日志: 策略=hybrid；问答可同时问到「青云直上」和「紫气东来」
=================================
"""
    )


if __name__ == "__main__":
    main()
