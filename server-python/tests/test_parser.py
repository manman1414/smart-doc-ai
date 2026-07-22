# 作者：yangkunpeng1
# 日期：2026-07-21
"""解析模块单测（不依赖服务进程 / LM Studio）。"""

from __future__ import annotations

import os
import tempfile
import unittest

from services.parser import (
    _max_pdf_pages,
    _pdf_exception_message,
    dedupe_ocr_against_tables,
    html_table_to_markdown,
    is_read_error,
    iter_parse_events,
    read_file_content,
    should_run_ocr,
    table_to_markdown,
)


class ParserTests(unittest.TestCase):
    def test_html_table_to_markdown(self):
        html = """
        <table>
          <tr><td>姓名</td><td>金额</td></tr>
          <tr><td>张三</td><td>100</td></tr>
        </table>
        """
        md = html_table_to_markdown(html)
        self.assertIn("| 姓名 | 金额 |", md)
        self.assertIn("| 张三 | 100 |", md)

    def test_html_table_to_markdown_empty(self):
        self.assertEqual(html_table_to_markdown(""), "")
        self.assertEqual(html_table_to_markdown("<div>no table</div>"), "")

    def test_dedupe_ocr_against_tables(self):
        tables = table_to_markdown(
            [
                ["姓名", "金额"],
                ["张三", "100"],
            ]
        )
        tables = f"【图像表格 1】\n{tables}"
        ocr = "报表标题\n姓名\n金额\n张三\n100\n备注说明"
        cleaned = dedupe_ocr_against_tables(ocr, tables)
        self.assertIn("报表标题", cleaned)
        self.assertIn("备注说明", cleaned)
        self.assertNotIn("张三", cleaned)
        self.assertNotIn("100", cleaned)

    def test_dedupe_ocr_no_table(self):
        self.assertEqual(dedupe_ocr_against_tables("一行\n两行", ""), "一行\n两行")

    def test_pdf_exception_password(self):
        msg = _pdf_exception_message(Exception("document closed or encrypted - password needed"))
        self.assertIn("加密", msg)

    def test_max_pdf_pages_env(self):
        old = os.environ.get("SMARTDOC_MAX_PDF_PAGES")
        try:
            os.environ["SMARTDOC_MAX_PDF_PAGES"] = "5"
            self.assertEqual(_max_pdf_pages(), 5)
            os.environ["SMARTDOC_MAX_PDF_PAGES"] = "bad"
            self.assertEqual(_max_pdf_pages(), 120)
        finally:
            if old is None:
                os.environ.pop("SMARTDOC_MAX_PDF_PAGES", None)
            else:
                os.environ["SMARTDOC_MAX_PDF_PAGES"] = old

    def test_should_run_ocr_digital_rich(self):
        text = "这是一段足够长的可检索数字层正文，用来模拟正常PDF文字页内容。" * 3
        self.assertGreaterEqual(len(text.strip()), 80)
        self.assertFalse(should_run_ocr(text, 0.9))

    def test_should_run_ocr_scan_full_image(self):
        self.assertTrue(should_run_ocr("", 0.8))

    def test_should_run_ocr_sparse_text_high_image(self):
        self.assertTrue(should_run_ocr("第 1 页", 0.6))

    def test_should_run_ocr_blank_no_image(self):
        self.assertFalse(should_run_ocr("", 0.0))
        self.assertFalse(should_run_ocr("短", 0.05))

    def test_chunk_pages_keeps_page(self):
        from services.chunker import chunk_pages

        chunks = chunk_pages(
            [
                {"page": 1, "text": "第一页内容足够长。" * 20},
                {"page": 2, "text": "第二页短文"},
            ],
            chunk_size=40,
            overlap=5,
        )
        self.assertTrue(chunks)
        self.assertTrue(all("page" in c for c in chunks))
        self.assertIn(2, {c["page"] for c in chunks})

    def test_format_chunks_for_prompt(self):
        from services.vector_store import format_chunks_for_prompt

        text = format_chunks_for_prompt(
            [{"text": "合同金额十万", "page": 3}, {"text": "交付周期", "page": 1}]
        )
        self.assertNotIn("【第", text)
        self.assertIn("合同金额十万", text)
        self.assertIn("交付周期", text)

    def test_format_chunks_dedupes(self):
        from services.vector_store import format_chunks_for_prompt

        text = format_chunks_for_prompt(
            [
                {"text": "Mixed PDF - Page 1 digital layer", "page": 1},
                {"text": "Mixed PDF - Page 1 digital layer", "page": 1},
                {"text": "Mixed PDF - Page 1 digital layer extra", "page": 1},
            ]
        )
        self.assertNotIn("【第", text)
        self.assertEqual(text.count("Mixed PDF - Page 1 digital layer"), 1)
        self.assertIn("extra", text)

    def test_format_chunks_dedupes_overlap_window(self):
        from services.vector_store import format_chunks_for_prompt

        # 模拟分块滑窗：两段高度重叠但不互相完整包含
        a = "插图区域文字应走区域OCR暗号乙：紫气东来交换机单价：899元整"
        b = "区域文字应走区域OCR暗号乙：紫气东来交换机单价：899元整补充"
        text = format_chunks_for_prompt(
            [{"text": a, "page": 1}, {"text": b, "page": 1}]
        )
        self.assertEqual(text.count("紫气东来"), 1)

    def test_empty_path(self):
        text = read_file_content("", "a.txt")
        self.assertTrue(is_read_error(text))
        self.assertIn("路径为空", text)

    def test_missing_file(self):
        text = read_file_content(r"C:\nonexistent\smart-doc-ai-test.txt", "a.txt")
        self.assertTrue(is_read_error(text))
        self.assertIn("不存在", text)

    def test_unsupported_ext(self):
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
            f.write(b"hello")
            path = f.name
        try:
            text = read_file_content(path, "report.docx")
            self.assertTrue(is_read_error(text))
            self.assertIn("不支持", text)
        finally:
            os.unlink(path)

    def test_txt_utf8(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write("智能文档问答\n第二行".encode("utf-8"))
            path = f.name
        try:
            text = read_file_content(path, "demo.txt")
            self.assertFalse(is_read_error(text))
            self.assertIn("智能文档", text)
            self.assertIn("第二行", text)
        finally:
            os.unlink(path)

    def test_txt_gbk(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write("合同摘要测试".encode("gbk"))
            path = f.name
        try:
            text = read_file_content(path, "gbk.txt")
            self.assertFalse(is_read_error(text))
            self.assertEqual(text.strip(), "合同摘要测试")
        finally:
            os.unlink(path)

    def test_txt_utf8_bom(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            # utf-8-sig 会自动写入 BOM，字符串本身不要再带 \ufeff
            f.write("带BOM内容".encode("utf-8-sig"))
            path = f.name
        try:
            text = read_file_content(path, "bom.txt")
            self.assertFalse(is_read_error(text))
            self.assertTrue(text.startswith("带BOM"), repr(text[:20]))
        finally:
            os.unlink(path)

    def test_is_read_error_false_positive(self):
        self.assertFalse(is_read_error("[重要] 条款一：付款方式"))

    def test_table_to_markdown(self):
        md = table_to_markdown(
            [
                ["姓名", "金额"],
                ["张三", "100"],
                ["李四", "200"],
            ]
        )
        self.assertIn("| 姓名 | 金额 |", md)
        self.assertIn("| --- | --- |", md)
        self.assertIn("| 张三 | 100 |", md)

    def test_table_to_markdown_empty(self):
        self.assertEqual(table_to_markdown([]), "")
        self.assertEqual(table_to_markdown([[None, None]]), "")

    def test_txt_iter_progress(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write("进度测试".encode("utf-8"))
            path = f.name
        try:
            events = list(iter_parse_events(path, "p.txt"))
            types = [e["type"] for e in events]
            self.assertIn("progress", types)
            self.assertEqual(types[-1], "result")
            progress = next(e for e in events if e["type"] == "progress")
            self.assertEqual(progress["page"], 1)
            self.assertEqual(progress["total"], 1)
        finally:
            os.unlink(path)

    def test_pdf_digital_text_and_progress(self):
        try:
            try:
                import pymupdf as fitz
            except ImportError:
                import fitz
        except ImportError as e:
            self.skipTest(f"PyMuPDF 未安装: {e}")

        fd, path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        try:
            doc = fitz.open()
            page = doc.new_page()
            # 足够长，避免单测误触发 OCR（阈值 30）
            page.insert_text(
                (72, 72),
                "SmartDoc digital page one - enough chars for text layer",
            )
            page2 = doc.new_page()
            page2.insert_text(
                (72, 72),
                "SmartDoc digital page two - enough chars for text layer",
            )
            doc.save(path)
            doc.close()

            events = list(iter_parse_events(path, "demo.pdf"))
            progress_events = [e for e in events if e["type"] == "progress"]
            self.assertGreaterEqual(
                len(progress_events),
                2,
                msg=f"events={events!r}",
            )
            self.assertEqual(progress_events[0]["total"], 2)
            errors = [e for e in events if e["type"] == "error"]
            self.assertEqual(errors, [], msg=f"events={events!r}")
            result = next(e for e in events if e["type"] == "result")
            self.assertIn("page one", result["text"].lower())
            self.assertIn("page two", result["text"].lower())
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_pdf_page_limit(self):
        try:
            import pymupdf as fitz
        except ImportError:
            try:
                import fitz
            except ImportError as e:
                self.skipTest(f"PyMuPDF 未安装: {e}")

        old = os.environ.get("SMARTDOC_MAX_PDF_PAGES")
        os.environ["SMARTDOC_MAX_PDF_PAGES"] = "1"
        fd, path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        try:
            doc = fitz.open()
            doc.new_page().insert_text((72, 72), "page A enough text for digital layer xx")
            doc.new_page().insert_text((72, 72), "page B enough text for digital layer yy")
            doc.save(path)
            doc.close()
            events = list(iter_parse_events(path, "limit.pdf"))
            errors = [e for e in events if e["type"] == "error"]
            self.assertTrue(errors, msg=f"events={events!r}")
            self.assertIn("上限", errors[0]["message"])
        finally:
            if old is None:
                os.environ.pop("SMARTDOC_MAX_PDF_PAGES", None)
            else:
                os.environ["SMARTDOC_MAX_PDF_PAGES"] = old
            if os.path.exists(path):
                os.unlink(path)


if __name__ == "__main__":
    unittest.main()
