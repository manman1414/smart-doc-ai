# 作者：yangkunpeng1
# 日期：2026-07-21
"""分块模块单测。"""

from __future__ import annotations

import unittest

from services.chunker import (
    _apply_overlap,
    _merge_tiny_chunks,
    chunk_pages,
    chunk_text,
    chunk_text_recursive,
    chunk_text_units,
    resolve_strategy,
)


class ChunkerTests(unittest.TestCase):
    def test_sentence_boundary_zh(self):
        text = "甲方应支付货款。乙方应交付货物。违约责任另行约定。"
        chunks = chunk_text(text, chunk_size=20, overlap=0, strategy="structure")
        self.assertTrue(chunks)
        joined = "".join(chunks)
        self.assertIn("甲方应支付货款。", joined)
        self.assertIn("乙方应交付货物。", joined)

    def test_markdown_table_kept_together(self):
        text = (
            "说明文字在前。\n\n"
            "| 姓名 | 金额 |\n"
            "| --- | --- |\n"
            "| 张三 | 100 |\n"
            "| 李四 | 200 |\n"
        )
        units = chunk_text_units(text, chunk_size=500, overlap=0, strategy="structure")
        kinds = [u["kind"] for u in units]
        self.assertIn("table", kinds)
        table = next(u for u in units if u["kind"] == "table")
        self.assertIn("| 张三 | 100 |", table["text"])
        self.assertIn("| 李四 | 200 |", table["text"])

    def test_figure_section_separate(self):
        text = (
            "正文暗号甲：青云直上。\n\n"
            "【图内文字 p2-1】\n"
            "暗号乙：紫气东来\n"
            "交换机单价：899\n"
        )
        units = chunk_text_units(text, chunk_size=500, overlap=0, strategy="structure")
        self.assertTrue(any(u["kind"] == "figure" for u in units))
        fig = next(u for u in units if u["kind"] == "figure")
        self.assertIn("紫气东来", fig["text"])
        self.assertEqual(fig.get("figure_id"), "p2-1")
        self.assertTrue(fig["text"].startswith("【图内文字 p2-1】"))
        body = next(u for u in units if u["kind"] == "text")
        self.assertIn("青云直上", body["text"])
        self.assertNotIn("紫气东来", body["text"])

    def test_two_long_figures_keep_stable_ids(self):
        """一页两图、各超长拆多块时，每块保留同一 figure_id，且两图不混。"""
        fig1 = "图一说明。" * 40
        fig2 = "图二说明。" * 40
        text = (
            f"【图内文字 p5-1】\n{fig1}\n\n"
            f"【图内文字 p5-2】\n{fig2}\n"
        )
        chunks = chunk_pages(
            [{"page": 5, "text": text}],
            chunk_size=80,
            overlap=0,
            source_ext="pdf",
        )
        figs = [c for c in chunks if c.get("kind") == "figure"]
        self.assertGreaterEqual(len(figs), 4)
        ids = {c.get("figure_id") for c in figs}
        self.assertEqual(ids, {"p5-1", "p5-2"})
        for c in figs:
            self.assertTrue(
                c["text"].startswith(f"【图内文字 {c['figure_id']}】"),
                msg=c["text"][:40],
            )
            if c["figure_id"] == "p5-1":
                self.assertIn("图一说明", c["text"])
                self.assertNotIn("图二说明", c["text"])
            else:
                self.assertIn("图二说明", c["text"])
                self.assertNotIn("图一说明", c["text"])

    def test_legacy_figure_marker_upgraded_with_page(self):
        text = "【图内文字 1】\n" + ("旧标记续写。" * 30)
        chunks = chunk_pages(
            [{"page": 3, "text": text}],
            chunk_size=60,
            overlap=0,
            source_ext="pdf",
        )
        figs = [c for c in chunks if c.get("kind") == "figure"]
        self.assertTrue(figs)
        self.assertTrue(all(c.get("figure_id") == "p3-1" for c in figs))
        self.assertTrue(all(c["text"].startswith("【图内文字 p3-1】") for c in figs))

    def test_chunk_pages_kind_and_page(self):
        chunks = chunk_pages(
            [
                {
                    "page": 2,
                    "text": "第一句。第二句。第三句。" + "补充说明。" * 30,
                }
            ],
            chunk_size=40,
            overlap=0,
        )
        self.assertTrue(chunks)
        self.assertTrue(all(c["page"] == 2 for c in chunks))
        self.assertTrue(all("kind" in c for c in chunks))

    def test_long_table_splits_with_header(self):
        rows = ["| 姓名 | 金额 |", "| --- | --- |"]
        for i in range(40):
            rows.append(f"| 用户{i} | {i * 10} |")
        table = "\n".join(rows)
        units = chunk_text_units(table, chunk_size=120, overlap=0, strategy="structure")
        self.assertTrue(len(units) >= 2)
        self.assertTrue(all(u["kind"] == "table" for u in units))
        self.assertTrue(all("| 姓名 | 金额 |" in u["text"] for u in units))

    def test_txt_uses_recursive_strategy(self):
        self.assertEqual(resolve_strategy("auto", source_ext="txt"), "recursive")
        self.assertEqual(
            resolve_strategy("auto", sample_text="甲方支付货款。乙方交货。", source_ext="pdf"),
            "recursive",
        )
        self.assertEqual(
            resolve_strategy("auto", sample_text="| a | b |\n| --- | --- |", source_ext="pdf"),
            "structure",
        )
        self.assertEqual(
            resolve_strategy("auto", sample_text="【图内文字 1】\n某某", source_ext="pdf"),
            "structure",
        )

    def test_chunk_pages_plain_pdf_recursive(self):
        pages = [{"page": 1, "text": "第一段。\n\n第二段。\n\n" + ("补充。" * 40)}]
        chunks = chunk_pages(pages, chunk_size=60, overlap=0, source_ext="pdf")
        self.assertTrue(chunks)
        self.assertEqual(chunks[0].get("strategy"), "recursive")

    def test_chunk_pages_per_page_strategy(self):
        pages = [
            {"page": 1, "text": "纯文字第一页。" + ("内容。" * 20)},
            {
                "page": 2,
                "text": "| 姓名 | 金额 |\n| --- | --- |\n| 张三 | 100 |\n",
            },
        ]
        chunks = chunk_pages(pages, chunk_size=80, overlap=0, source_ext="pdf")
        by_page = {}
        for c in chunks:
            by_page.setdefault(c["page"], set()).add(c["strategy"])
        self.assertEqual(by_page[1], {"recursive"})
        self.assertEqual(by_page[2], {"structure"})

    def test_recursive_prefers_paragraphs(self):
        text = (
            "第一段内容比较短。\n\n"
            "第二段也有几句。这里继续写。\n\n"
            + ("第三段很长很长。" * 40)
        )
        chunks = chunk_text_recursive(text, chunk_size=80, overlap=0)
        self.assertTrue(len(chunks) >= 2)
        self.assertTrue(any("第一段" in c for c in chunks))
        self.assertTrue(any("第三段" in c for c in chunks))

    def test_chunk_pages_txt_ext(self):
        pages = [{"page": 1, "text": "甲。\n\n乙。\n\n" + ("丙。" * 50)}]
        chunks = chunk_pages(pages, chunk_size=60, overlap=0, source_ext="txt")
        self.assertTrue(chunks)
        self.assertEqual(chunks[0].get("strategy"), "recursive")

    def test_overlap_respects_chunk_size(self):
        parts = ["甲" * 25, "乙" * 25]
        size = 30
        out = _apply_overlap(parts, overlap=20, chunk_size=size)
        self.assertTrue(all(len(c) <= size for c in out))
        # 第二块在剩余 5 字空间内带上一段尾巴
        self.assertTrue(out[1].startswith("甲"))
        self.assertIn("乙", out[1])

    def test_merge_tiny_chunks(self):
        chunks = [
            "这是一段足够长的正文内容用来占位测试。",
            "短。",
            "又一段足够长的正文继续往下写一些字。",
        ]
        merged = _merge_tiny_chunks(chunks, chunk_size=30, min_ratio=0.3)
        self.assertEqual(len(merged), 2)
        self.assertIn("短。", merged[0])


if __name__ == "__main__":
    unittest.main()
