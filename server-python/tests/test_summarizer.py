# 作者：yangkunpeng1
# 日期：2026-07-21
"""摘要分段逻辑单测（不调 LLM）。"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from services.summarizer import (
    build_merge_prompt,
    build_section_prompt,
    build_single_prompt,
    split_for_summary,
    summarize_document,
)


class SplitForSummaryTests(unittest.TestCase):
    def test_short_text_single_section(self):
        text = "这是一段不长的文档内容。" * 5
        parts = split_for_summary(text, section_size=2500, max_sections=8)
        self.assertEqual(len(parts), 1)

    def test_page_markers_packed(self):
        pages = [f"【第{i}页】\n本页内容{'字' * 40}" for i in range(1, 6)]
        text = "\n".join(pages)
        parts = split_for_summary(text, section_size=200, max_sections=8)
        self.assertGreaterEqual(len(parts), 2)
        self.assertTrue(any("【第1页】" in p for p in parts))

    def test_max_sections_samples_evenly(self):
        pages = [f"【第{i}页】\n{'内容' * 80}" for i in range(1, 21)]
        text = "\n".join(pages)
        parts = split_for_summary(text, section_size=150, max_sections=4)
        self.assertLessEqual(len(parts), 4)
        joined = "\n".join(parts)
        self.assertIn("【第1页】", joined)
        self.assertIn("【第20页】", joined)


class SummarizeDocumentTests(unittest.TestCase):
    @patch("services.summarizer._section_chars", return_value=200)
    @patch("services.summarizer._max_sections", return_value=8)
    def test_map_reduce_calls_llm(self, _max_sec, _sec_chars):
        calls: list[str] = []

        def fake_llm(prompt: str) -> str:
            calls.append(prompt)
            if "部分" in prompt and "要点" in prompt:
                return "分段要点"
            if "不同部分的要点" in prompt:
                return "最终摘要全文覆盖"
            return "单段摘要"

        text = "【第1页】\n" + ("甲" * 300) + "\n【第2页】\n" + ("乙" * 300)
        out = summarize_document(text, llm_complete=fake_llm)
        self.assertTrue(out)
        self.assertGreaterEqual(len(calls), 2)
        self.assertIn("最终摘要", out)

    def test_single_shot(self):
        def fake_llm(prompt: str) -> str:
            self.assertIn("核心内容", prompt)
            return "短摘要"

        out = summarize_document("很短的文档。", llm_complete=fake_llm)
        self.assertEqual(out, "短摘要")

    def test_prompts_non_empty(self):
        self.assertIn("要点", build_section_prompt("x", 1, 2))
        self.assertIn("摘要", build_merge_prompt(["a", "b"]))
        self.assertIn("核心内容", build_single_prompt("hello"))


if __name__ == "__main__":
    unittest.main()
