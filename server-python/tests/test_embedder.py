# 作者：yangkunpeng1
# 日期：2026-07-21
"""嵌入模块轻量单测（不加载 BGE 模型）。"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from services.embedder import (
    QUERY_INSTRUCTION,
    embed_documents,
    embed_query,
    format_query_for_embed,
)


class EmbedderTests(unittest.TestCase):
    def test_format_query_adds_instruction(self):
        q = format_query_for_embed("合同金额多少")
        self.assertTrue(q.startswith(QUERY_INSTRUCTION))
        self.assertIn("合同金额多少", q)

    def test_format_query_idempotent(self):
        once = format_query_for_embed("你好")
        twice = format_query_for_embed(once)
        self.assertEqual(once.count(QUERY_INSTRUCTION), 1)
        self.assertEqual(twice, once)

    def test_format_query_empty(self):
        self.assertEqual(format_query_for_embed("  "), "")

    @patch("services.embedder.get_model")
    def test_embed_documents_no_query_prefix(self, get_model):
        model = MagicMock()
        model.encode.return_value = MagicMock(tolist=lambda: [[0.1, 0.2], [0.3, 0.4]])
        get_model.return_value = model

        out = embed_documents(["文档甲", "文档乙"])
        self.assertEqual(len(out), 2)
        args, kwargs = model.encode.call_args
        self.assertEqual(list(args[0]), ["文档甲", "文档乙"])
        self.assertTrue(kwargs.get("normalize_embeddings"))

    @patch("services.embedder.get_model")
    def test_embed_query_uses_instruction(self, get_model):
        model = MagicMock()
        model.encode.return_value = MagicMock(tolist=lambda: [[0.5, 0.6]])
        get_model.return_value = model

        embed_query("价款")
        args, _kwargs = model.encode.call_args
        self.assertEqual(args[0][0], QUERY_INSTRUCTION + "价款")


if __name__ == "__main__":
    unittest.main()
