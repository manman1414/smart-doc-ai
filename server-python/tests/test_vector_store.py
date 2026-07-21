# 作者：yangkunpeng1
# 日期：2026-07-21
"""检索模块单测（mock Chroma / embed，不启真实库）。"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from services.vector_store import (
    _score_from_distance,
    format_chunks_for_prompt,
    resolve_recall_k,
    resolve_sim_threshold,
    resolve_top_k,
    search_similar,
)


class ScoreConvertTests(unittest.TestCase):
    def test_cosine_distance(self):
        self.assertAlmostEqual(_score_from_distance(0.0, "cosine"), 1.0)
        self.assertAlmostEqual(_score_from_distance(0.2, "cosine"), 0.8)

    def test_l2_unit_vector(self):
        self.assertAlmostEqual(_score_from_distance(0.0, "l2"), 1.0)
        self.assertGreater(_score_from_distance(0.5, "l2"), 0.8)


class AdaptiveRecallTests(unittest.TestCase):
    @patch("services.vector_store.doc_chunk_count", return_value=8)
    def test_small_doc_recalls_all(self, _cnt):
        # floor = max(3*3, 8)=9；n=8≤9 → 全召回
        self.assertEqual(resolve_recall_k("d1", final_k=3), 8)

    @patch("services.vector_store.doc_chunk_count", return_value=100)
    def test_medium_doc_uses_ratio(self, _cnt):
        # floor=9, by_ratio=ceil(100*0.12)=12, by_topk=15 → 15
        self.assertEqual(resolve_recall_k("d1", final_k=3), 15)

    @patch("services.vector_store.doc_chunk_count", return_value=500)
    def test_large_doc_capped(self, _cnt):
        # by_ratio=60 → 受上限默认 50
        self.assertEqual(resolve_recall_k("d1", final_k=3), 50)

    def test_explicit_override(self):
        self.assertEqual(resolve_recall_k("d1", final_k=3, recall_k=7), 7)


class AdaptiveTopKTests(unittest.TestCase):
    @patch("services.vector_store.doc_chunk_count", return_value=8)
    def test_small_doc_uses_min(self, _cnt):
        self.assertEqual(resolve_top_k("d1"), 2)

    @patch("services.vector_store.doc_chunk_count", return_value=40)
    def test_medium_doc(self, _cnt):
        self.assertEqual(resolve_top_k("d1"), 3)

    @patch("services.vector_store.doc_chunk_count", return_value=300)
    def test_large_doc_capped(self, _cnt):
        self.assertEqual(resolve_top_k("d1"), 5)

    def test_respects_min_max_and_candidates(self):
        self.assertEqual(resolve_top_k("d1", top_k=10), 5)  # clamp to MAX
        self.assertEqual(resolve_top_k("d1", top_k=1), 2)  # clamp to MIN
        self.assertEqual(resolve_top_k(None, top_k=4, candidate_n=2), 2)
        self.assertEqual(resolve_top_k(None, top_k=3, candidate_n=0), 0)
        self.assertEqual(resolve_top_k(None, top_k=3, candidate_n=1), 1)  # 不足下限


class SimThresholdTests(unittest.TestCase):
    def test_small_doc_looser_than_large(self):
        large = resolve_sim_threshold(100, base=0.35, use_rerank=False)
        small = resolve_sim_threshold(3, base=0.35, use_rerank=False)
        self.assertLess(small, large)
        self.assertAlmostEqual(small, 0.05)

    def test_rerank_caps_large_doc(self):
        t = resolve_sim_threshold(100, base=0.35, use_rerank=True)
        self.assertAlmostEqual(t, 0.12)

    def test_lexical_boost_question_in_page(self):
        from services.vector_store import _lexical_boost

        text = "1. 地球绕太阳公转一圈大约需要多少天？\nB. 约 365 天\n答案：B"
        b = _lexical_boost("地球绕太阳公转一圈大约需要多少天？", text)
        self.assertGreaterEqual(b, 0.5)


class SearchSimilarTests(unittest.TestCase):
    @patch("services.reranker.rerank_enabled", return_value=False)
    @patch("services.vector_store.collection")
    @patch("services.embedder.embed_query", return_value=[0.1, 0.2])
    def test_query_with_where_and_threshold(self, _embed, coll, _rerank_off):
        coll.metadata = {"hnsw:space": "cosine"}
        coll.query.return_value = {
            "documents": [["高相关片段", "低相关片段"]],
            "metadatas": [[{"page": 2, "kind": "text"}, {"page": 3, "kind": "text"}]],
            "distances": [[0.1, 0.8]],
        }
        hits = search_similar("合同金额", doc_id="doc-1", top_k=3, min_score=0.35)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["page"], 2)
        self.assertGreaterEqual(hits[0]["score"], 0.35)
        kwargs = coll.query.call_args.kwargs
        self.assertEqual(kwargs["where"], {"doc_id": {"$eq": "doc-1"}})
        self.assertIn("distances", kwargs["include"])
        self.assertGreaterEqual(kwargs["n_results"], 3)

    @patch("services.reranker.rerank_enabled", return_value=False)
    @patch("services.vector_store.collection")
    @patch("services.embedder.embed_query", return_value=[0.1, 0.2])
    def test_empty_query(self, _embed, coll, _rerank_off):
        self.assertEqual(search_similar("  "), [])
        coll.query.assert_not_called()

    @patch("services.reranker.rerank_enabled", return_value=False)
    @patch("services.vector_store.collection")
    @patch("services.embedder.embed_query", return_value=[0.1, 0.2])
    def test_query_exception_returns_empty(self, _embed, coll, _rerank_off):
        coll.metadata = {}
        coll.query.side_effect = Exception("no results")
        coll.get.return_value = {"ids": []}
        self.assertEqual(search_similar("问题", doc_id="x"), [])

    @patch("services.reranker.rerank_enabled", return_value=False)
    @patch("services.vector_store.collection")
    @patch("services.embedder.embed_query", return_value=[0.1, 0.2])
    def test_recall_then_top_k_and_dedupe(self, _embed, coll, _rerank_off):
        coll.metadata = {"hnsw:space": "cosine"}
        docs = [
            "合同金额为十万元整补充说明一二",
            "合同金额为十万元整补充说明一二三四",
            "交付周期为三十个工作日完整描述",
            "验收标准另行约定完整描述文字",
            "无关但分数够的另一段完整描述",
        ]
        coll.query.return_value = {
            "documents": [docs],
            "metadatas": [[{"page": i + 1, "kind": "text"} for i in range(5)]],
            "distances": [[0.05, 0.06, 0.1, 0.12, 0.15]],
        }
        hits = search_similar(
            "合同金额",
            doc_id="doc-1",
            top_k=2,
            recall_k=5,
            min_score=0.3,
        )
        self.assertEqual(len(hits), 2)
        kwargs = coll.query.call_args.kwargs
        self.assertEqual(kwargs["n_results"], 5)
        texts = [h["text"] for h in hits]
        self.assertTrue(any("十万元" in t for t in texts))
        self.assertEqual(sum(1 for t in texts if "十万元" in t), 1)


class FormatChunksTests(unittest.TestCase):
    def test_keeps_page_marker(self):
        text = format_chunks_for_prompt(
            [{"text": "合同金额十万", "page": 3}, {"text": "交付周期", "page": 1}]
        )
        self.assertIn("【第3页】", text)
        self.assertIn("【第1页】", text)

    def test_txt_skips_page_marker(self):
        text = format_chunks_for_prompt(
            [{"text": "纯文本内容", "page": 1, "source_ext": "txt"}]
        )
        self.assertNotIn("【第", text)
        self.assertIn("纯文本内容", text)

    def test_page_zero_skips_marker(self):
        text = format_chunks_for_prompt([{"text": "无页码片段", "page": 0}])
        self.assertNotIn("【第", text)
        self.assertIn("无页码片段", text)


if __name__ == "__main__":
    unittest.main()
