# 作者：yangkunpeng1
# 日期：2026-07-21
"""重排模块单测（mock CrossEncoder，不下载模型）。"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from services.reranker import _sigmoid, rerank_chunks, rerank_pairs


class RerankerUnitTests(unittest.TestCase):
    def test_sigmoid_bounds(self):
        self.assertGreater(_sigmoid(10), 0.99)
        self.assertLess(_sigmoid(-10), 0.01)
        self.assertAlmostEqual(_sigmoid(0), 0.5)

    @patch("services.reranker.get_reranker")
    def test_rerank_pairs_order(self, get_model):
        model = MagicMock()
        # 第二段更高分
        model.predict.return_value = [0.0, 2.0, -1.0]
        get_model.return_value = model

        ranked = rerank_pairs("问", ["a", "b", "c"], top_k=2, min_score=0.0)
        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked[0][0], 1)  # "b"
        self.assertGreater(ranked[0][1], ranked[1][1])

    @patch("services.reranker.get_reranker")
    def test_rerank_chunks_keeps_meta(self, get_model):
        model = MagicMock()
        model.predict.return_value = [1.0, 3.0]
        get_model.return_value = model

        chunks = [
            {"text": "次相关", "page": 1, "score": 0.9, "kind": "text"},
            {"text": "最相关", "page": 2, "score": 0.5, "kind": "table"},
        ]
        out = rerank_chunks("问题", chunks, top_k=1, min_score=0.0)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["text"], "最相关")
        self.assertEqual(out[0]["page"], 2)
        self.assertEqual(out[0]["kind"], "table")
        self.assertIn("dense_score", out[0])


class SearchWithRerankTests(unittest.TestCase):
    @patch("services.reranker.rerank_enabled", return_value=True)
    @patch("services.vector_store.collection")
    @patch("services.embedder.embed_query", return_value=[0.1, 0.2])
    @patch("services.reranker.get_reranker")
    def test_search_uses_rerank(self, get_reranker, _embed, coll, _on):
        from services.vector_store import search_similar

        coll.metadata = {"hnsw:space": "cosine"}
        coll.query.return_value = {
            "documents": [["片段甲内容足够长", "片段乙内容足够长", "片段丙内容足够长"]],
            "metadatas": [
                [
                    {"page": 1, "kind": "text"},
                    {"page": 2, "kind": "text"},
                    {"page": 3, "kind": "text"},
                ]
            ],
            "distances": [[0.2, 0.25, 0.3]],
        }
        model = MagicMock()
        # 让乙最高
        model.predict.return_value = [0.0, 4.0, 1.0]
        get_reranker.return_value = model

        hits = search_similar("测试", doc_id="d1", top_k=2, recall_k=3, min_score=0.0)
        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0]["text"], "片段乙内容足够长")
        self.assertEqual(hits[0]["page"], 2)


if __name__ == "__main__":
    unittest.main()
