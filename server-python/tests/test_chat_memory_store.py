# 作者：yangkunpeng1
# 日期：2026-07-23
"""对话溢出向量：index / search / delete（mock embed 与 chroma）。"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch


class ChatMemoryStoreTests(unittest.TestCase):
    @patch("services.chat_memory_store.embed_documents")
    @patch("services.chat_memory_store.chat_collection")
    def test_index_overflow(self, mock_col, mock_embed):
        from services.chat_memory_store import index_overflow_messages

        mock_embed.return_value = [[0.1, 0.2], [0.3, 0.4]]
        n = index_overflow_messages(
            "conv-1",
            [
                {"role": "user", "content": "交货多久"},
                {"role": "assistant", "content": "30天"},
            ],
            start_index=4,
            doc_id="doc-a",
        )
        self.assertEqual(n, 2)
        mock_col.upsert.assert_called_once()
        kwargs = mock_col.upsert.call_args.kwargs
        self.assertEqual(len(kwargs["ids"]), 2)
        self.assertTrue(all(i.startswith("chat:conv-1:") for i in kwargs["ids"]))
        self.assertEqual(kwargs["metadatas"][0]["msg_index"], 4)
        self.assertEqual(kwargs["metadatas"][1]["msg_index"], 5)

    @patch("services.chat_memory_store.embed_query")
    @patch("services.chat_memory_store._collection_space", return_value="l2")
    @patch("services.chat_memory_store.chat_collection")
    def test_search_chat_turns(self, mock_col, _space, mock_qemb):
        from services.chat_memory_store import search_chat_turns

        mock_col.get.return_value = {"ids": ["a", "b"]}
        mock_qemb.return_value = [0.1, 0.2]
        mock_col.query.return_value = {
            "documents": [["用户：交货多久"]],
            "metadatas": [[{"role": "user", "msg_index": 4}]],
            "distances": [[0.2]],
        }
        hits = search_chat_turns("交货", "conv-1", top_k=2)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["source"], "chat")
        self.assertIn("交货", hits[0]["text"])

    def test_index_requires_conversation_id(self):
        from services.chat_memory_store import index_overflow_messages

        self.assertEqual(
            index_overflow_messages("", [{"role": "user", "content": "x"}]),
            0,
        )


if __name__ == "__main__":
    unittest.main()
