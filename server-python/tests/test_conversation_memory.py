# 作者：yangkunpeng1
# 日期：2026-07-22
"""多轮记忆：拆分 / 拼装（不依赖真实 LLM）。"""

from __future__ import annotations

import os
import unittest

from services.conversation_memory import (
    build_history_prompt_messages,
    format_turns_text,
    history_keep_messages,
    history_keep_turns,
    merge_memory_summary,
    normalize_history,
    rewrite_question,
    split_history,
)


class ConversationMemoryTests(unittest.TestCase):
    def test_keep_turns_clamped(self):
        os.environ["SMARTDOC_HISTORY_TURNS"] = "2"
        self.assertEqual(history_keep_turns(), 4)
        os.environ["SMARTDOC_HISTORY_TURNS"] = "99"
        self.assertEqual(history_keep_turns(), 6)
        os.environ["SMARTDOC_HISTORY_TURNS"] = "5"
        self.assertEqual(history_keep_turns(), 5)
        self.assertEqual(history_keep_messages(), 10)

    def test_split_keeps_recent(self):
        hist = []
        for i in range(8):
            hist.append({"role": "user", "content": f"u{i}"})
            hist.append({"role": "assistant", "content": f"a{i}"})
        older, recent = split_history(hist, keep_messages=6)
        self.assertEqual(len(recent), 6)
        self.assertEqual(len(older), 10)
        self.assertEqual(recent[-1]["content"], "a7")
        self.assertEqual(older[0]["content"], "u0")

    def test_split_short_history(self):
        hist = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好"},
        ]
        older, recent = split_history(hist, keep_messages=10)
        self.assertEqual(older, [])
        self.assertEqual(len(recent), 2)

    def test_normalize_filters(self):
        out = normalize_history(
            [
                {"role": "system", "content": "x"},
                {"role": "user", "content": "  "},
                {"role": "user", "content": "问"},
            ]
        )
        self.assertEqual(out, [{"role": "user", "content": "问"}])

    def test_merge_summary_via_chat_fn(self):
        def fake_chat(_msgs, _max_tokens):
            return "用户关注合同金额与交付日期。"

        text = merge_memory_summary(
            "旧摘要",
            [{"role": "user", "content": "金额多少"}, {"role": "assistant", "content": "十万"}],
            fake_chat,
        )
        self.assertIn("合同金额", text)

    def test_rewrite_question(self):
        def fake_chat(_msgs, _max_tokens):
            return "采购合同的总金额是多少？"

        q = rewrite_question(
            "那金额呢？",
            memory_summary="在聊采购合同",
            recent_messages=[{"role": "user", "content": "合同主题是什么"}],
            chat_fn=fake_chat,
        )
        self.assertIn("金额", q)

    def test_build_prompt_messages(self):
        msgs = build_history_prompt_messages(
            "早期关注交付",
            [
                {"role": "user", "content": "交货期？"},
                {"role": "assistant", "content": "30天"},
            ],
        )
        self.assertEqual(msgs[0]["role"], "system")
        self.assertIn("早期关注交付", msgs[0]["content"])
        self.assertEqual(len(msgs), 3)

    def test_format_turns(self):
        t = format_turns_text(
            [{"role": "user", "content": "A"}, {"role": "assistant", "content": "B"}]
        )
        self.assertIn("用户：A", t)
        self.assertIn("助手：B", t)


if __name__ == "__main__":
    unittest.main()
