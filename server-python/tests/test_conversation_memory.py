# 作者：yangkunpeng1
# 日期：2026-07-23
"""多轮记忆：拆分 / 硬事实 / 拼装（不依赖真实 LLM）。"""

from __future__ import annotations

import os
import unittest

from services.conversation_memory import (
    build_history_prompt_messages,
    format_turns_text,
    history_keep_messages,
    history_keep_turns,
    merge_fact_lines,
    merge_memory_facts,
    merge_memory_summary,
    normalize_history,
    parse_fact_lines,
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

    def test_merge_facts_append_only(self):
        def fake_chat(_msgs, _max_tokens):
            return "- 订单号：PO-001\n- [pinned] 金额：10万"

        text = merge_memory_facts(
            "- 公司：云启",
            [{"role": "user", "content": "单号是 PO-001"}, {"role": "assistant", "content": "好的"}],
            fake_chat,
        )
        self.assertIn("公司：云启", text)
        self.assertIn("订单号：PO-001", text)
        self.assertIn("[pinned] 金额：10万", text)

    def test_merge_fact_lines_dedupe_and_pin_upgrade(self):
        merged = merge_fact_lines(
            "- 订单号：PO-001\n- 金额：10万",
            ["[pinned] 金额：10万", "订单号：PO-001", "交货期：30天"],
        )
        lines = parse_fact_lines(merged)
        self.assertEqual(len(lines), 3)
        self.assertTrue(any(x.startswith("[pinned]") and "金额" in x for x in lines))
        self.assertTrue(any("交货期" in x for x in lines))

    def test_rewrite_question(self):
        def fake_chat(_msgs, _max_tokens):
            return "采购合同的总金额是多少？"

        q = rewrite_question(
            "那金额呢？",
            memory_summary="在聊采购合同",
            memory_facts="- 合同名：采购合同",
            recent_messages=[{"role": "user", "content": "合同主题是什么"}],
            chat_fn=fake_chat,
        )
        self.assertIn("金额", q)

    def test_build_prompt_messages_with_facts(self):
        msgs = build_history_prompt_messages(
            "早期关注交付",
            [
                {"role": "user", "content": "交货期？"},
                {"role": "assistant", "content": "30天"},
            ],
            memory_facts="- 订单号：PO-9",
        )
        self.assertEqual(msgs[0]["role"], "system")
        self.assertIn("硬事实", msgs[0]["content"])
        self.assertIn("PO-9", msgs[0]["content"])
        self.assertIn("早期关注交付", msgs[1]["content"])
        self.assertEqual(len(msgs), 4)

    def test_final_trim_recent_and_clip(self):
        from services.conversation_memory import (
            build_history_prompt_messages,
            clip_text,
            trim_recent_for_final,
        )

        os.environ["SMARTDOC_FINAL_RECENT_MSGS"] = "2"
        os.environ["SMARTDOC_FINAL_MSG_CHARS"] = "10"
        os.environ["SMARTDOC_FINAL_SUMMARY_CHARS"] = "20"
        os.environ["SMARTDOC_FINAL_FACTS_CHARS"] = "30"
        long_hist = []
        for i in range(6):
            long_hist.append({"role": "user", "content": f"用户问题内容很长{i}" * 5})
            long_hist.append({"role": "assistant", "content": f"助手回答内容很长{i}" * 5})
        trimmed = trim_recent_for_final(long_hist)
        self.assertEqual(len(trimmed), 2)
        self.assertTrue(all(len(m["content"]) <= 10 for m in trimmed))

        msgs = build_history_prompt_messages(
            "摘要" * 50,
            long_hist,
            memory_facts="事实" * 50,
            for_final=True,
            question="",
        )
        self.assertLessEqual(len(msgs), 4)
        self.assertLessEqual(len(msgs[0]["content"]), 80)
        self.assertEqual(clip_text("abcdefghij", 5), "abcd…")

    def test_retrieve_relevant_facts_and_summary(self):
        from unittest.mock import patch

        from services.conversation_memory import (
            retrieve_relevant_facts,
            retrieve_relevant_summary,
        )

        facts = (
            "- 订单号：PO-001\n"
            "- 交货期：30天\n"
            "- [pinned] 币种：人民币\n"
            "- 天气：晴朗"
        )
        with patch(
            "services.conversation_memory._score_texts",
            side_effect=lambda q, texts: [
                0.9 if "交货" in t else (0.8 if "订单" in t else 0.1) for t in texts
            ],
        ):
            out = retrieve_relevant_facts("交货多久", facts, top_k=1)
        self.assertIn("交货期", out)
        self.assertIn("币种", out)  # pinned 必留
        self.assertNotIn("天气", out)

        summary = "用户在聊采购合同。接着确认了交货周期。后来又问了发票抬头。"
        with patch(
            "services.conversation_memory._score_texts",
            side_effect=lambda q, texts: [
                0.9 if "交货" in t else 0.2 for t in texts
            ],
        ):
            s = retrieve_relevant_summary("交货相关", summary, top_sents=1)
        self.assertIn("交货", s)
        self.assertNotIn("发票", s)

    def test_format_turns(self):
        t = format_turns_text(
            [{"role": "user", "content": "A"}, {"role": "assistant", "content": "B"}]
        )
        self.assertIn("用户：A", t)
        self.assertIn("助手：B", t)


if __name__ == "__main__":
    unittest.main()
