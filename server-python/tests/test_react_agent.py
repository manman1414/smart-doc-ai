# 作者：yangkunpeng1
# 日期：2026-07-23
"""ReAct 调度：含快路径 / search_chat（不依赖真实 LLM / 向量库）。"""

from __future__ import annotations

import json
import unittest

from services.react_agent import (
    looks_like_doc_query,
    looks_like_memory_query,
    parse_react_decision,
    run_react,
)


class ReactAgentTests(unittest.TestCase):
    def test_parse_json_decision(self):
        thought, action, inp = parse_react_decision(
            json.dumps(
                {
                    "thought": "需要查单号",
                    "action": "search_doc",
                    "action_input": "订单号是多少",
                },
                ensure_ascii=False,
            )
        )
        self.assertEqual(action, "search_doc")
        self.assertIn("订单", inp)
        self.assertIn("单号", thought)

    def test_parse_search_chat(self):
        _, action, inp = parse_react_decision(
            json.dumps(
                {"thought": "翻旧对话", "action": "search_chat", "action_input": "交货期"},
                ensure_ascii=False,
            )
        )
        self.assertEqual(action, "search_chat")
        self.assertIn("交货", inp)

    def test_hint_classifiers(self):
        self.assertTrue(looks_like_doc_query("合同金额是多少"))
        self.assertTrue(looks_like_memory_query("我们之前聊了啥"))
        self.assertFalse(looks_like_doc_query("你好"))

    def test_doc_fast_path_no_planner(self):
        calls = {"planner": 0, "search": 0}

        def chat_fn(_msgs, _max_tokens):
            calls["planner"] += 1
            return json.dumps({"thought": "x", "action": "finish", "action_input": ""})

        def search_fn(query: str):
            calls["search"] += 1
            return [{"text": f"命中:{query}", "page": 1, "score": 0.9}]

        result = run_react(
            "采购合同金额是多少？",
            memory_summary="在聊合同",
            memory_facts="- 合同名：采购合同",
            recent_messages=[],
            chat_fn=chat_fn,
            search_fn=search_fn,
            max_steps=3,
        )
        self.assertEqual(calls["planner"], 0)
        self.assertEqual(calls["search"], 1)
        self.assertEqual(result.finish_reason, "fast_doc")
        self.assertTrue(result.chunks)

    def test_memory_fast_path(self):
        def search_fn(_query: str):
            raise AssertionError("回顾类不应搜文档")

        def search_chat_fn(query: str):
            return [{"text": "用户：交货多久\n助手：30天", "score": 0.88, "source": "chat"}]

        def chat_fn(_msgs, _max_tokens):
            raise AssertionError("快路径不应调用规划 LLM")

        result = run_react(
            "我们之前聊了啥？",
            memory_summary="讨论了交付周期",
            memory_facts="- 交货期：30天",
            recent_messages=[],
            chat_fn=chat_fn,
            search_fn=search_fn,
            search_chat_fn=search_chat_fn,
            max_steps=3,
        )
        self.assertEqual(result.finish_reason, "fast_memory")
        self.assertTrue(result.used_memory)
        self.assertTrue(result.chat_chunks)


if __name__ == "__main__":
    unittest.main()
