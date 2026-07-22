# 作者：yangkunpeng1
# 日期：2026-07-22
"""回答去重单测。"""

from __future__ import annotations

import unittest

from services.answer_cleanup import collapse_duplicate_lines, is_highly_repetitive


class AnswerCleanupTests(unittest.TestCase):
    def test_exact_paragraph_dup(self):
        text = (
            "核心架构：项目采用标准的三层架构设计（第4页）。\n"
            "核心架构：项目采用标准的三层架构设计（第4页）。\n"
            "技术实现：后端基于 Express（第3、4页）。\n"
            "核心架构：项目采用标准的三层架构设计（第4页）。\n"
            "技术实现：后端基于 Express（第3、4页）。\n"
            "功能验证：文档完整展示了用户认证系统。\n"
        )
        out = collapse_duplicate_lines(text)
        self.assertEqual(out.count("核心架构"), 1)
        self.assertEqual(out.count("技术实现"), 1)
        self.assertEqual(out.count("功能验证"), 1)

    def test_keeps_different_points(self):
        text = (
            "项目采用三层架构分离路由层（定义 URL）。\n"
            "项目采用三层架构分离控制器层（处理校验）。\n"
            "Prisma Client 基于 Schema 生成客户端。\n"
        )
        out = collapse_duplicate_lines(text)
        self.assertIn("路由层", out)
        self.assertIn("控制器层", out)
        self.assertIn("Prisma", out)

    def test_consecutive_same_sentence(self):
        text = (
            "核心架构：三层设计（第4页）。"
            "核心架构：三层设计（第4页）。"
            "技术实现：Express。"
        )
        out = collapse_duplicate_lines(text)
        self.assertEqual(out.count("核心架构"), 1)
        self.assertIn("技术实现", out)


if __name__ == "__main__":
    unittest.main()
