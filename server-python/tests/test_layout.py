# 作者：yangkunpeng1
# 日期：2026-07-21
"""版面分析单测。"""

from __future__ import annotations

import unittest

from services.layout import choose_strategy, merge_reading_order


class LayoutTests(unittest.TestCase):
    def test_choose_full_ocr(self):
        s, _ = choose_strategy("", 0.9, {"image_regions": [], "max_image_ratio": 0.9})
        self.assertEqual(s, "full_ocr")

    def test_choose_digital(self):
        text = "这是一段足够长的可检索数字层正文内容，用来模拟正常PDF文字页。" * 3
        s, _ = choose_strategy(text, 0.05, {"image_regions": [], "max_image_ratio": 0.0})
        self.assertEqual(s, "digital")

    def test_choose_hybrid(self):
        text = "这是一段足够长的可检索数字层正文内容，用来模拟正常PDF文字页。" * 3
        layout = {
            "image_regions": [
                {
                    "bbox": (100, 400, 400, 700),
                    "area_ratio": 0.2,
                    "y": 400,
                    "x": 100,
                }
            ],
            "max_image_ratio": 0.2,
        }
        s, reason = choose_strategy(text, 0.2, layout)
        self.assertEqual(s, "hybrid", reason)

    def test_merge_reading_order(self):
        text = merge_reading_order(
            [
                {"y": 200, "x": 10, "text": "下面"},
                {"y": 50, "x": 10, "text": "上面"},
                {"y": 50, "x": 100, "text": "右侧"},
            ]
        )
        self.assertEqual(text.split("\n\n")[0], "上面")
        self.assertIn("右侧", text)
        self.assertTrue(text.endswith("下面"))


if __name__ == "__main__":
    unittest.main()
