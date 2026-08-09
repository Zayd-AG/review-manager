"""Tests for the local model response validator."""

from __future__ import annotations

import unittest

from backend.app.services.classifier import parse_label


class LabelParsingTests(unittest.TestCase):
    def test_accepts_valid_json_label(self) -> None:
        label = parse_label(
            '{"category":"bug","severity":"high","justification":"The review reports a failure."}'
        )
        self.assertEqual(label["category"], "bug")
        self.assertEqual(label["severity"], "high")

    def test_rejects_unknown_category(self) -> None:
        with self.assertRaises(ValueError):
            parse_label(
                '{"category":"unknown","severity":"high","justification":"Invalid category."}'
            )


if __name__ == "__main__":
    unittest.main()
