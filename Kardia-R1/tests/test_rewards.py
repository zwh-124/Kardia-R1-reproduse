import unittest

from kardia import (
    parse_spans,
    spans_to_text,
    format_reward,
    format_violations,
    emotion_matching_reward,
    rubric_reward,
    unified_reward,
)


class TestSpanFormat(unittest.TestCase):
    def setUp(self):
        self.good = spans_to_text(
            understanding="you feel anxious about work",
            reasoning="your INFP nature amplifies it",
            emotion="anxious",
            response="that sounds really tough",
        )

    def test_roundtrip(self):
        spans = parse_spans(self.good)
        self.assertEqual(spans["understanding"], "you feel anxious about work")
        self.assertEqual(spans["reasoning"], "your INFP nature amplifies it")
        self.assertEqual(spans["emotion"], "anxious")
        self.assertEqual(spans["response"], "that sounds really tough")

    def test_missing_span_is_none(self):
        spans = parse_spans("<|understanding_begin|>x<|understanding_end|>")
        self.assertIsNone(spans["reasoning"])


class TestFormatReward(unittest.TestCase):
    def test_perfect_format(self):
        out = spans_to_text("a", "b", "c", "d")
        self.assertEqual(format_reward(out), 1.0)
        self.assertEqual(format_violations(out), [])

    def test_empty_output(self):
        self.assertLess(format_reward(""), 1.0)

    def test_missing_tags(self):
        out = "no tags here"
        v = format_violations(out)
        self.assertIn("missing_understanding_begin", v)
        self.assertIn("wrong_order", v)

    def test_wrong_order(self):
        # response span placed before understanding
        out = (
            "<|response_begin|>d<|response_end|> "
            "<|understanding_begin|>a<|understanding_end|> "
            "<|reasoning_begin|>b<|reasoning_end|> "
            "<|emotion_begin|>c<|emotion_end|>"
        )
        self.assertIn("wrong_order", format_violations(out))

    def test_empty_span(self):
        out = spans_to_text("a", "b", "", "d")
        self.assertIn("empty_emotion", format_violations(out))

    def test_reward_in_unit_interval(self):
        for out in ["", "x", spans_to_text("a", "b", "c", "d")]:
            r = format_reward(out)
            self.assertGreaterEqual(r, 0.0)
            self.assertLessEqual(r, 1.0)


class TestEmotionReward(unittest.TestCase):
    def test_match(self):
        self.assertEqual(emotion_matching_reward("anxious", "anxious"), 1.0)

    def test_case_insensitive_match(self):
        self.assertEqual(emotion_matching_reward(" Anxious ", "ANXIOUS"), 1.0)

    def test_mismatch(self):
        self.assertEqual(emotion_matching_reward("sad", "anxious"), 0.0)

    def test_empty(self):
        self.assertEqual(emotion_matching_reward("", "anxious"), 0.0)


class TestRubricReward(unittest.TestCase):
    def test_perfect_scores(self):
        scores = {k: 5 for k in ("relevance", "fluency", "empathy",
                                 "persona_consistency", "safety")}
        self.assertEqual(rubric_reward(scores), 1.0)

    def test_min_scores(self):
        scores = {k: 1 for k in ("relevance", "fluency", "empathy",
                                 "persona_consistency", "safety")}
        self.assertEqual(rubric_reward(scores), 0.0)

    def test_mid_scores(self):
        scores = {k: 3 for k in ("relevance", "fluency", "empathy",
                                 "persona_consistency", "safety")}
        self.assertAlmostEqual(rubric_reward(scores), 0.5)


class TestUnifiedReward(unittest.TestCase):
    def test_equal_weights(self):
        # 1/3 * (1 + 1 + 1) = 1
        self.assertAlmostEqual(unified_reward(1.0, 1.0, 1.0), 1.0)

    def test_partial(self):
        # 1/3 * (0 + 1 + 1) = 2/3
        self.assertAlmostEqual(unified_reward(0.0, 1.0, 1.0), 2 / 3)


if __name__ == "__main__":
    unittest.main()
