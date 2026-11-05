import unittest

from kardia import (
    Dialogue,
    Turn,
    UserProfile,
    partition_dialogues,
    is_easy,
)
from kardia.prompts import build_sft_prompt, build_rubric_prompt, parse_rubric_output


class TestPartition(unittest.TestCase):
    def test_is_easy_cond1(self):
        # p_fail < p_pass + p_solved
        self.assertTrue(is_easy(0.2, 0.3, 0.5))

    def test_is_easy_cond2(self):
        # p_fail + p_pass < p_solved
        self.assertTrue(is_easy(0.1, 0.1, 0.8))

    def test_is_hard(self):
        # failure dominant: 0.6 >= 0.4 and 0.6+0.2=0.8 >= 0.2
        self.assertFalse(is_easy(0.6, 0.2, 0.2))

    def test_partition(self):
        def d(id_, pf, pp, ps):
            return Dialogue(dialogue_id=id_, profile=UserProfile("u"), p_fail=pf,
                            p_pass=pp, p_solved=ps)

        dialogues = [
            d("e1", 0.2, 0.3, 0.5),
            d("h1", 0.6, 0.2, 0.2),
            d("e2", 0.1, 0.1, 0.8),
        ]
        easy, hard = partition_dialogues(dialogues)
        self.assertEqual([x.dialogue_id for x in easy], ["e1", "e2"])
        self.assertEqual([x.dialogue_id for x in hard], ["h1"])


class TestPrompts(unittest.TestCase):
    def test_sft_prompt_contains_profile_and_tags(self):
        prof = UserProfile(user_id="a008b489", mbti="ISFJ", about="likes music")
        turn = Turn(query="Do you think it's silly?", understanding="u",
                    reasoning="r", emotion="hopeful", response="res")
        dialogue = Dialogue("d1", prof, situation="waiting for good news",
                            turns=[turn])
        prompt = build_sft_prompt(dialogue)
        self.assertIn("a008b489", prompt)
        self.assertIn("ISFJ", prompt)
        self.assertIn("Do you think it's silly?", prompt)
        # target spans must NOT appear (they are the label)
        self.assertNotIn("hopeful", prompt)

    def test_rubric_prompt_and_parse(self):
        prof = UserProfile(user_id="u1", mbti="INFP")
        prompt = build_rubric_prompt(prof, "I feel anxious", "that sounds hard")
        self.assertIn("relevance", prompt)
        self.assertIn("persona_consistency", prompt)
        parsed = parse_rubric_output(
            '{"relevance": 4, "fluency": 5, "empathy": 4, '
            '"persona_consistency": 3, "safety": 5}'
        )
        self.assertEqual(parsed["fluency"], 5)

    def test_parse_rubric_with_prose(self):
        out = 'Here are scores: {"relevance": 4, "fluency": 5}'
        parsed = parse_rubric_output(out)
        self.assertEqual(parsed["relevance"], 4)


if __name__ == "__main__":
    unittest.main()
