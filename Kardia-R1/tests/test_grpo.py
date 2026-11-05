import unittest

import numpy as np

from kardia import compute_advantages, grpo_surrogate_loss, grpo_ratio, k3_kl


class TestAdvantages(unittest.TestCase):
    def test_zero_mean(self):
        a = compute_advantages([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(float(a.mean()), 0.0, places=6)

    def test_unit_variance(self):
        a = compute_advantages([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(float(a.std()), 1.0, places=6)

    def test_constant_rewards_give_zero(self):
        a = compute_advantages([2.0, 2.0, 2.0, 2.0])
        self.assertTrue(np.allclose(a, 0.0))


class TestRatioAndKL(unittest.TestCase):
    def test_ratio(self):
        r = grpo_ratio(np.array([0.0]), np.array([-np.log(2.0)]))
        self.assertAlmostEqual(float(r[0]), 2.0)

    def test_k3_kl_nonnegative(self):
        kl = k3_kl(np.array([-1.0, -2.0]), np.array([-0.5, -1.0]))
        self.assertTrue(np.all(kl >= 0.0))

    def test_k3_kl_zero_when_equal(self):
        kl = k3_kl(np.array([-1.0]), np.array([-1.0]))
        self.assertAlmostEqual(float(kl[0]), 0.0, places=6)


class TestGRPOLoss(unittest.TestCase):
    def test_positive_advantage_prefers_higher_logprob(self):
        # Positive advantage: the loss should *decrease* as logprob increases.
        adv = np.array([1.0])
        ref = np.array([0.0])
        lp_low = grpo_surrogate_loss(np.array([-2.0]), ref, adv, beta=0.0)
        lp_high = grpo_surrogate_loss(np.array([0.0]), ref, adv, beta=0.0)
        self.assertLess(lp_high, lp_low)

    def test_clip_bounds_surrogate(self):
        # With a huge ratio the clipped objective should cap the update.
        adv = np.array([1.0])
        ref = np.array([0.0])
        huge_lp = np.array([10.0])  # ratio = e^10
        loss = grpo_surrogate_loss(huge_lp, ref, adv, beta=0.0, clip_epsilon=0.2)
        # surrogate clipped to (1+eps)*adv = 1.2, so loss = -1.2
        self.assertAlmostEqual(loss, -1.2, places=5)


if __name__ == "__main__":
    unittest.main()
