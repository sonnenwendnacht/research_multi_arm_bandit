import unittest

import numpy as np

from bandit_cost_quality.simulator import (
    PolicySpec, Scenario, cumulative_pseudo_regret, default_policies, reward_streams, simulate,
)


class SimulatorTests(unittest.TestCase):
    def setUp(self):
        self.scenario = Scenario("small", 0.5, (1.0, 2.0, 0.5), (0.4, 0.7, 0.6), (1, 1, 1))

    def test_horizon_history_counts_and_observed_means(self):
        policies = default_policies()
        policies += [PolicySpec("feature_score", exponent, random)
                     for exponent in (0, 0.5, 1) for random in (False, True)]
        for policy in policies:
            for horizon in (0, 1, 2, 3, 4, 100):
                with self.subTest(policy=policy, horizon=horizon):
                    result = simulate(self.scenario, horizon, 12, policy)
                    self.assertEqual(len(result.arms), horizon)
                    self.assertEqual(len(result.rewards), horizon)
                    self.assertEqual(int(result.counts.sum()), horizon)
                    np.testing.assert_array_equal(np.bincount(result.arms, minlength=3), result.counts)
                    for arm, count in enumerate(result.counts):
                        expected = result.rewards[result.arms == arm].mean() if count else 0.0
                        self.assertAlmostEqual(result.estimated_means[arm], expected)
                    quality, cost = cumulative_pseudo_regret(self.scenario, result.arms)
                    np.testing.assert_array_equal(result.quality_curve, quality)
                    np.testing.assert_array_equal(result.cost_curve, cost)

    def test_historical_double_count_regression(self):
        scenario = Scenario("five", 0.5, (5, 4, 3, 2, 1), (1, 1, 1, 1, 1), (0, 0, 0, 0, 0))
        policy = PolicySpec("feature_score", 0.5, weights=(0, -1, 0, 0, 0, 0, 0, 0))
        result = simulate(scenario, 100, 0, policy)
        # Historical code pulled only 95 times and omitted the first 9 from history.
        self.assertEqual(result.arms.size, 100)
        self.assertEqual(result.arms[:9].tolist(), [0, 1, 2, 3, 4, 0, 1, 2, 3])
        self.assertTrue(np.all(result.arms[9:] == 4))

    def test_hand_calculated_metrics_are_independent_of_reward_noise(self):
        scenario = Scenario("known", 0.5, (0, 2, 1), (0.2, 0.8, 0.5), (0, 0, 0))
        quality, cost = cumulative_pseudo_regret(scenario, [0, 1, 2, 0])
        np.testing.assert_allclose(quality, [0.3, 0.3, 0.3, 0.6])
        np.testing.assert_allclose(cost, [0, 1, 1, 1])
        self.assertEqual(scenario.optimal_arm, 2)

    def test_all_empirically_feasible_selects_cheapest_and_ties_by_index(self):
        scenario = Scenario("ties", 0.5, (9, 1, 1), (0.5, 0.5, 0.5), (0, 0, 0))
        for policy in default_policies()[:2]:
            result = simulate(scenario, 8, 0, policy)
            self.assertEqual(result.arms.tolist(), [0, 1, 2, 1, 1, 1, 1, 1])

    def test_empty_empirical_feasible_set_commits_to_best_observation(self):
        scenario = Scenario("fallback", 0.5, (0, 1), (0.8, 0.7), (1, 1))
        streams = np.array([[-2.0] * 8, [-1.0] * 8])
        result = simulate(scenario, 8, 0, PolicySpec("explore_then_commit"), streams)
        self.assertEqual(result.arms.tolist(), [0, 1, 1, 1, 1, 1, 1, 1])

    def test_ucb_empty_feasible_set_uses_largest_ucb(self):
        scenario = Scenario("fallback", 100, (0, 1), (101, 102), (1, 1))
        streams = np.array([[-2.0] * 3, [-1.0] * 3])
        result = simulate(scenario, 3, 0, PolicySpec("threshold_ucb"), streams)
        self.assertEqual(result.arms.tolist(), [0, 1, 1])

    def test_same_arm_pull_number_uses_same_reward_across_policies(self):
        streams = reward_streams(self.scenario, 100, 7)
        for policy in default_policies():
            result = simulate(self.scenario, 100, 7, policy, streams)
            for arm, count in enumerate(result.counts):
                np.testing.assert_array_equal(result.rewards[result.arms == arm], streams[arm, :count])

    def test_seed_reproducibility_and_global_rng_isolation(self):
        np.random.seed(823)
        state = np.random.get_state()
        policy = PolicySpec("feature_score", 0.9, True)
        first = simulate(self.scenario, 100, 9, policy)
        second = simulate(self.scenario, 100, 9, policy)
        after = np.random.get_state()
        np.testing.assert_array_equal(first.arms, second.arms)
        np.testing.assert_array_equal(first.rewards, second.rewards)
        self.assertEqual(state[0], after[0])
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])
        self.assertFalse(np.array_equal(reward_streams(self.scenario, 100, 9),
                                        reward_streams(self.scenario, 100, 10)))

    def test_selection_does_not_observe_hidden_means(self):
        other = Scenario("different_hidden_means", 0.5, self.scenario.costs, (3, -1, 2), (1, 1, 1))
        streams = reward_streams(self.scenario, 100, 8)
        for policy in default_policies():
            first = simulate(self.scenario, 100, 8, policy, streams)
            second = simulate(other, 100, 8, policy, streams)
            np.testing.assert_array_equal(first.arms, second.arms)
            np.testing.assert_array_equal(first.rewards, second.rewards)
            self.assertFalse(np.array_equal(first.quality_curve, second.quality_curve))

    def test_scenario_json_round_trip(self):
        self.assertEqual(Scenario.from_dict(self.scenario.as_dict()), self.scenario)

    def test_scenario_validation(self):
        cases = []
        for key, value in (("threshold", 0), ("threshold", float("nan")), ("threshold", True),
                           ("schema_version", 2), ("schema_version", True), ("arms", [])):
            data = self.scenario.as_dict()
            data[key] = value
            cases.append(data)
        for key, value in (("cost", -1), ("std", -1), ("mean", float("inf")), ("cost", "1")):
            data = self.scenario.as_dict()
            data["arms"][0][key] = value
            cases.append(data)
        missing = self.scenario.as_dict()
        del missing["arms"][0]["std"]
        cases.append(missing)
        extra = self.scenario.as_dict()
        extra["code"] = "untrusted code is not accepted"
        cases.append(extra)
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                Scenario.from_dict(case)
        with self.assertRaisesRegex(ValueError, "truly feasible"):
            Scenario("infeasible", 1, (1, 2), (0, 0), (1, 1))
        with self.assertRaises(ValueError):
            Scenario("mismatch", 1, (1,), (2, 2), (1,))

    def test_policy_and_simulation_input_validation(self):
        for alpha in (-1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                PolicySpec("feature_score", alpha)
        for weights in ((1,), (0,) * 7 + (float("inf"),)):
            with self.assertRaises(ValueError):
                PolicySpec("feature_score", weights=weights)
        for horizon in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                simulate(self.scenario, horizon, 0, PolicySpec("threshold_ucb"))
        with self.assertRaises(ValueError):
            simulate(self.scenario, 3, -1, PolicySpec("threshold_ucb"))
        with self.assertRaises(ValueError):
            simulate(self.scenario, 3, 0, PolicySpec("threshold_ucb"), np.zeros((3, 2)))
        for indices in ([-1], [3], [0.5], [[0]]):
            with self.assertRaises(ValueError):
                cumulative_pseudo_regret(self.scenario, indices)

    def test_numerical_extremes_are_finite_or_rejected(self):
        scenario = Scenario("extreme", 1, (1,), (1,), (1,))
        result = simulate(scenario, 2, 0, PolicySpec("explore_then_commit"), np.array([[1e308, -1e308]]))
        self.assertEqual(result.estimated_means[0], 0)
        largest = np.finfo(float).max
        repeated = simulate(scenario, 10, 0, PolicySpec("explore_then_commit"), np.full((1, 10), largest))
        self.assertEqual(repeated.estimated_means[0], largest)
        extreme = Scenario("extreme_regret", 1, (1, 1), (-1e308, 1), (0, 0))
        with self.assertRaisesRegex(ValueError, "pseudo-regret overflowed"):
            cumulative_pseudo_regret(extreme, [0, 0])


if __name__ == "__main__":
    unittest.main()
