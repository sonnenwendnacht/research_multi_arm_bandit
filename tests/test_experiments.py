import importlib.util
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from bandit_cost_quality.cli import load_json, main
from bandit_cost_quality.experiments import canonical_hash, compare, statistics
from bandit_cost_quality.simulator import Scenario, default_policies
from bandit_cost_quality.tuning import tune


ROOT = Path(__file__).resolve().parents[1]


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.scenario = Scenario("tiny", 0.5, (0, 1), (0.4, 0.7), (1, 1))

    def test_statistics_and_single_seed_uncertainty(self):
        result = statistics([1, 3])
        self.assertEqual(result["mean"], 2)
        self.assertAlmostEqual(result["standard_error"], 1)
        self.assertIsNone(statistics([1])["standard_error"])
        with self.assertRaisesRegex(ValueError, "aggregate overflowed"):
            statistics([1e308, 1e308])

    def test_complete_paired_experiment_and_provenance(self):
        result = compare(self.scenario, 50, [2, 3, 4], default_policies())
        self.assertEqual(result["scenario_sha256"], canonical_hash(self.scenario.as_dict()))
        self.assertIn("bandit_cost_quality/simulator.py", result["provenance"]["source_sha256"])
        for policy in default_policies():
            records = result["runs"][policy.name]
            self.assertTrue(all(record["pulls"] == 50 for record in records))
            for metric in ("quality_pseudo_regret", "cost_pseudo_regret"):
                expected = sum(record[metric] for record in records) / 3
                self.assertAlmostEqual(result["aggregates"][policy.name][metric]["mean"], expected)
            curves = result["aggregates"][policy.name]["curves"]
            self.assertEqual(curves["pulls"][-1], 50)
            self.assertAlmostEqual(curves["quality"]["mean"][-1],
                                   result["aggregates"][policy.name]["quality_pseudo_regret"]["mean"])
        baseline = result["runs"]["explore_then_commit"]
        other = result["runs"]["threshold_ucb"]
        differences = [run["quality_pseudo_regret"] - ref["quality_pseudo_regret"]
                       for run, ref in zip(other, baseline)]
        self.assertEqual(result["paired_differences"]["policies"]["threshold_ucb"]["quality_pseudo_regret"],
                         statistics(differences))
        json.dumps(result, allow_nan=False)

    def test_zero_horizon_and_seed_validation(self):
        result = compare(self.scenario, 0, [0], default_policies())
        self.assertEqual(result["aggregates"]["threshold_ucb"]["curves"]["pulls"], [])
        for seeds in ([], [0, 0], [-1], [True]):
            with self.assertRaises(ValueError):
                compare(self.scenario, 10, seeds, default_policies())

    def test_tuning_rejects_overlap_before_optional_import(self):
        with self.assertRaisesRegex(ValueError, "disjoint"):
            tune(self.scenario, 10, [1, 2], [2, 3], 2, 42)

    @unittest.skipUnless(importlib.util.find_spec("skopt"), "optional tuning dependency not installed")
    def test_bounded_tuning_uses_held_out_seeds_and_exact_trials(self):
        result = tune(self.scenario, 12, [0, 1], [100, 101], 6, 42)
        self.assertEqual(len(result["queries"]), 6)
        self.assertEqual(result["evaluation"]["seeds"], [100, 101])
        best = min(result["queries"], key=lambda query: query["objective"])
        self.assertEqual(result["selected_policy"], best["policy"])
        self.assertEqual(result["feature_names"][-1], "ucb_feasible")
        json.dumps(result, allow_nan=False)

    def test_json_rejects_executable_config_duplicates_and_nan(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            for text in ('{"threshold": NaN}', '{"x": 1, "x": 2}', 'threshold = 0.5'):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    load_json(path)

    def test_cli_smoke_and_non_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            command = [sys.executable, "-m", "bandit_cost_quality", "compare", "--scenario",
                       str(ROOT / "scenarios/file_1.json"), "--horizon", "20", "--seeds", "0", "1",
                       "--output", str(output)]
            run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            result = json.loads(output.read_text())
            self.assertEqual(result["horizon"], 20)
            self.assertEqual(result["seeds"], [0, 1])
            self.assertEqual(len(result["policies"]), 3)
            self.assertIn("policy", run.stdout)
            original = output.read_bytes()
            duplicate = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(duplicate.returncode, 0)
            self.assertEqual(output.read_bytes(), original)

    def test_serialization_failure_creates_neither_json_nor_plot(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            plot = Path(directory) / "plot.png"
            args = ["compare", "--scenario", str(ROOT / "scenarios/file_1.json"),
                    "--output", str(output), "--plot", str(plot)]
            with patch("bandit_cost_quality.cli.compare", return_value={"bad": float("nan")}), \
                    patch("bandit_cost_quality.cli.plot_comparison") as plotter, \
                    redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                main(args)
            self.assertEqual(raised.exception.code, 2)
            plotter.assert_not_called()
            self.assertFalse(output.exists())
            self.assertFalse(plot.exists())


if __name__ == "__main__":
    unittest.main()
