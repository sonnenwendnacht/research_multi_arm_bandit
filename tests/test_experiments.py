import importlib.util
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from bandit_cost_quality.cli import load_json, main
from bandit_cost_quality.experiments import canonical_hash, compare, plot_comparison, statistics
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

    def test_extensionless_plot_preserves_existing_actual_target_before_computation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / "figure.png"
            existing.write_bytes(b"existing plot")
            output = root / "result.json"
            args = ["compare", "--scenario", str(ROOT / "scenarios/file_1.json"),
                    "--output", str(output), "--plot", str(root / "figure")]
            with patch("bandit_cost_quality.cli.compare") as comparison, \
                    redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                main(args)
            self.assertEqual(raised.exception.code, 2)
            comparison.assert_not_called()
            self.assertEqual(existing.read_bytes(), b"existing plot")
            self.assertFalse(output.exists())

    def test_normalized_json_plot_alias_rejected_before_computation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = ["compare", "--scenario", str(ROOT / "scenarios/file_1.json"),
                    "--output", str(root / "figure.png"), "--plot", str(root / "figure")]
            with patch("bandit_cost_quality.cli.compare") as comparison, \
                    redirect_stderr(io.StringIO()) as errors, self.assertRaises(SystemExit) as raised:
                main(args)
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("different paths", errors.getvalue())
            comparison.assert_not_called()
            self.assertEqual(list(root.iterdir()), [])

    def test_direct_plot_preserves_existing_target_with_or_without_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / "figure.png"
            existing.write_bytes(b"existing plot")
            for target in (existing, root / "figure"):
                with self.subTest(target=target), self.assertRaises(FileExistsError):
                    plot_comparison({}, target)
                self.assertEqual(existing.read_bytes(), b"existing plot")

    def test_output_symlinks_are_preserved_before_computation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / "missing.png"
            link = root / "figure.png"
            link.symlink_to(missing)
            with self.assertRaises(FileExistsError):
                plot_comparison({}, root / "figure")
            self.assertTrue(link.is_symlink())
            self.assertFalse(missing.exists())
            for outputs in (("--output", str(link)),
                            ("--output", str(root / "result.json"), "--plot", str(root / "figure"))):
                with self.subTest(outputs=outputs), patch("bandit_cost_quality.cli.compare") as comparison, \
                        redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                    main(["compare", "--scenario", str(ROOT / "scenarios/file_1.json"), *outputs])
                self.assertEqual(raised.exception.code, 2)
                comparison.assert_not_called()
                self.assertTrue(link.is_symlink())
                self.assertFalse(missing.exists())
            self.assertFalse((root / "result.json").exists())

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "optional plotting dependency not installed")
    def test_extensionless_plot_success_uses_explicit_png_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "result.json"
            args = ["compare", "--scenario", str(ROOT / "scenarios/file_1.json"),
                    "--horizon", "2", "--seeds", "0", "--output", str(output),
                    "--plot", str(root / "figure")]
            with redirect_stdout(io.StringIO()):
                main(args)
            self.assertTrue((root / "figure.png").read_bytes().startswith(b"\x89PNG"))
            self.assertFalse((root / "figure").exists())
            self.assertEqual(json.loads(output.read_text())["horizon"], 2)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "optional plotting dependency not installed")
    def test_direct_plot_preserves_explicit_formats_and_extension_case(self):
        result = compare(self.scenario, 2, [0], default_policies())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for suffix, header in ((".PNG", b"\x89PNG"), (".svg", b"<?xml"), (".pdf", b"%PDF")):
                output = root / ("figure" + suffix)
                with self.subTest(suffix=suffix):
                    self.assertEqual(plot_comparison(result, output), output)
                    self.assertTrue(output.read_bytes().startswith(header))

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "optional plotting dependency not installed")
    def test_unsupported_plot_format_creates_no_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unsupported = root / "figure.unsupported"
            with self.assertRaisesRegex(ValueError, "unsupported plot format"):
                plot_comparison({}, unsupported)
            args = ["compare", "--scenario", str(ROOT / "scenarios/file_1.json"),
                    "--output", str(root / "result.json"), "--plot", str(unsupported)]
            with patch("bandit_cost_quality.cli.compare") as comparison, \
                    redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args)
            comparison.assert_not_called()
            self.assertEqual(list(root.iterdir()), [])

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "optional plotting dependency not installed")
    def test_plot_renderer_failure_leaves_no_empty_file(self):
        result = compare(self.scenario, 2, [0], default_policies())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "figure.png"
            with patch("matplotlib.figure.Figure.savefig", side_effect=ValueError("renderer failed")), \
                    self.assertRaisesRegex(ValueError, "renderer failed"):
                plot_comparison(result, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
