"""Safe JSON command-line interface; historical scripts are never imported."""

import argparse
import json
from pathlib import Path
import sys

from .experiments import (canonical_hash, compare, normalize_plot_path, plot_comparison,
                          validate_plot_path)
from .simulator import FEATURE_NAMES, PolicySpec, Scenario, default_policies


def load_json(path):
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"nonfinite JSON value: {value}")

    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_keys,
                      parse_constant=reject_constant)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fixed-threshold Gaussian bandit research demo")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("compare", "tune"):
        command = commands.add_parser(name)
        command.add_argument("--scenario", required=True, type=Path)
        command.add_argument("--horizon", type=int, default=2000)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--plot", type=Path)
        if name == "compare":
            command.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
            command.add_argument("--tuned", type=Path, help="versioned tuning JSON for this scenario and horizon")
        else:
            command.add_argument("--train-seeds", type=int, nargs="+", default=list(range(5)))
            command.add_argument("--evaluation-seeds", type=int, nargs="+", default=list(range(100, 120)))
            command.add_argument("--trials", type=int, default=10)
            command.add_argument("--optimizer-seed", type=int, default=42)
            command.add_argument("--quality-weight", type=float, default=1.0)
            command.add_argument("--cost-weight", type=float, default=1.0)
    args = parser.parse_args(argv)
    try:
        if args.plot:
            args.plot = normalize_plot_path(args.plot)
        if args.output.exists() or args.output.is_symlink():
            raise ValueError("output already exists; choose a new path")
        if args.plot:
            output_path, plot_path = args.output.resolve(), args.plot.resolve()
            if output_path == plot_path:
                raise ValueError("JSON and plot outputs must have different paths")
            if output_path in plot_path.parents or plot_path in output_path.parents:
                raise ValueError("JSON and plot outputs cannot be ancestors of one another")
            validate_plot_path(args.plot)
        scenario = Scenario.from_dict(load_json(args.scenario))
        if args.command == "compare":
            policies = default_policies()
            tuning_reference = None
            if args.tuned:
                tuned = load_json(args.tuned)
                if (not isinstance(tuned, dict) or tuned.get("schema_version") != 1 or
                    tuned.get("kind") != "tuning" or tuned.get("feature_version") != 1 or
                    tuned.get("feature_names") != list(FEATURE_NAMES) or
                    tuned.get("scenario_sha256") != canonical_hash(scenario.as_dict()) or
                    tuned.get("horizon") != args.horizon):
                    raise ValueError("tuning artifact does not match this scenario, horizon, and feature version")
                if set(args.seeds) & set(tuned["train_seeds"]):
                    raise ValueError("comparison seeds overlap the artifact's training seeds")
                policies[-1] = PolicySpec(**tuned["selected_policy"])
                tuning_reference = canonical_hash(tuned)
            result = compare(scenario, args.horizon, args.seeds, policies)
            result["tuning_artifact_sha256"] = tuning_reference
            evaluation = result
        else:
            from .tuning import tune
            result = tune(scenario, args.horizon, args.train_seeds, args.evaluation_seeds,
                          args.trials, args.optimizer_seed, args.quality_weight, args.cost_weight)
            evaluation = result["evaluation"]
        result["invocation"] = list(argv if argv is not None else sys.argv[1:])
        payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.plot:
            args.plot.parent.mkdir(parents=True, exist_ok=True)
            plot_comparison(evaluation, args.plot)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as output:
            output.write(payload)
        print("policy                   quality pseudo-regret    cost pseudo-regret")
        for name, values in evaluation["aggregates"].items():
            print(f"{name:24} {values['quality_pseudo_regret']['mean']:21.4f} {values['cost_pseudo_regret']['mean']:21.4f}")
        print(f"Saved {args.output}")
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
