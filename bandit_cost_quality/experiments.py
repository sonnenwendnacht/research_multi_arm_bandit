"""Paired experiments, transparent aggregation, and artifact provenance."""

import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform

import numpy as np

from . import __version__
from .simulator import nonnegative_integer, reward_streams, simulate


def canonical_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def source_hashes():
    package = Path(__file__).resolve().parent
    paths = sorted(package.glob("*.py"))
    if (package.parent / "pyproject.toml").is_file():
        paths.append(package.parent / "pyproject.toml")
    return {str(path.relative_to(package.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def provenance():
    versions = {"python": platform.python_version(), "numpy": np.__version__}
    for name in ("scikit-optimize", "scikit-learn", "scipy"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return {"package_version": __version__, "versions": versions,
            "platform": platform.system(), "source_sha256": source_hashes(),
            "rng_scheme": "NumPy default_rng; reward SeedSequence([seed,0,arm]); policy [seed,1]",
            "pairing": "same nth reward draw for each arm across policies at each seed"}


def validate_seeds(seeds):
    seeds = [nonnegative_integer(seed, "seed") for seed in seeds]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("provide at least one seed, with no duplicates")
    return seeds


def statistics(values):
    values = np.asarray(values, dtype=float)
    n = values.size
    mean, sd = mean_and_std(values)
    return {"mean": float(mean), "sample_std": float(sd) if sd is not None else None,
            "standard_error": sd / np.sqrt(n) if sd is not None else None, "n": n}


def mean_and_std(values):
    """Reject nonrepresentable summaries instead of emitting invalid JSON numbers."""
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("aggregation requires nonempty finite observations")
    try:
        with np.errstate(over="raise", invalid="raise"):
            mean = np.mean(values, axis=0)
            sd = np.std(values, axis=0, ddof=1) if len(values) > 1 else None
    except FloatingPointError as error:
        raise ValueError("aggregate overflowed; reduce scenario scale or horizon") from error
    return mean, sd


def compare(scenario, horizon, seeds, policies):
    horizon = nonnegative_integer(horizon, "horizon")
    seeds = validate_seeds(seeds)
    policies = list(policies)
    if not policies or len({p.name for p in policies}) != len(policies):
        raise ValueError("comparison requires uniquely named policies")
    checkpoints = np.unique(np.linspace(1, horizon, min(100, horizon), dtype=int)) if horizon else np.array([], dtype=int)
    runs = {policy.name: [] for policy in policies}
    curves = {policy.name: {"quality": [], "cost": []} for policy in policies}
    for seed in seeds:
        streams = reward_streams(scenario, horizon, seed)
        for policy in policies:
            run = simulate(scenario, horizon, seed, policy, streams=streams)
            runs[policy.name].append({"seed": seed, **run.summary()})
            curves[policy.name]["quality"].append(run.quality_curve[checkpoints - 1])
            curves[policy.name]["cost"].append(run.cost_curve[checkpoints - 1])
    aggregates = {}
    metrics = ("quality_pseudo_regret", "cost_pseudo_regret")
    for policy in policies:
        aggregates[policy.name] = {
            metric: statistics([run[metric] for run in runs[policy.name]]) for metric in metrics
        }
        aggregates[policy.name]["curves"] = {"pulls": checkpoints.tolist()}
        for metric in ("quality", "cost"):
            data = np.asarray(curves[policy.name][metric])
            mean, sd = mean_and_std(data)
            aggregates[policy.name]["curves"][metric] = {
                "mean": mean.tolist(),
                "standard_error": (sd / np.sqrt(len(seeds))).tolist() if sd is not None else None,
            }
    baseline = policies[0].name
    paired = {}
    for policy in policies[1:]:
        paired[policy.name] = {
            metric: statistics([run[metric] - base[metric]
                                for run, base in zip(runs[policy.name], runs[baseline])])
            for metric in metrics
        }
    return {"schema_version": 1, "kind": "comparison", "provenance": provenance(),
            "scenario": scenario.as_dict(), "scenario_sha256": canonical_hash(scenario.as_dict()),
            "horizon": horizon, "seeds": seeds, "policies": [p.as_dict() for p in policies],
            "metrics": {
                "quality_pseudo_regret": "sum_t max(0, threshold - true_mean[action_t])",
                "cost_pseudo_regret": "sum_t max(0, cost[action_t] - cheapest_truly_feasible_cost)",
                "uncertainty": "sample standard deviation and standard error across independent seeds; not confidence bounds",
            }, "runs": runs, "aggregates": aggregates,
            "paired_differences": {"reference": baseline, "definition": "policy minus reference", "policies": paired}}


def plot_comparison(result, output):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ValueError("plotting requires installation with the [plot] extra") from error
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, aggregate in result["aggregates"].items():
        curves = aggregate["curves"]
        for axis, metric in zip(axes, ("quality", "cost")):
            mean = np.asarray(curves[metric]["mean"])
            axis.plot(curves["pulls"], mean, label=name)
            if curves[metric]["standard_error"] is not None:
                se = np.asarray(curves[metric]["standard_error"])
                axis.fill_between(curves["pulls"], mean - se, mean + se, alpha=0.15)
            axis.set(xlabel="Pulls", ylabel=f"Cumulative {metric} pseudo-regret")
            axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{result['scenario']['name']}: {len(result['seeds'])} seeds; mean ± one standard error")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
