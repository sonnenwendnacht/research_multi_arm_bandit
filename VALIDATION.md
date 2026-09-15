# Validation and reproducibility

Local validation used Python 3.12.3 on Linux with the package versions in [requirements-repro.txt](requirements-repro.txt). CI is configured for Python 3.11 and 3.12; a configured workflow is not itself evidence that those remote runs have passed.

## Tests

```bash
python -m unittest discover -s tests -v
```

Result: 21 test methods passed, including the optional Bayesian-search test with `scikit-optimize` installed. The tests cover:

- Exact horizons 0, 1, `K-1`, `K`, `K+1`, and 100; exploration exponents 0, 0.5, and 1; both exploration modes.
- Agreement between action history, reward history, counts, empirical averages, and metric curves.
- The historical `T=100,K=5,alpha=0.5` regression: now 100 pulls, including all nine exploration pulls in history.
- Hand-computed pseudo-regret, inclusive threshold equality, cheapest-feasible selection, index tie breaking, and empirical-infeasibility fallbacks.
- Repeatability, global NumPy RNG isolation, shared per-arm pull-index rewards, and action invariance under different hidden means with identical supplied observations.
- Invalid scenarios, unsupported configurations, duplicate JSON keys, executable text masquerading as JSON, nonfinite values, and numeric overflow handling.
- Correct aggregate and paired statistics, undefined single-seed standard errors, a noninteractive CLI round trip, existing-output preservation, and no partial artifacts when JSON serialization fails.
- Disjoint tuning/evaluation seeds and six actual optimizer trials, including an iteration after the initial random evaluations.

When the optional tuning extra is absent, its execution test is explicitly skipped. Historical scripts and pickle contents are not executed by this suite.

An independent review also exercised 162 simulator cases and separate probes
for policy-order independence, hidden-mean independence, and numerical extremes.

## Recorded experiments

Install the recorded dependencies and package, then use new output paths. The
checked-in `examples/` files already exist even in a fresh checkout, and the
CLI refuses to overwrite them:

```bash
python -m pip install -r requirements-repro.txt
python -m pip install -e . --no-deps
python -m bandit_cost_quality compare --scenario scenarios/file_1.json --horizon 2000 --output results/reproduction/file_1_comparison.json --plot results/reproduction/file_1_comparison.png
python -m bandit_cost_quality compare --scenario scenarios/file_2.json --horizon 2000 --output results/reproduction/file_2_comparison.json --plot results/reproduction/file_2_comparison.png
python -m bandit_cost_quality compare --scenario scenarios/file_3.json --horizon 2000 --output results/reproduction/file_3_comparison.json
python -m bandit_cost_quality tune --scenario scenarios/file_2.json --horizon 500 --trials 8 --train-seeds 0 1 2 --evaluation-seeds 100 101 102 103 104 --output results/reproduction/file_2_tuning_smoke.json
```

The three comparisons each use the default seeds 0–19 and three policies: 60 runs per scenario, 180 runs total, and exactly 360,000 recorded pulls. Final means are:

| Scenario | Policy | Quality pseudo-regret | Cost pseudo-regret |
| --- | --- | ---: | ---: |
| 1 | explore_then_commit | 50.0351 | 452.3750 |
| 1 | threshold_ucb | 0.1501 | 3.5000 |
| 1 | feature_score | 1.3508 | 31.0000 |
| 2 | explore_then_commit | 79.9601 | 202.5000 |
| 2 | threshold_ucb | 110.9001 | 3.0000 |
| 2 | feature_score | 174.5808 | 26.5000 |
| 3 | explore_then_commit | 628.9000 | 10275.0000 |
| 3 | threshold_ucb | 659.8500 | 300.0000 |
| 3 | feature_score | 5115.1300 | 2650.0000 |

The eight-trial tuning smoke experiment uses 3 training seeds and 5 held-out evaluation seeds. Its selected feature policy has evaluation means 19.9399 quality pseudo-regret and 250.5000 cost pseudo-regret at horizon 500. Corresponding threshold-UCB means are 46.5301 and 3.0000. This illustrates a tradeoff; it does not establish superiority. The untuned comparison and tuning experiments use different horizons and must not be compared as if they were the same benchmark.

A separate full rerun in the recorded environment reproduced every comparison
result, source hash, and both PNG files exactly. Repeating all eight tuning
trials reproduced every training query, the selected policy, and the held-out
results. Changing output paths changes the recorded invocation, not those
experiment results.

## Source and configuration provenance

Each JSON artifact embeds its scenario, canonical scenario SHA-256, complete policy parameters, seed sets, runtime versions, and SHA-256 values for every maintained package source file plus `pyproject.toml`. Source hashes identify the actual implementation even when the working tree has not yet been committed. The recorded invocation explains the command that produced the artifact. Plot curves come from the same result object as the JSON and show at most 100 cumulative checkpoints, including the final pull.

The historical sources have a separate byte-for-byte manifest:

```bash
sha256sum -c historical/SHA256SUMS
```

Original public sources and recovered laptop changes are distinguished in [historical/README.md](historical/README.md). The maintained JSON `file_3` intentionally uses the recovered laptop configuration, not the earlier public one. No historical tuning result has been reused.

## Limits

These are small synthetic experiments with known, positive thresholds, nonnegative costs, and at least one truly feasible arm. Per-arm reward streams require `O(KT)` memory. Policy confidence formulas retain a unit scale; changing the noise scale does not calibrate them automatically. Standard errors summarize the sampled seeds and are not theoretical confidence guarantees. Bayesian search is a small, discontinuous black-box optimization over the supplied scenarios and may find poor policies. Neither asymptotic regret rates nor generalization to other bandit settings has been established here.
