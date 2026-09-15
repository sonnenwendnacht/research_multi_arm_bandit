"""Optional finite-budget Bayesian search; evaluation seeds never enter its loss."""

from .experiments import canonical_hash, compare, provenance, validate_seeds
from .simulator import (FEATURE_NAMES, PolicySpec, default_policies, finite_number,
                        nonnegative_integer, simulate)


def tune(scenario, horizon, train_seeds, evaluation_seeds, trials, optimizer_seed,
         quality_weight=1.0, cost_weight=1.0):
    horizon = nonnegative_integer(horizon, "horizon")
    trials = nonnegative_integer(trials, "trials")
    optimizer_seed = nonnegative_integer(optimizer_seed, "optimizer_seed")
    train_seeds, evaluation_seeds = validate_seeds(train_seeds), validate_seeds(evaluation_seeds)
    if horizon == 0 or trials == 0:
        raise ValueError("tuning requires positive horizon and trials")
    if set(train_seeds) & set(evaluation_seeds):
        raise ValueError("training and evaluation seeds must be disjoint")
    quality_weight = finite_number(quality_weight, "quality_weight")
    cost_weight = finite_number(cost_weight, "cost_weight")
    if quality_weight < 0 or cost_weight < 0 or quality_weight + cost_weight <= 0:
        raise ValueError("objective weights must be nonnegative, with at least one positive")
    try:
        from skopt import gp_minimize
        from skopt.space import Categorical, Real
    except ImportError as error:
        raise ValueError("Bayesian tuning requires installation with the [tuning] extra") from error

    dimensions = [Real(-5.0, 5.0, name=name) for name in FEATURE_NAMES]
    dimensions += [Real(0.0, 1.0, name="exploration_exponent"), Categorical([False, True], name="random_exploration")]
    queries = []

    def objective(parameters):
        policy = PolicySpec("feature_score", float(parameters[-2]), bool(parameters[-1]), tuple(parameters[:-2]))
        outcomes = [simulate(scenario, horizon, seed, policy).summary() for seed in train_seeds]
        quality = sum(run["quality_pseudo_regret"] for run in outcomes) / len(outcomes)
        cost = sum(run["cost_pseudo_regret"] for run in outcomes) / len(outcomes)
        loss = (quality_weight * quality + cost_weight * cost) / horizon
        queries.append({"policy": policy.as_dict(), "training_quality_mean": quality,
                        "training_cost_mean": cost, "objective": loss})
        return loss

    gp_minimize(objective, dimensions, n_calls=trials, n_initial_points=min(5, trials),
                random_state=optimizer_seed, acq_func="EI", n_points=1000)
    best = min(queries, key=lambda query: query["objective"])
    policy = PolicySpec(**best["policy"])
    evaluation = compare(scenario, horizon, evaluation_seeds, default_policies()[:2] + [policy])
    return {"schema_version": 1, "kind": "tuning", "feature_version": 1,
            "feature_names": list(FEATURE_NAMES), "provenance": provenance(),
            "scenario": scenario.as_dict(), "scenario_sha256": canonical_hash(scenario.as_dict()),
            "horizon": horizon, "train_seeds": train_seeds, "evaluation_seeds": evaluation_seeds,
            "optimizer_seed": optimizer_seed, "trials": trials,
            "search_space": {"feature_weights": [-5.0, 5.0], "exploration_exponent": [0.0, 1.0],
                             "random_exploration": [False, True]},
            "objective": {"formula": "(quality_weight * quality + cost_weight * cost) / horizon",
                          "quality_weight": quality_weight, "cost_weight": cost_weight},
            "queries": queries, "selected_policy": policy.as_dict(), "evaluation": evaluation}
