"""Small Gaussian simulator derived from the archived 2024 experiments."""

from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np


FEATURE_NAMES = (
    "sample_mean", "cost", "relative_shortfall", "relative_overflow",
    "pull_count", "confidence_radius", "empirically_feasible", "ucb_feasible",
)
POLICIES = ("explore_then_commit", "threshold_ucb", "feature_score")


def nonnegative_integer(value, label):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return int(value)


def finite_number(value, label):
    try:
        valid = not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{label} must be a finite number")
    return float(value)


@dataclass(frozen=True)
class Scenario:
    name: str
    threshold: float
    costs: tuple
    means: tuple
    stds: tuple

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("scenario name must be a nonempty string")
        threshold = finite_number(self.threshold, "threshold")
        if threshold <= 0:
            raise ValueError("threshold must be positive for relative score features")
        if not self.costs or not (len(self.costs) == len(self.means) == len(self.stds)):
            raise ValueError("costs, means, and stds must have equal nonzero lengths")
        for label in ("costs", "means", "stds"):
            values = tuple(finite_number(v, label) for v in getattr(self, label))
            if label != "means" and any(v < 0 for v in values):
                raise ValueError(f"{label} must be nonnegative")
            object.__setattr__(self, label, values)
        object.__setattr__(self, "threshold", threshold)
        if not any(mean >= threshold for mean in self.means):
            raise ValueError("at least one truly feasible arm is required to define cost regret")

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or set(data) != {"schema_version", "name", "threshold", "arms"}:
            raise ValueError("scenario requires exactly schema_version, name, threshold, and arms")
        if type(data["schema_version"]) is not int or data["schema_version"] != 1:
            raise ValueError("unsupported scenario schema_version")
        arms = data["arms"]
        if not isinstance(arms, list) or not arms:
            raise ValueError("arms must be a nonempty list")
        for arm in arms:
            if not isinstance(arm, dict) or set(arm) != {"mean", "cost", "std"}:
                raise ValueError("each Gaussian arm requires exactly mean, cost, and std")
        return cls(data["name"], data["threshold"], tuple(a["cost"] for a in arms),
                   tuple(a["mean"] for a in arms), tuple(a["std"] for a in arms))

    def as_dict(self):
        return {"schema_version": 1, "name": self.name, "threshold": self.threshold,
                "arms": [{"mean": m, "cost": c, "std": s}
                         for m, c, s in zip(self.means, self.costs, self.stds)]}

    @property
    def num_arms(self):
        return len(self.costs)

    @property
    def optimal_arm(self):
        feasible = [i for i, mean in enumerate(self.means) if mean >= self.threshold]
        return min(feasible, key=lambda i: (self.costs[i], i))


@dataclass(frozen=True)
class PolicySpec:
    name: str
    exploration_exponent: float = 0.0
    random_exploration: bool = False
    weights: tuple | None = None

    def __post_init__(self):
        if self.name not in POLICIES:
            raise ValueError(f"unknown policy: {self.name}")
        alpha = finite_number(self.exploration_exponent, "exploration_exponent")
        if not 0 <= alpha <= 1:
            raise ValueError("exploration_exponent must be between 0 and 1")
        if type(self.random_exploration) is not bool:
            raise ValueError("random_exploration must be boolean")
        if self.name == "threshold_ucb" and (alpha != 0 or self.random_exploration):
            raise ValueError("threshold_ucb does not use an extra exploration phase")
        if self.weights is not None:
            if self.name != "feature_score" or len(self.weights) != len(FEATURE_NAMES):
                raise ValueError("feature_score requires exactly eight weights")
            object.__setattr__(self, "weights", tuple(finite_number(w, "weight") for w in self.weights))

    def as_dict(self):
        return {"name": self.name, "exploration_exponent": self.exploration_exponent,
                "random_exploration": self.random_exploration,
                "weights": list(self.weights) if self.weights is not None else None}


def default_policies():
    return [PolicySpec("explore_then_commit"), PolicySpec("threshold_ucb"),
            PolicySpec("feature_score", exploration_exponent=0.5)]


def reward_streams(scenario, horizon, seed):
    """Arm i's nth pull uses the same draw under every policy in a comparison."""
    horizon = nonnegative_integer(horizon, "horizon")
    seed = nonnegative_integer(seed, "seed")
    return np.array([
        np.random.default_rng(np.random.SeedSequence([seed, 0, arm])).normal(mean, std, horizon)
        for arm, (mean, std) in enumerate(zip(scenario.means, scenario.stds))
    ])


def cumulative_pseudo_regret(scenario, arms):
    """Realized action-path sums of expected per-arm gaps (not reward noise)."""
    arms = np.asarray(arms)
    if arms.ndim != 1 or (arms.size and (
        not np.issubdtype(arms.dtype, np.integer) or np.any(arms < 0) or np.any(arms >= scenario.num_arms)
    )):
        raise ValueError("arms must be a one-dimensional sequence of valid arm indices")
    arms = arms.astype(int)
    try:
        with np.errstate(over="raise", invalid="raise"):
            quality = np.maximum(0.0, scenario.threshold - np.asarray(scenario.means)[arms])
            cost = np.maximum(0.0, np.asarray(scenario.costs)[arms] - scenario.costs[scenario.optimal_arm])
            return np.cumsum(quality), np.cumsum(cost)
    except FloatingPointError as error:
        raise ValueError("pseudo-regret overflowed; reduce scenario scale or horizon") from error


@dataclass
class Run:
    arms: np.ndarray
    rewards: np.ndarray
    counts: np.ndarray
    estimated_means: np.ndarray
    quality_curve: np.ndarray
    cost_curve: np.ndarray

    def summary(self):
        return {"pulls": int(self.arms.size), "arm_counts": self.counts.tolist(),
                "estimated_means": self.estimated_means.tolist(),
                "quality_pseudo_regret": float(self.quality_curve[-1]) if self.arms.size else 0.0,
                "cost_pseudo_regret": float(self.cost_curve[-1]) if self.arms.size else 0.0}


def simulate(scenario, horizon, seed, policy, streams=None):
    horizon = nonnegative_integer(horizon, "horizon")
    seed = nonnegative_integer(seed, "seed")
    k = scenario.num_arms
    if streams is None:
        streams = reward_streams(scenario, horizon, seed)
    streams = np.asarray(streams, dtype=float)
    if streams.shape != (k, horizon) or not np.isfinite(streams).all():
        raise ValueError("reward streams must be a finite (num_arms, horizon) array")
    policy_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    counts = np.zeros(k, dtype=int)
    estimates = np.zeros(k)
    arms = np.empty(horizon, dtype=int)
    rewards = np.empty(horizon)
    costs = np.asarray(scenario.costs)
    extra_target = math.ceil(horizon ** policy.exploration_exponent) - 1 if horizon else 0
    exploration = min(horizon, max(k, extra_target))
    committed_arm = None
    weights = np.asarray(policy.weights if policy.weights is not None else
                         [0, -1, 0, 0, 0, 0, 0, float(max(costs) + 1)])

    def cheapest_feasible_or_max(values):
        feasible = np.flatnonzero(values >= scenario.threshold)
        if feasible.size:
            return int(min(feasible, key=lambda arm: (costs[arm], arm)))
        return int(np.argmax(values))

    for t in range(horizon):
        if t < k:
            arm = t
        elif policy.name != "threshold_ucb" and t < exploration:
            arm = int(policy_rng.integers(k)) if policy.random_exploration else (t - k) % k
        elif policy.name == "explore_then_commit":
            if committed_arm is None:
                committed_arm = cheapest_feasible_or_max(estimates)
            arm = committed_arm
        elif policy.name == "threshold_ucb":
            radius = np.sqrt(2 * math.log(max(horizon, 2)) / counts)
            arm = cheapest_feasible_or_max(estimates + radius)
        else:
            radius = np.sqrt(4 * math.log(t + 1) / counts)
            features = np.column_stack((
                estimates, costs, np.maximum(0, scenario.threshold - estimates) / scenario.threshold,
                np.maximum(0, estimates - scenario.threshold) / scenario.threshold,
                counts, radius, estimates >= scenario.threshold,
                estimates + radius >= scenario.threshold,
            ))
            scores = features @ weights
            if not np.isfinite(scores).all():
                raise ValueError("feature scores overflowed; reduce scenario scale or weights")
            arm = int(np.argmax(scores))
        reward = streams[arm, counts[arm]]
        counts[arm] += 1
        # Subtraction is safe for matching signs; split opposite extremes to avoid overflow.
        old_mean = estimates[arm]
        if np.signbit(old_mean) == np.signbit(reward):
            estimates[arm] = old_mean + (reward - old_mean) / counts[arm]
        else:
            estimates[arm] = old_mean * (1.0 - 1.0 / counts[arm]) + reward / counts[arm]
        if not np.isfinite(estimates[arm]):
            raise ValueError("empirical mean overflowed; reduce reward scale")
        arms[t], rewards[t] = arm, reward
    quality, cost = cumulative_pseudo_regret(scenario, arms)
    return Run(arms, rewards, counts, estimates, quality, cost)
