#!/usr/bin/env python3
"""
StateMem Minimal Experiment
===========================

Tests whether belief reconciliation beats simple baselines for
single-agent temporal state estimation under noisy observations.

Baselines:
1) Latest observation
2) Recency-weighted evidence
3) Recency x confidence
4) StateMem Bayesian belief reconciler

No third-party libraries required.

Run:
    python3 statemem_minimal_experiment.py
    python3 statemem_minimal_experiment.py --episodes 200 --steps 250
    python3 statemem_minimal_experiment.py --show-example
    python3 statemem_minimal_experiment.py --input my_lmee_states.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

DEFAULT_STATES = ["counter", "sink", "table"]


@dataclass
class Observation:
    episode_id: str
    t: int
    true_state: str
    observed_state: str
    confidence: float


def generate_episode(
    episode_id: str,
    steps: int,
    corruption_rate: float,
    states: List[str],
    switch_prob: float,
    rng: random.Random,
) -> List[Observation]:
    true_state = rng.choice(states)
    out = []

    for t in range(steps):
        if t > 0 and rng.random() < switch_prob:
            true_state = rng.choice([s for s in states if s != true_state])

        is_wrong = rng.random() < corruption_rate

        if is_wrong:
            observed_state = rng.choice([s for s in states if s != true_state])
            confidence = rng.uniform(0.20, 0.65)
        else:
            observed_state = true_state
            confidence = rng.uniform(0.65, 0.98)

        out.append(
            Observation(
                episode_id=episode_id,
                t=t,
                true_state=true_state,
                observed_state=observed_state,
                confidence=confidence,
            )
        )

    return out


def generate_dataset(
    episodes: int,
    steps: int,
    corruption_rate: float,
    states: List[str],
    switch_prob: float,
    seed: int,
) -> Dict[str, List[Observation]]:
    rng = random.Random(seed)
    data = {}

    for i in range(episodes):
        eid = f"ep_{i:04d}"
        data[eid] = generate_episode(
            episode_id=eid,
            steps=steps,
            corruption_rate=corruption_rate,
            states=states,
            switch_prob=switch_prob,
            rng=rng,
        )

    return data


def load_jsonl(path: str) -> Dict[str, List[Observation]]:
    """
    Expected one JSON object per line:

    {
      "episode_id": "ep_001",
      "t": 95,
      "true_state": "sink",
      "observed_state": "counter",
      "confidence": 0.22
    }
    """
    grouped = defaultdict(list)

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            required = {
                "episode_id",
                "t",
                "true_state",
                "observed_state",
                "confidence",
            }
            missing = required - row.keys()
            if missing:
                raise ValueError(
                    f"{path}:{line_no}: missing fields {sorted(missing)}"
                )

            obs = Observation(
                episode_id=str(row["episode_id"]),
                t=int(row["t"]),
                true_state=str(row["true_state"]),
                observed_state=str(row["observed_state"]),
                confidence=float(row["confidence"]),
            )
            grouped[obs.episode_id].append(obs)

    for eid in grouped:
        grouped[eid].sort(key=lambda x: x.t)

    return dict(grouped)


def predict_latest(history: List[Observation], states: List[str]) -> str:
    return history[-1].observed_state


def predict_recency(
    history: List[Observation],
    states: List[str],
    decay: float = 0.25,
) -> str:
    now = history[-1].t
    scores = {s: 0.0 for s in states}

    for obs in history:
        age = now - obs.t
        scores[obs.observed_state] += math.exp(-decay * age)

    return max(scores, key=scores.get)


def predict_recency_confidence(
    history: List[Observation],
    states: List[str],
    decay: float = 0.25,
) -> str:
    now = history[-1].t
    scores = {s: 0.0 for s in states}

    for obs in history:
        age = now - obs.t
        weight = obs.confidence * math.exp(-decay * age)
        scores[obs.observed_state] += weight

    return max(scores, key=scores.get)


class StateMemBelief:
    """Minimal probabilistic temporal belief reconciler."""

    def __init__(self, states: List[str], stay_probability: float = 0.90):
        if len(states) < 2:
            raise ValueError("Need at least two possible states.")

        self.states = list(states)
        self.stay_probability = stay_probability
        self.belief = {s: 1.0 / len(states) for s in states}

    def _predict(self) -> Dict[str, float]:
        n = len(self.states)
        change_probability = (1.0 - self.stay_probability) / (n - 1)
        predicted = {s: 0.0 for s in self.states}

        for prev_state, prev_prob in self.belief.items():
            for next_state in self.states:
                trans = (
                    self.stay_probability
                    if next_state == prev_state
                    else change_probability
                )
                predicted[next_state] += prev_prob * trans

        return predicted

    def update(self, observed_state: str, confidence: float) -> Dict[str, float]:
        predicted = self._predict()
        n = len(self.states)
        confidence = max(0.01, min(0.99, confidence))

        likelihood = {}
        for state in self.states:
            if state == observed_state:
                likelihood[state] = confidence
            else:
                likelihood[state] = (1.0 - confidence) / (n - 1)

        posterior = {
            state: predicted[state] * likelihood[state]
            for state in self.states
        }

        z = sum(posterior.values())
        if z <= 0:
            posterior = {s: 1.0 / n for s in self.states}
        else:
            posterior = {s: p / z for s, p in posterior.items()}

        self.belief = posterior
        return dict(self.belief)

    def predict(self) -> str:
        return max(self.belief, key=self.belief.get)

    def uncertainty_entropy(self) -> float:
        n = len(self.states)
        h = 0.0
        for p in self.belief.values():
            if p > 0:
                h -= p * math.log(p)
        return h / math.log(n)


def compress_same_state_runs(history: List[Observation]) -> List[dict]:
    """Compress consecutive identical observed states into intervals."""
    if not history:
        return []

    intervals = []
    current = {
        "state": history[0].observed_state,
        "start_t": history[0].t,
        "end_t": history[0].t,
        "support_count": 1,
        "confidence_sum": history[0].confidence,
        "min_confidence": history[0].confidence,
        "max_confidence": history[0].confidence,
    }

    for obs in history[1:]:
        if obs.observed_state == current["state"]:
            current["end_t"] = obs.t
            current["support_count"] += 1
            current["confidence_sum"] += obs.confidence
            current["min_confidence"] = min(
                current["min_confidence"], obs.confidence
            )
            current["max_confidence"] = max(
                current["max_confidence"], obs.confidence
            )
        else:
            current["mean_confidence"] = (
                current["confidence_sum"] / current["support_count"]
            )
            intervals.append(current)
            current = {
                "state": obs.observed_state,
                "start_t": obs.t,
                "end_t": obs.t,
                "support_count": 1,
                "confidence_sum": obs.confidence,
                "min_confidence": obs.confidence,
                "max_confidence": obs.confidence,
            }

    current["mean_confidence"] = (
        current["confidence_sum"] / current["support_count"]
    )
    intervals.append(current)

    for item in intervals:
        item.pop("confidence_sum", None)

    return intervals


def evaluate_dataset(
    data: Dict[str, List[Observation]],
    states: List[str],
    decay: float,
    stay_probability: float,
    warmup: int = 1,
) -> dict:
    correct = {
        "latest": 0,
        "recency": 0,
        "recency_conf": 0,
        "statemem": 0,
    }
    total = 0
    brier_sum = 0.0
    entropy_sum = 0.0
    raw_evidence_count = 0
    compressed_interval_count = 0

    for episode in data.values():
        model = StateMemBelief(states, stay_probability)
        history = []

        for obs in episode:
            history.append(obs)
            belief = model.update(obs.observed_state, obs.confidence)

            if len(history) < warmup:
                continue

            truth = obs.true_state

            predictions = {
                "latest": predict_latest(history, states),
                "recency": predict_recency(history, states, decay),
                "recency_conf": predict_recency_confidence(
                    history, states, decay
                ),
                "statemem": model.predict(),
            }

            for name, pred in predictions.items():
                correct[name] += int(pred == truth)

            for s in states:
                target = 1.0 if s == truth else 0.0
                brier_sum += (belief[s] - target) ** 2

            entropy_sum += model.uncertainty_entropy()
            total += 1

        raw_evidence_count += len(episode)
        compressed_interval_count += len(compress_same_state_runs(episode))

    return {
        "n": total,
        "latest": correct["latest"] / total,
        "recency": correct["recency"] / total,
        "recency_conf": correct["recency_conf"] / total,
        "statemem": correct["statemem"] / total,
        "brier": brier_sum / total,
        "mean_entropy": entropy_sum / total,
        "compression_ratio": compressed_interval_count / raw_evidence_count,
    }


def print_result_table(results):
    print()
    print("=" * 86)
    print("StateMem minimal experiment")
    print("=" * 86)
    print(
        f"{'Noise':>7} | {'Latest':>8} | {'Recency':>8} | "
        f"{'Rec×Conf':>9} | {'StateMem':>8} | {'Brier':>8} | {'Compress':>9}"
    )
    print("-" * 86)

    for label, r in results:
        print(
            f"{label:>7} | "
            f"{100*r['latest']:7.2f}% | "
            f"{100*r['recency']:7.2f}% | "
            f"{100*r['recency_conf']:8.2f}% | "
            f"{100*r['statemem']:7.2f}% | "
            f"{r['brier']:8.4f} | "
            f"{100*r['compression_ratio']:8.2f}%"
        )

    print("=" * 86)
    print("Lower Brier is better. Lower Compress means more interval compression.")


def show_example_episode(
    episode: List[Observation],
    states: List[str],
    stay_probability: float,
    max_rows: int = 30,
):
    print("\nExample belief trace")
    print("-" * 100)
    print(
        f"{'t':>4}  {'truth':>9}  {'observed':>9}  {'conf':>6}  "
        + "  ".join(f"P({s})" for s in states)
        + "  prediction"
    )
    print("-" * 100)

    model = StateMemBelief(states, stay_probability)

    for obs in episode[:max_rows]:
        belief = model.update(obs.observed_state, obs.confidence)
        probs = "  ".join(f"{belief[s]:.3f}" for s in states)
        print(
            f"{obs.t:4d}  {obs.true_state:>9}  {obs.observed_state:>9}  "
            f"{obs.confidence:6.2f}  {probs}  {model.predict()}"
        )

    print("-" * 100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--switch-prob", type=float, default=0.08)
    parser.add_argument("--decay", type=float, default=0.25)
    parser.add_argument("--stay-prob", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--show-example", action="store_true")
    args = parser.parse_args()

    if args.input:
        data = load_jsonl(args.input)
        states = sorted(
            {x.true_state for ep in data.values() for x in ep}
            | {x.observed_state for ep in data.values() for x in ep}
        )
        result = evaluate_dataset(
            data,
            states,
            args.decay,
            args.stay_prob,
        )
        print_result_table([("input", result)])
        if args.show_example:
            show_example_episode(
                next(iter(data.values())), states, args.stay_prob
            )
        return

    results = []
    for corruption in [0.00, 0.10, 0.20, 0.30, 0.40]:
        data = generate_dataset(
            episodes=args.episodes,
            steps=args.steps,
            corruption_rate=corruption,
            states=DEFAULT_STATES,
            switch_prob=args.switch_prob,
            seed=args.seed + int(corruption * 1000),
        )
        result = evaluate_dataset(
            data,
            DEFAULT_STATES,
            args.decay,
            args.stay_prob,
        )
        results.append((f"{int(corruption*100)}%", result))

    print_result_table(results)

    if args.show_example:
        data = generate_dataset(
            episodes=1,
            steps=min(args.steps, 40),
            corruption_rate=0.30,
            states=DEFAULT_STATES,
            switch_prob=args.switch_prob,
            seed=args.seed,
        )
        show_example_episode(
            next(iter(data.values())), DEFAULT_STATES, args.stay_prob
        )


if __name__ == "__main__":
    main()
