#!/usr/bin/env python3
"""
StateMem Stress Test
====================

Validates whether StateMem belief reconciliation beats simple baselines across:
  1) observation corruption
  2) true world transition frequency
  3) confidence reliability
  4) different StateMem stay probabilities

Protocol:
- Tune stay_probability ONLY on validation data.
- Freeze it.
- Evaluate on an independent test split.

No third-party packages required.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

STATES = ["counter", "sink", "table"]


@dataclass
class Observation:
    t: int
    true_state: str
    observed_state: str
    confidence: float
    corrupted: bool


def sample_confidence(correct: bool, mode: str, rng: random.Random) -> float:
    if mode == "reliable":
        return rng.uniform(0.70, 0.98) if correct else rng.uniform(0.10, 0.55)
    if mode == "overlapping":
        return rng.uniform(0.55, 0.95) if correct else rng.uniform(0.35, 0.85)
    if mode == "misleading":
        return rng.uniform(0.45, 0.85) if correct else rng.uniform(0.55, 0.95)
    raise ValueError(f"Unknown confidence mode: {mode}")


def generate_episode(
    steps: int,
    corruption_rate: float,
    switch_prob: float,
    confidence_mode: str,
    rng: random.Random,
) -> List[Observation]:
    true_state = rng.choice(STATES)
    episode: List[Observation] = []

    for t in range(steps):
        if t > 0 and rng.random() < switch_prob:
            true_state = rng.choice([s for s in STATES if s != true_state])

        corrupted = rng.random() < corruption_rate
        observed_state = (
            rng.choice([s for s in STATES if s != true_state])
            if corrupted else true_state
        )
        confidence = sample_confidence(not corrupted, confidence_mode, rng)

        episode.append(
            Observation(
                t=t,
                true_state=true_state,
                observed_state=observed_state,
                confidence=confidence,
                corrupted=corrupted,
            )
        )

    return episode


def generate_dataset(
    episodes: int,
    steps: int,
    corruption_rate: float,
    switch_prob: float,
    confidence_mode: str,
    seed: int,
) -> List[List[Observation]]:
    rng = random.Random(seed)
    return [
        generate_episode(
            steps, corruption_rate, switch_prob, confidence_mode, rng
        )
        for _ in range(episodes)
    ]


def latest_prediction(history: List[Observation]) -> str:
    return history[-1].observed_state


def recency_prediction(history: List[Observation], decay: float) -> str:
    now = history[-1].t
    scores = {s: 0.0 for s in STATES}
    for obs in history:
        age = now - obs.t
        scores[obs.observed_state] += math.exp(-decay * age)
    return max(scores, key=scores.get)


def recency_conf_prediction(history: List[Observation], decay: float) -> str:
    now = history[-1].t
    scores = {s: 0.0 for s in STATES}
    for obs in history:
        age = now - obs.t
        scores[obs.observed_state] += obs.confidence * math.exp(-decay * age)
    return max(scores, key=scores.get)


class StateMem:
    def __init__(self, stay_probability: float):
        self.stay_probability = stay_probability
        self.belief = {s: 1.0 / len(STATES) for s in STATES}

    def _predict_prior(self) -> Dict[str, float]:
        n = len(STATES)
        p_change_each = (1.0 - self.stay_probability) / (n - 1)
        prior = {s: 0.0 for s in STATES}

        for prev_state, prev_prob in self.belief.items():
            for next_state in STATES:
                transition = (
                    self.stay_probability
                    if next_state == prev_state
                    else p_change_each
                )
                prior[next_state] += prev_prob * transition

        return prior

    def update(self, observed_state: str, confidence: float) -> Dict[str, float]:
        prior = self._predict_prior()
        c = max(0.01, min(0.99, confidence))
        n = len(STATES)

        likelihood = {}
        for s in STATES:
            likelihood[s] = c if s == observed_state else (1.0 - c) / (n - 1)

        posterior = {s: prior[s] * likelihood[s] for s in STATES}
        z = sum(posterior.values())
        self.belief = (
            {s: posterior[s] / z for s in STATES}
            if z > 0 else {s: 1.0 / n for s in STATES}
        )
        return dict(self.belief)

    def predict(self) -> str:
        return max(self.belief, key=self.belief.get)


def transition_lags(
    truths: List[str], predictions: List[str]
) -> Tuple[float, int, int]:
    lags: List[int] = []
    misses = 0
    n = len(truths)
    transition_indices = [i for i in range(1, n) if truths[i] != truths[i - 1]]

    for pos, start in enumerate(transition_indices):
        new_state = truths[start]
        end = transition_indices[pos + 1] if pos + 1 < len(transition_indices) else n
        found = False

        for j in range(start, end):
            if predictions[j] == new_state:
                lags.append(j - start)
                found = True
                break

        if not found:
            misses += 1

    mean_lag = sum(lags) / len(lags) if lags else float("nan")
    return mean_lag, len(lags), misses


def evaluate_dataset(
    dataset: List[List[Observation]],
    stay_probability: float,
    decay: float,
) -> Dict[str, float]:
    methods = ["latest", "recency", "recency_conf", "statemem"]
    correct = {m: 0 for m in methods}
    corrupted_correct = {m: 0 for m in methods}
    lag_sum = {m: 0.0 for m in methods}
    lag_successes = {m: 0 for m in methods}
    lag_misses = {m: 0 for m in methods}
    total = 0
    corrupted_total = 0
    brier_sum = 0.0

    for episode in dataset:
        history: List[Observation] = []
        model = StateMem(stay_probability)
        preds = {m: [] for m in methods}
        truths: List[str] = []

        for obs in episode:
            history.append(obs)
            belief = model.update(obs.observed_state, obs.confidence)
            pred_map = {
                "latest": latest_prediction(history),
                "recency": recency_prediction(history, decay),
                "recency_conf": recency_conf_prediction(history, decay),
                "statemem": model.predict(),
            }

            truths.append(obs.true_state)
            total += 1

            for m in methods:
                pred = pred_map[m]
                preds[m].append(pred)
                correct[m] += int(pred == obs.true_state)
                if obs.corrupted:
                    corrupted_correct[m] += int(pred == obs.true_state)

            if obs.corrupted:
                corrupted_total += 1

            for s in STATES:
                target = 1.0 if s == obs.true_state else 0.0
                brier_sum += (belief[s] - target) ** 2

        for m in methods:
            mean_lag, successes, misses = transition_lags(truths, preds[m])
            if successes > 0 and not math.isnan(mean_lag):
                lag_sum[m] += mean_lag * successes
                lag_successes[m] += successes
            lag_misses[m] += misses

    out: Dict[str, float] = {"n": total, "brier": brier_sum / total}

    for m in methods:
        out[f"{m}_acc"] = correct[m] / total
        out[f"{m}_noise_acc"] = (
            corrupted_correct[m] / corrupted_total if corrupted_total > 0 else float("nan")
        )
        out[f"{m}_transition_lag"] = (
            lag_sum[m] / lag_successes[m] if lag_successes[m] > 0 else float("nan")
        )
        denom = lag_successes[m] + lag_misses[m]
        out[f"{m}_transition_miss_rate"] = lag_misses[m] / denom if denom > 0 else 0.0

    return out


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def run_grid(
    episodes: int,
    steps: int,
    corruption_rates: List[float],
    switch_probs: List[float],
    confidence_modes: List[str],
    stay_probs: List[float],
    decay: float,
    seed_base: int,
) -> List[dict]:
    rows = []
    condition_idx = 0

    for confidence_mode in confidence_modes:
        for switch_prob in switch_probs:
            for corruption_rate in corruption_rates:
                dataset = generate_dataset(
                    episodes=episodes,
                    steps=steps,
                    corruption_rate=corruption_rate,
                    switch_prob=switch_prob,
                    confidence_mode=confidence_mode,
                    seed=seed_base + condition_idx * 1009,
                )

                for stay_prob in stay_probs:
                    metrics = evaluate_dataset(dataset, stay_prob, decay)
                    rows.append({
                        "confidence_mode": confidence_mode,
                        "switch_prob": switch_prob,
                        "corruption_rate": corruption_rate,
                        "stay_prob": stay_prob,
                        **metrics,
                    })

                condition_idx += 1

    return rows


def save_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def validation_score(rows: List[dict], stay_probability: float) -> Tuple[float, float]:
    relevant = [r for r in rows if r["stay_prob"] == stay_probability]
    mean_acc = sum(r["statemem_acc"] for r in relevant) / len(relevant)
    mean_brier = sum(r["brier"] for r in relevant) / len(relevant)
    return mean_acc, mean_brier


def print_validation_summary(rows: List[dict], stay_probs: List[float]) -> float:
    print("\n" + "=" * 76)
    print("VALIDATION: selecting stay_probability")
    print("=" * 76)
    print(f"{'stay_prob':>10} | {'mean accuracy':>13} | {'mean Brier':>11}")
    print("-" * 76)

    scored = []
    for p in stay_probs:
        acc, brier = validation_score(rows, p)
        scored.append((acc, -brier, p))
        print(f"{p:10.2f} | {100*acc:12.2f}% | {brier:11.4f}")

    scored.sort(reverse=True)
    best_p = scored[0][2]
    print("-" * 76)
    print(f"Frozen stay_probability for TEST = {best_p:.2f}")
    print("=" * 76)
    return best_p


def aggregate_test_rows(rows: List[dict]) -> dict:
    keys = [
        "latest_acc", "recency_acc", "recency_conf_acc", "statemem_acc",
        "latest_noise_acc", "recency_noise_acc", "recency_conf_noise_acc", "statemem_noise_acc",
        "latest_transition_lag", "recency_transition_lag", "recency_conf_transition_lag", "statemem_transition_lag",
        "brier",
    ]
    out = {}
    for key in keys:
        vals = [r[key] for r in rows if not (isinstance(r[key], float) and math.isnan(r[key]))]
        out[key] = sum(vals) / len(vals) if vals else float("nan")
    return out


def print_test_overall(rows: List[dict], chosen_p: float) -> None:
    agg = aggregate_test_rows(rows)
    print("\n" + "=" * 92)
    print(f"TEST RESULTS — frozen stay_probability = {chosen_p:.2f}")
    print("=" * 92)
    print(f"{'Method':<18} | {'Accuracy':>9} | {'Noise Acc':>9} | {'Transition Lag':>14}")
    print("-" * 92)

    for label, key in [
        ("Latest", "latest"),
        ("Recency", "recency"),
        ("Rec×Conf", "recency_conf"),
        ("StateMem", "statemem"),
    ]:
        print(
            f"{label:<18} | "
            f"{100*agg[key + '_acc']:8.2f}% | "
            f"{100*agg[key + '_noise_acc']:8.2f}% | "
            f"{agg[key + '_transition_lag']:14.3f}"
        )

    print("-" * 92)
    print(f"StateMem mean Brier: {agg['brier']:.4f}")
    print("=" * 92)


def print_condition_tables(rows: List[dict]) -> None:
    print("\nPer-condition TEST accuracy\n")
    modes = sorted(set(r["confidence_mode"] for r in rows))
    switch_probs = sorted(set(r["switch_prob"] for r in rows))
    corruptions = sorted(set(r["corruption_rate"] for r in rows))

    for mode in modes:
        for switch_prob in switch_probs:
            print(f"Confidence={mode} | true switch probability={switch_prob:.3f}")
            print(f"{'Noise':>7} | {'Latest':>8} | {'Recency':>8} | {'Rec×Conf':>9} | {'StateMem':>8} | {'Brier':>8}")
            print("-" * 70)

            for corruption in corruptions:
                match = [
                    r for r in rows
                    if r["confidence_mode"] == mode
                    and r["switch_prob"] == switch_prob
                    and r["corruption_rate"] == corruption
                ]
                if not match:
                    continue
                r = match[0]
                print(
                    f"{100*corruption:6.0f}% | "
                    f"{100*r['latest_acc']:7.2f}% | "
                    f"{100*r['recency_acc']:7.2f}% | "
                    f"{100*r['recency_conf_acc']:8.2f}% | "
                    f"{100*r['statemem_acc']:7.2f}% | "
                    f"{r['brier']:8.4f}"
                )
            print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-val", type=int, default=80)
    parser.add_argument("--episodes-test", type=int, default=160)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--corruptions", default="0,0.1,0.2,0.3,0.4")
    parser.add_argument("--switch-probs", default="0.01,0.05,0.10,0.20")
    parser.add_argument("--stay-probs", default="0.50,0.60,0.70,0.75,0.80,0.85,0.90,0.95,0.98")
    parser.add_argument("--confidence-modes", default="reliable,overlapping,misleading")
    parser.add_argument("--decay", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out-dir", default="statemem_stress_results")
    args = parser.parse_args()

    corruption_rates = parse_float_list(args.corruptions)
    switch_probs = parse_float_list(args.switch_probs)
    stay_probs = parse_float_list(args.stay_probs)
    confidence_modes = [x.strip() for x in args.confidence_modes.split(",") if x.strip()]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating validation grid...")
    val_rows = run_grid(
        args.episodes_val,
        args.steps,
        corruption_rates,
        switch_probs,
        confidence_modes,
        stay_probs,
        args.decay,
        args.seed,
    )
    save_csv(out_dir / "validation_grid.csv", val_rows)
    chosen_p = print_validation_summary(val_rows, stay_probs)

    print("\nGenerating independent test grid with frozen hyperparameter...")
    test_rows = run_grid(
        args.episodes_test,
        args.steps,
        corruption_rates,
        switch_probs,
        confidence_modes,
        [chosen_p],
        args.decay,
        args.seed + 1_000_000,
    )
    save_csv(out_dir / "test_grid.csv", test_rows)

    print_test_overall(test_rows, chosen_p)
    print_condition_tables(test_rows)

    config = {
        "selected_stay_probability": chosen_p,
        "episodes_validation_per_condition": args.episodes_val,
        "episodes_test_per_condition": args.episodes_test,
        "steps_per_episode": args.steps,
        "corruption_rates": corruption_rates,
        "switch_probabilities": switch_probs,
        "confidence_modes": confidence_modes,
        "candidate_stay_probabilities": stay_probs,
        "decay": args.decay,
        "seed": args.seed,
        "note": "stay_probability selected on validation only; test uses an independent seed",
    }
    with (out_dir / "experiment_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("\nSaved:")
    print(f"  {out_dir / 'validation_grid.csv'}")
    print(f"  {out_dir / 'test_grid.csv'}")
    print(f"  {out_dir / 'experiment_config.json'}")


if __name__ == "__main__":
    main()
