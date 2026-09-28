#!/usr/bin/env python3
"""
StateMem temporal-stress robustness analysis
============================================

Uses the frozen predictions from the completed real-RGB FindingDory experiment.
NO VLM inference and NO parameter re-tuning are performed.

Predefined stress regimes:
  full              : original observation stream
  delay_goal_1      : hide first post-transition GOAL observation
  delay_goal_2      : hide first two post-transition GOAL observations
  delay_goal_3      : hide first three post-transition GOAL observations
  sparse_75         : retain 75% of observations after the first START anchor
  sparse_50         : retain 50%
  sparse_33         : retain 33%

Sparse regimes average over 20 deterministic, label-independent masks.

Statistics:
  - episode-cluster paired bootstrap
  - StateMem-Calibrated vs Latest
  - StateMem-Calibrated vs Recency-Calibrated

Important:
This is a post-hoc robustness analysis on the same official validation set.
It is useful evidence about behavior under stress, but it is not a fresh
confirmatory test set.

Run:
  cd ~/Downloads/findingdory_rgb_statemem_temporal_calfix
  python3 ../statemem_temporal_stress_test.py \
      --steps test_step_results.csv \
      --overall test_overall.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


METHODS = [
    "Latest",
    "Recency-Calibrated",
    "StateMem-Calibrated",
]


def clamp(x, lo=1e-6, hi=1 - 1e-6):
    return max(lo, min(hi, x))


def logit(p):
    p = clamp(p)
    return math.log(p / (1 - p))


def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)


def effective_stay(base_stay, delta):
    base_stay = max(0.500001, min(0.999999, base_stay))
    return 0.5 + 0.5 * ((2 * base_stay - 1) ** max(0.0, float(delta)))


def predict_through_time(p_goal, base_stay, delta):
    s = effective_stay(base_stay, delta)
    return s * p_goal + (1 - s) * (1 - p_goal)


def bayes_update(p_goal, state, reliability):
    if state not in {"START", "GOAL"} or reliability is None:
        return p_goal

    r = clamp(float(reliability), 0.001, 0.999)
    if state == "GOAL":
        like_goal, like_start = r, 1 - r
    else:
        like_goal, like_start = 1 - r, r

    num = p_goal * like_goal
    den = num + (1 - p_goal) * like_start
    return p_goal if den <= 0 else num / den


def hard_state(p_goal):
    return "GOAL" if p_goal >= 0.5 else "START"


def recency_state(history, current_frame, decay, time_scale):
    if not history:
        return None, 0.5

    score = 0.0
    for frame_idx, state, reliability in history:
        age = max(0, current_frame - frame_idx) / max(time_scale, 1e-9)
        weight = math.exp(-decay * age)
        signed = logit(clamp(reliability, 0.01, 0.99))
        score += weight * signed * (1 if state == "GOAL" else -1)

    p_goal = sigmoid(score)
    return hard_state(p_goal), p_goal


def load_observations(step_path):
    rows = []
    with Path(step_path).open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            # Every method row repeats the same VLM observation; keep one copy.
            if r["method"] != "Latest":
                continue

            rows.append({
                "episode_id": r["episode_id"],
                "interaction_index": r["interaction_index"],
                "object_handle": r["object_handle"],
                "frame_idx": int(r["frame_idx"]),
                "gt_state": r["gt_state"],
                "vlm_state": r["vlm_state"] or None,
                "vlm_consistency": r["vlm_consistency"],
                "vlm_reliability": (
                    float(r["vlm_reliability"])
                    if r["vlm_reliability"] else None
                ),
                "delta_frame_units": float(r["delta_frame_units"]),
            })

    grouped = defaultdict(list)
    for r in rows:
        key = (
            r["episode_id"],
            r["interaction_index"],
            r["object_handle"],
        )
        grouped[key].append(r)

    for key in grouped:
        grouped[key].sort(key=lambda x: x["frame_idx"])

    return grouped


def infer_time_scale(grouped):
    ratios = []
    for rows in grouped.values():
        prev = None
        for r in rows:
            if prev is not None and r["delta_frame_units"] > 0:
                ratios.append(
                    (r["frame_idx"] - prev) / r["delta_frame_units"]
                )
            prev = r["frame_idx"]

    if not ratios:
        return 1.0
    return float(np.median(ratios))


def stable_int(*parts):
    s = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(s).digest()[:8], "big")


def stable_keep(key_parts, probability, seed):
    s = "|".join(map(str, (*key_parts, seed))).encode("utf-8")
    raw = int.from_bytes(hashlib.sha256(s).digest()[:8], "big")
    u = raw / 2**64
    return u < probability


def full_mask(obs, idx, rows, key):
    return True


def delay_mask(k):
    def mask(obs, idx, rows, key):
        if obs["gt_state"] == "START":
            return True

        goal_idx = (
            sum(1 for x in rows[:idx + 1] if x["gt_state"] == "GOAL") - 1
        )
        return goal_idx >= k

    return mask


def sparse_mask(probability, seed):
    def mask(obs, idx, rows, key):
        # Preserve one initial observation so all methods receive the same
        # minimum state anchor before availability becomes sparse.
        first_start = next(
            (i for i, x in enumerate(rows) if x["gt_state"] == "START"),
            0,
        )
        if idx == first_start:
            return True

        return stable_keep(
            (
                key[0],
                key[1],
                key[2],
                obs["frame_idx"],
            ),
            probability,
            seed,
        )

    return mask


def evaluate(grouped, stay, decay, global_r, time_scale, mask_fn):
    records = []

    for key, rows in grouped.items():
        latest = None
        recency_history = []
        p_cal = 0.5
        prev_frame = None
        goal_idx = 0

        for idx, obs in enumerate(rows):
            if prev_frame is not None:
                p_cal = predict_through_time(
                    p_cal,
                    stay,
                    obs["delta_frame_units"],
                )

            available = bool(mask_fn(obs, idx, rows, key))

            if available and obs["vlm_state"] in {"START", "GOAL"}:
                latest = obs["vlm_state"]
                recency_history.append((
                    obs["frame_idx"],
                    obs["vlm_state"],
                    obs["vlm_reliability"],
                ))
                p_cal = bayes_update(
                    p_cal,
                    obs["vlm_state"],
                    obs["vlm_reliability"],
                )

            rec_pred, rec_p = recency_state(
                recency_history,
                obs["frame_idx"],
                decay,
                time_scale,
            )

            predictions = {
                "Latest": (
                    latest,
                    1.0 if latest == "GOAL"
                    else 0.0 if latest == "START"
                    else 0.5,
                ),
                "Recency-Calibrated": (rec_pred, rec_p),
                "StateMem-Calibrated": (
                    hard_state(p_cal),
                    p_cal,
                ),
            }

            current_goal_idx = goal_idx if obs["gt_state"] == "GOAL" else None
            if obs["gt_state"] == "GOAL":
                goal_idx += 1

            for method, (pred, prob_goal) in predictions.items():
                records.append({
                    "episode_id": key[0],
                    "interaction_index": key[1],
                    "object_handle": key[2],
                    "frame_idx": obs["frame_idx"],
                    "gt_state": obs["gt_state"],
                    "goal_idx": current_goal_idx,
                    "method": method,
                    "prediction": pred,
                    "correct": int(pred == obs["gt_state"]),
                    "prob_goal": prob_goal,
                    "available": int(available),
                    "vlm_consistency": obs["vlm_consistency"],
                })

            prev_frame = obs["frame_idx"]

    return records


def summarize(records):
    result = {}

    for method in METHODS:
        rows = [r for r in records if r["method"] == method]
        start = [r for r in rows if r["gt_state"] == "START"]
        goal = [r for r in rows if r["gt_state"] == "GOAL"]
        immediate = [r for r in goal if r["goal_idx"] == 0]
        late = [r for r in goal if r["goal_idx"] != 0]

        by_transition = defaultdict(list)
        for r in rows:
            by_transition[
                (
                    r["episode_id"],
                    r["interaction_index"],
                    r["object_handle"],
                )
            ].append(r)

        finals = []
        misses = []
        lags = []

        for transition_rows in by_transition.values():
            transition_rows.sort(key=lambda x: x["frame_idx"])
            finals.append(
                int(transition_rows[-1]["prediction"] == "GOAL")
            )

            goal_rows = [
                r for r in transition_rows
                if r["gt_state"] == "GOAL"
            ]
            hit_indices = [
                i for i, r in enumerate(goal_rows)
                if r["prediction"] == "GOAL"
            ]

            if hit_indices:
                misses.append(0)
                lags.append(hit_indices[0])
            else:
                misses.append(1)

        def acc(rows_):
            return (
                sum(r["correct"] for r in rows_) / len(rows_)
                if rows_ else None
            )

        brier = np.mean([
            (
                r["prob_goal"]
                - (1.0 if r["gt_state"] == "GOAL" else 0.0)
            ) ** 2
            for r in rows
        ])

        result[method] = {
            "accuracy": acc(rows),
            "start_accuracy": acc(start),
            "goal_accuracy": acc(goal),
            "immediate_goal_accuracy": acc(immediate),
            "late_goal_accuracy": acc(late),
            "final_goal_accuracy": float(np.mean(finals)),
            "transition_miss_rate": float(np.mean(misses)),
            "goal_recovery_lag_observations": (
                float(np.mean(lags)) if lags else None
            ),
            "brier": float(brier),
            "num_steps": len(rows),
            "num_transitions": len(by_transition),
        }

    return result


def aligned_correctness(records, method_a, method_b, filter_fn=None):
    by_key = {}

    for r in records:
        if r["method"] not in {method_a, method_b}:
            continue
        if filter_fn is not None and not filter_fn(r):
            continue

        key = (
            r["episode_id"],
            r["interaction_index"],
            r["object_handle"],
            r["frame_idx"],
        )
        by_key.setdefault(key, {})[r["method"]] = r["correct"]

    by_episode = defaultdict(list)
    for key, d in by_key.items():
        if method_a in d and method_b in d:
            by_episode[key[0]].append(
                d[method_a] - d[method_b]
            )

    return by_episode


def cluster_bootstrap_diff(
    by_episode,
    n_boot=20000,
    seed=20260920,
):
    episodes = list(by_episode)
    if not episodes:
        return None, None, None

    point = float(np.mean([
        x for ep in episodes for x in by_episode[ep]
    ]))

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        sample = rng.choice(
            episodes,
            size=len(episodes),
            replace=True,
        )
        values = []
        for ep in sample:
            values.extend(by_episode[ep])
        boots[b] = np.mean(values)

    lo, hi = np.quantile(boots, [0.025, 0.975])
    return point, float(lo), float(hi)


def sparse_expected(
    grouped,
    stay,
    decay,
    global_r,
    time_scale,
    keep_probability,
    seeds,
):
    all_runs = []

    for seed in seeds:
        records = evaluate(
            grouped,
            stay,
            decay,
            global_r,
            time_scale,
            sparse_mask(keep_probability, seed),
        )
        all_runs.append(records)

    # Aggregate per-step correctness across label-independent masking seeds.
    sums = {
        method: defaultdict(list)
        for method in METHODS
    }
    metadata = {}

    for records in all_runs:
        for r in records:
            key = (
                r["episode_id"],
                r["interaction_index"],
                r["object_handle"],
                r["frame_idx"],
            )
            sums[r["method"]][key].append(r["correct"])
            metadata[key] = r

    avg_correct = {
        method: {
            key: float(np.mean(values))
            for key, values in d.items()
        }
        for method, d in sums.items()
    }

    mean_accuracy = {
        method: float(np.mean(list(avg_correct[method].values())))
        for method in METHODS
    }

    return all_runs, avg_correct, metadata, mean_accuracy


def sparse_cluster_diff(
    avg_correct,
    metadata,
    method_a,
    method_b,
    n_boot,
    seed,
):
    by_episode = defaultdict(list)

    keys = (
        set(avg_correct[method_a])
        & set(avg_correct[method_b])
    )

    for key in keys:
        ep = metadata[key]["episode_id"]
        by_episode[ep].append(
            avg_correct[method_a][key]
            - avg_correct[method_b][key]
        )

    return cluster_bootstrap_diff(
        by_episode,
        n_boot=n_boot,
        seed=seed,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="test_step_results.csv")
    ap.add_argument("--overall", default="test_overall.json")
    ap.add_argument("--out-dir", default="temporal_stress_test")
    ap.add_argument("--bootstrap", type=int, default=20000)
    ap.add_argument("--sparse-seeds", type=int, default=20)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped = load_observations(args.steps)
    with Path(args.overall).open("r", encoding="utf-8") as f:
        original = json.load(f)

    stay = float(original["selected_stay"])
    decay = float(original["selected_recency_decay"])
    global_r = float(original["global_calibration_reliability"])
    time_scale = infer_time_scale(grouped)

    print("=" * 94)
    print("STATEMEM TEMPORAL-STRESS ROBUSTNESS TEST")
    print("=" * 94)
    print("Frozen stay:", stay)
    print("Frozen recency decay:", decay)
    print("Frozen global reliability:", round(global_r, 6))
    print("Inferred frozen frame time-scale:", time_scale)
    print("Transitions:", len(grouped))
    print()

    # Baseline reconstruction sanity check.
    baseline = evaluate(
        grouped,
        stay,
        decay,
        global_r,
        time_scale,
        full_mask,
    )
    baseline_summary = summarize(baseline)

    for method in METHODS:
        expected = original["methods"][method]["accuracy"]
        observed = baseline_summary[method]["accuracy"]
        if abs(expected - observed) > 1e-12:
            raise RuntimeError(
                f"Baseline reconstruction mismatch for {method}: "
                f"{observed} != {expected}"
            )

    print("Baseline reconstruction: EXACT MATCH ✓")
    print()

    result_rows = []
    bootstrap_rows = []

    deterministic_regimes = [
        ("full", full_mask),
        ("delay_goal_1", delay_mask(1)),
        ("delay_goal_2", delay_mask(2)),
        ("delay_goal_3", delay_mask(3)),
    ]

    for regime, mask_fn in deterministic_regimes:
        records = evaluate(
            grouped,
            stay,
            decay,
            global_r,
            time_scale,
            mask_fn,
        )
        summary = summarize(records)

        for method in METHODS:
            result_rows.append({
                "regime": regime,
                "method": method,
                **summary[method],
            })

        for baseline_method in (
            "Latest",
            "Recency-Calibrated",
        ):
            by_ep = aligned_correctness(
                records,
                "StateMem-Calibrated",
                baseline_method,
            )
            point, lo, hi = cluster_bootstrap_diff(
                by_ep,
                n_boot=args.bootstrap,
                seed=stable_int(regime, baseline_method) % (2**32),
            )
            bootstrap_rows.append({
                "regime": regime,
                "comparison": (
                    f"StateMem-Calibrated - {baseline_method}"
                ),
                "metric": "overall_accuracy",
                "difference": point,
                "ci95_low": lo,
                "ci95_high": hi,
            })

        print(regime)
        for method in METHODS:
            x = summary[method]
            print(
                f"  {method:<22} "
                f"overall={100*x['accuracy']:.2f}% "
                f"late={100*x['late_goal_accuracy']:.2f}% "
                f"final={100*x['final_goal_accuracy']:.2f}%"
            )
        print()

    # Sparse availability stress. Masking is deterministic and independent of
    # labels/model correctness. Report expected performance over fixed seeds.
    sparse_specs = [
        ("sparse_75", 0.75),
        ("sparse_50", 0.50),
        ("sparse_33", 0.33),
    ]
    seeds = list(range(args.sparse_seeds))

    sparse_detail = {}

    for regime, probability in sparse_specs:
        all_runs, avg_correct, metadata, mean_accuracy = sparse_expected(
            grouped,
            stay,
            decay,
            global_r,
            time_scale,
            probability,
            seeds,
        )

        # Mean full metric summary across masking seeds.
        per_method_summaries = {
            method: []
            for method in METHODS
        }

        for records in all_runs:
            s = summarize(records)
            for method in METHODS:
                per_method_summaries[method].append(s[method])

        for method in METHODS:
            metrics = per_method_summaries[method]
            row = {
                "regime": regime,
                "method": method,
            }
            for field in (
                "accuracy",
                "start_accuracy",
                "goal_accuracy",
                "immediate_goal_accuracy",
                "late_goal_accuracy",
                "final_goal_accuracy",
                "transition_miss_rate",
                "goal_recovery_lag_observations",
                "brier",
            ):
                vals = [
                    x[field]
                    for x in metrics
                    if x[field] is not None
                ]
                row[field] = (
                    float(np.mean(vals)) if vals else None
                )

            row["num_steps"] = metrics[0]["num_steps"]
            row["num_transitions"] = metrics[0]["num_transitions"]
            result_rows.append(row)

        for baseline_method in (
            "Latest",
            "Recency-Calibrated",
        ):
            point, lo, hi = sparse_cluster_diff(
                avg_correct,
                metadata,
                "StateMem-Calibrated",
                baseline_method,
                n_boot=args.bootstrap,
                seed=stable_int(regime, baseline_method) % (2**32),
            )
            bootstrap_rows.append({
                "regime": regime,
                "comparison": (
                    f"StateMem-Calibrated - {baseline_method}"
                ),
                "metric": "overall_accuracy",
                "difference": point,
                "ci95_low": lo,
                "ci95_high": hi,
            })

        sparse_detail[regime] = {
            "keep_probability": probability,
            "num_mask_seeds": len(seeds),
            "mean_accuracy": mean_accuracy,
        }

        print(regime, f"(retain {int(probability*100)}%)")
        for method in METHODS:
            vals = per_method_summaries[method]
            print(
                f"  {method:<22} "
                f"overall={100*np.mean([x['accuracy'] for x in vals]):.2f}% "
                f"late={100*np.mean([x['late_goal_accuracy'] for x in vals]):.2f}% "
                f"final={100*np.mean([x['final_goal_accuracy'] for x in vals]):.2f}%"
            )
        print()

    # Natural observable evidence-quality slices in the untouched full stream.
    natural_slices = [
        ("consistent_current_evidence",
         lambda r: r["vlm_consistency"] == "consistent"),
        ("single_judged_current_evidence",
         lambda r: r["vlm_consistency"] == "single_judged"),
        ("no_current_state_judgment",
         lambda r: r["vlm_consistency"] in {
             "conflict", "double_abstain"
         }),
        ("weak_or_absent_current_evidence",
         lambda r: r["vlm_consistency"] != "consistent"),
    ]

    slice_rows = []
    for slice_name, fn in natural_slices:
        for method in METHODS:
            rr = [
                r for r in baseline
                if r["method"] == method and fn(r)
            ]
            slice_rows.append({
                "slice": slice_name,
                "method": method,
                "n": len(rr),
                "accuracy": (
                    sum(r["correct"] for r in rr) / len(rr)
                    if rr else None
                ),
            })

        for baseline_method in (
            "Latest",
            "Recency-Calibrated",
        ):
            by_ep = aligned_correctness(
                baseline,
                "StateMem-Calibrated",
                baseline_method,
                filter_fn=fn,
            )
            point, lo, hi = cluster_bootstrap_diff(
                by_ep,
                n_boot=args.bootstrap,
                seed=stable_int(slice_name, baseline_method) % (2**32),
            )
            bootstrap_rows.append({
                "regime": slice_name,
                "comparison": (
                    f"StateMem-Calibrated - {baseline_method}"
                ),
                "metric": "slice_accuracy",
                "difference": point,
                "ci95_low": lo,
                "ci95_high": hi,
            })

    def write_csv(path, rows):
        keys = list(rows[0].keys())
        with Path(path).open(
            "w",
            encoding="utf-8",
            newline="",
        ) as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    write_csv(
        out_dir / "stress_results.csv",
        result_rows,
    )
    write_csv(
        out_dir / "stress_bootstrap.csv",
        bootstrap_rows,
    )
    write_csv(
        out_dir / "natural_evidence_slices.csv",
        slice_rows,
    )

    report = {
        "analysis_type": "post_hoc_temporal_robustness",
        "confirmatory_status": (
            "exploratory: same official validation set was previously examined"
        ),
        "frozen_parameters": {
            "stay": stay,
            "recency_decay": decay,
            "global_calibration_reliability": global_r,
            "time_scale": time_scale,
        },
        "num_transitions": len(grouped),
        "num_episodes": len(set(k[0] for k in grouped)),
        "bootstrap_resamples": args.bootstrap,
        "sparse_mask_seeds": args.sparse_seeds,
        "sparse_detail": sparse_detail,
        "baseline_exactly_reconstructed": True,
        "regimes": [
            "full",
            "delay_goal_1",
            "delay_goal_2",
            "delay_goal_3",
            "sparse_75",
            "sparse_50",
            "sparse_33",
        ],
        "interpretation": (
            "The stress test asks whether StateMem degrades more slowly than "
            "Latest/Recency when fresh observations are delayed or sparse. "
            "No VLM outputs, model parameters, thresholds, or calibration "
            "parameters are changed."
        ),
    }

    (out_dir / "stress_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("=" * 94)
    print("OUTPUTS")
    print("=" * 94)
    for name in (
        "stress_report.json",
        "stress_results.csv",
        "stress_bootstrap.csv",
        "natural_evidence_slices.csv",
    ):
        print(out_dir / name)


if __name__ == "__main__":
    main()
