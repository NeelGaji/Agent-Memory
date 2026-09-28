#!/usr/bin/env python3
"""
StateMem downstream multi-object world-state QA
================================================

Purpose
-------
Evaluate whether the frozen memory state estimates support downstream reasoning
over *multiple objects in the same episode*, rather than only per-observation
state classification.

Tasks at each query time:
  1) Object-state QA:
       "Is each tracked object at START or GOAL?"
  2) Set retrieval:
       "Which tracked objects are currently at GOAL?"
  3) Count QA:
       "How many tracked objects are currently at GOAL?"
  4) Whole-world exact match:
       "Is the entire tracked world state correct?"

Methods:
  - Latest
  - Recency-Calibrated
  - StateMem-Calibrated

The script uses the already frozen VLM observations and hyperparameters from the
completed FindingDory real-RGB experiment. It performs NO VLM inference and NO
parameter tuning.

Important limitation
--------------------
This is an exploratory downstream analysis on the same official validation set
already examined in the state-estimation experiment. It is useful for showing
whether state estimates propagate into structured reasoning, but it is not a
fresh independent test split.

Run from:
  ~/Downloads/findingdory_rgb_statemem_temporal_calfix

Example:
  python3 ../statemem_downstream_world_qa.py \
      --steps test_step_results.csv \
      --overall test_overall.json \
      --out-dir downstream_world_qa \
      --bootstrap 20000
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

METHODS = ["Latest", "Recency-Calibrated", "StateMem-Calibrated"]


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


def stable_int(*parts):
    raw = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def load_observations(step_path):
    rows = []
    with Path(step_path).open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            # The same observation is repeated once per evaluated method.
            if r["method"] != "Latest":
                continue
            rows.append({
                "episode_id": str(r["episode_id"]),
                "interaction_index": str(r["interaction_index"]),
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
        key = (r["episode_id"], r["interaction_index"], r["object_handle"])
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
                ratios.append((r["frame_idx"] - prev) / r["delta_frame_units"])
            prev = r["frame_idx"]
    return float(np.median(ratios)) if ratios else 1.0


def predict_object_at(rows, query_frame, stay, decay, time_scale):
    latest = None
    history = []
    p_goal = 0.5
    prev_frame = None

    for obs in rows:
        if obs["frame_idx"] > query_frame:
            break

        if prev_frame is not None:
            delta = (obs["frame_idx"] - prev_frame) / max(time_scale, 1e-9)
            p_goal = predict_through_time(p_goal, stay, delta)

        if obs["vlm_state"] in {"START", "GOAL"}:
            latest = obs["vlm_state"]
            history.append(
                (obs["frame_idx"], obs["vlm_state"], obs["vlm_reliability"])
            )
            p_goal = bayes_update(
                p_goal, obs["vlm_state"], obs["vlm_reliability"]
            )

        prev_frame = obs["frame_idx"]

    if prev_frame is not None and query_frame > prev_frame:
        delta = (query_frame - prev_frame) / max(time_scale, 1e-9)
        p_goal = predict_through_time(p_goal, stay, delta)

    recency_pred, recency_p = recency_state(
        history, query_frame, decay, time_scale
    )

    return {
        "Latest": (latest, 1.0 if latest == "GOAL"
                   else 0.0 if latest == "START" else 0.5),
        "Recency-Calibrated": (recency_pred, recency_p),
        "StateMem-Calibrated": (hard_state(p_goal), p_goal),
    }


def build_query_records(grouped, stay, decay, time_scale):
    by_episode = defaultdict(list)
    for key, rows in grouped.items():
        by_episode[key[0]].append((key, rows))

    records = []

    for episode_id, objects in by_episode.items():
        # Multi-object reasoning only.
        if len(objects) < 2:
            continue

        # Start asking questions only after every tracked object has at least
        # one observation, so no method is punished simply for being uninitialized.
        init_frame = max(min(r["frame_idx"] for r in rows) for _, rows in objects)

        query_frames = sorted({
            r["frame_idx"]
            for _, rows in objects
            for r in rows
            if r["frame_idx"] >= init_frame
        })

        for qf in query_frames:
            gt = {}
            predictions = {m: {} for m in METHODS}
            probabilities = {m: {} for m in METHODS}

            for key, rows in objects:
                obj_id = f"{key[1]}::{key[2]}"
                goal_frames = [
                    r["frame_idx"] for r in rows if r["gt_state"] == "GOAL"
                ]
                if not goal_frames:
                    continue

                first_goal = min(goal_frames)
                gt[obj_id] = "GOAL" if qf >= first_goal else "START"

                pp = predict_object_at(rows, qf, stay, decay, time_scale)
                for method in METHODS:
                    predictions[method][obj_id] = pp[method][0]
                    probabilities[method][obj_id] = pp[method][1]

            if len(gt) < 2:
                continue

            gt_goal_set = {o for o, s in gt.items() if s == "GOAL"}
            gt_count = len(gt_goal_set)

            for method in METHODS:
                pred = predictions[method]
                pred_goal_set = {
                    o for o, s in pred.items() if s == "GOAL"
                }

                object_correct = [
                    int(pred.get(o) == gt[o]) for o in gt
                ]
                world_exact = int(all(object_correct))
                pred_count = len(pred_goal_set)

                inter = len(gt_goal_set & pred_goal_set)
                union = len(gt_goal_set | pred_goal_set)
                denom = len(gt_goal_set) + len(pred_goal_set)

                jaccard = 1.0 if union == 0 else inter / union
                set_f1 = 1.0 if denom == 0 else (2 * inter / denom)

                records.append({
                    "episode_id": episode_id,
                    "query_frame": qf,
                    "num_objects": len(gt),
                    "method": method,
                    "object_state_accuracy": float(np.mean(object_correct)),
                    "world_exact": world_exact,
                    "count_exact": int(pred_count == gt_count),
                    "count_abs_error": abs(pred_count - gt_count),
                    "goal_set_jaccard": jaccard,
                    "goal_set_f1": set_f1,
                    "gt_goal_count": gt_count,
                    "pred_goal_count": pred_count,
                })

    return records


def summarize(records):
    out = {}
    for method in METHODS:
        rr = [r for r in records if r["method"] == method]
        out[method] = {
            "num_queries": len(rr),
            "num_episodes": len(set(r["episode_id"] for r in rr)),
            "mean_objects_per_query": float(np.mean([r["num_objects"] for r in rr])),
            "object_state_accuracy": float(np.mean([r["object_state_accuracy"] for r in rr])),
            "whole_world_exact_accuracy": float(np.mean([r["world_exact"] for r in rr])),
            "count_exact_accuracy": float(np.mean([r["count_exact"] for r in rr])),
            "count_mae": float(np.mean([r["count_abs_error"] for r in rr])),
            "goal_set_jaccard": float(np.mean([r["goal_set_jaccard"] for r in rr])),
            "goal_set_f1": float(np.mean([r["goal_set_f1"] for r in rr])),
        }
    return out


def episode_cluster_bootstrap(records, metric, a, b, n_boot, seed):
    # One row per method per (episode, query_frame), so pair by query.
    paired = defaultdict(dict)
    for r in records:
        if r["method"] not in {a, b}:
            continue
        key = (r["episode_id"], r["query_frame"])
        paired[key][r["method"]] = r[metric]

    by_episode = defaultdict(list)
    for (ep, qf), d in paired.items():
        if a in d and b in d:
            by_episode[ep].append(float(d[a]) - float(d[b]))

    episodes = sorted(by_episode)
    values = [x for ep in episodes for x in by_episode[ep]]
    point = float(np.mean(values))

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sampled_eps = rng.choice(episodes, size=len(episodes), replace=True)
        sample_values = []
        for ep in sampled_eps:
            sample_values.extend(by_episode[ep])
        boots[i] = float(np.mean(sample_values))

    lo, hi = np.quantile(boots, [0.025, 0.975])
    return point, float(lo), float(hi)


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="test_step_results.csv")
    ap.add_argument("--overall", default="test_overall.json")
    ap.add_argument("--out-dir", default="downstream_world_qa")
    ap.add_argument("--bootstrap", type=int, default=20000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped = load_observations(args.steps)
    with Path(args.overall).open("r", encoding="utf-8") as f:
        overall = json.load(f)

    stay = float(overall["selected_stay"])
    decay = float(overall["selected_recency_decay"])
    reliability = float(overall["global_calibration_reliability"])
    time_scale = infer_time_scale(grouped)

    records = build_query_records(grouped, stay, decay, time_scale)
    summary = summarize(records)

    comparisons = []
    metrics = [
        "object_state_accuracy",
        "world_exact",
        "count_exact",
        "count_abs_error",
        "goal_set_jaccard",
        "goal_set_f1",
    ]

    for baseline in ["Latest", "Recency-Calibrated"]:
        for metric in metrics:
            point, lo, hi = episode_cluster_bootstrap(
                records,
                metric,
                "StateMem-Calibrated",
                baseline,
                args.bootstrap,
                stable_int(metric, baseline) % (2**32),
            )
            comparisons.append({
                "comparison": f"StateMem-Calibrated - {baseline}",
                "metric": metric,
                "difference": point,
                "ci95_low": lo,
                "ci95_high": hi,
                "lower_is_better": metric == "count_abs_error",
            })

    report = {
        "analysis": "downstream_multi_object_world_state_qa",
        "status": (
            "exploratory: same official validation set previously examined"
        ),
        "frozen_parameters": {
            "stay": stay,
            "recency_decay": decay,
            "global_calibration_reliability": reliability,
            "time_scale": time_scale,
        },
        "protocol": {
            "multi_object_episodes_only": True,
            "query_start_rule": (
                "queries begin only after every tracked object has at least "
                "one observation"
            ),
            "query_frames": (
                "union of selected official RGB evidence frames within episode"
            ),
            "tasks": [
                "per-object START/GOAL state QA",
                "GOAL-object set retrieval",
                "GOAL-object count QA",
                "whole-world exact state",
            ],
            "no_vlm_rerun": True,
            "no_parameter_retuning": True,
            "episode_cluster_bootstrap_resamples": args.bootstrap,
        },
        "summary": summary,
    }

    (out_dir / "downstream_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    write_csv(out_dir / "downstream_query_results.csv", records)
    write_csv(out_dir / "downstream_bootstrap.csv", comparisons)

    print("=" * 96)
    print("STATEMEM DOWNSTREAM MULTI-OBJECT WORLD-STATE QA")
    print("=" * 96)
    print(f"Frozen stay: {stay}")
    print(f"Frozen recency decay: {decay}")
    print(f"Frozen calibration reliability: {reliability:.6f}")
    print(f"Inferred frame time-scale: {time_scale}")
    print()

    for method in METHODS:
        s = summary[method]
        print(
            f"{method:<22} "
            f"object={100*s['object_state_accuracy']:.2f}%  "
            f"world_exact={100*s['whole_world_exact_accuracy']:.2f}%  "
            f"count_exact={100*s['count_exact_accuracy']:.2f}%  "
            f"set_F1={100*s['goal_set_f1']:.2f}%  "
            f"count_MAE={s['count_mae']:.3f}"
        )

    print()
    print("Paired episode-cluster bootstrap:")
    for row in comparisons:
        if row["metric"] not in {
            "object_state_accuracy",
            "world_exact",
            "count_exact",
            "goal_set_f1",
        }:
            continue
        sign = 100 if row["metric"] != "count_abs_error" else 1
        suffix = " pp" if sign == 100 else ""
        print(
            f"  {row['comparison']:<48} {row['metric']:<22} "
            f"{sign*row['difference']:+.2f}{suffix} "
            f"[{sign*row['ci95_low']:+.2f}, {sign*row['ci95_high']:+.2f}]"
        )

    print()
    print("Outputs:")
    print(out_dir / "downstream_summary.json")
    print(out_dir / "downstream_query_results.csv")
    print(out_dir / "downstream_bootstrap.csv")


if __name__ == "__main__":
    main()
