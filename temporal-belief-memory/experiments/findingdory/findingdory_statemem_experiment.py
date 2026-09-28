#!/usr/bin/env python3
"""
FindingDory native-transition StateMem diagnostic
=================================================

Uses REAL FindingDory object rearrangements:
    start receptacle -> goal receptacle
    initial XYZ -> goal XYZ

The state transition itself is native FindingDory/Habitat data.

Still controlled in this diagnostic:
- when the endpoint transition occurs inside a short evidence sequence
- whether the object is observed on a step
- observation corruption
- raw perception confidence

Protocol
--------
TRAIN split only:
  - scene-disjoint calibration subset -> fit ONE pooled confidence calibrator
  - scene-disjoint tuning subset      -> tune StateMem hyperparameters

Official FindingDory VAL split:
  - untouched final test

Methods
-------
- Latest
- Recency
- Recency x Raw Confidence
- StateMem-Raw
- StateMem-Calibrated
- StateMem-Adaptive

The adaptive version only changes persistence when a high-reliability
observation contradicts the current belief.

Outputs
-------
findingdory_statemem_results/
  split_report.json
  pooled_calibration.json
  validation_stay_sweep.csv
  validation_adaptive_sweep.csv
  test_results.csv
  test_overall.json
  test_slices.json
  experiment_config.json

No third-party packages required.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

METHODS = [
    "latest",
    "recency",
    "recency_rawconf",
    "statemem_raw",
    "statemem_calibrated",
    "statemem_adaptive",
]


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def stable_seed(base_seed, *parts):
    s = "|".join([str(base_seed)] + [str(x) for x in parts])
    d = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(d[:8], "big") & 0x7FFFFFFF


def mean_valid(xs):
    vals = [
        x for x in xs
        if not (isinstance(x, float) and math.isnan(x))
    ]
    return sum(vals) / len(vals) if vals else float("nan")


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def parse_float_list(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


# ---------------------------------------------------------------------
# FindingDory native transitions
# ---------------------------------------------------------------------

def load_transitions(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)

            # Every extracted FindingDory target should have a true relocation.
            if not r.get("start_receptacle") or not r.get("goal_receptacle"):
                continue
            if r["start_receptacle"] == r["goal_receptacle"]:
                continue

            rows.append(r)

    return rows


def make_train_scene_split(train_rows, seed):
    scenes = sorted({r["scene_id"] for r in train_rows})
    rng = random.Random(seed)
    rng.shuffle(scenes)

    cut = max(1, len(scenes) // 2)

    calibration = set(scenes[:cut])
    tuning = set(scenes[cut:])

    return {
        "calibration": sorted(calibration),
        "tuning": sorted(tuning),
    }


def rows_for_scenes(rows, scenes):
    s = set(scenes)
    return [r for r in rows if r["scene_id"] in s]


# ---------------------------------------------------------------------
# Controlled evidence stream around a REAL transition
# ---------------------------------------------------------------------

def sample_confidence(correct, mode, rng):
    if mode == "reliable":
        return (
            rng.uniform(0.70, 0.98)
            if correct
            else rng.uniform(0.10, 0.55)
        )

    if mode == "overlapping":
        return (
            rng.uniform(0.55, 0.95)
            if correct
            else rng.uniform(0.35, 0.85)
        )

    if mode == "misleading":
        return (
            rng.uniform(0.45, 0.85)
            if correct
            else rng.uniform(0.55, 0.95)
        )

    raise ValueError(mode)


def make_evidence_sequence(
    row,
    observation_rate,
    corruption_rate,
    confidence_mode,
    pre_steps,
    post_steps,
    seed,
):
    """
    The TRUE states come directly from FindingDory:
      before transition = start receptacle
      after transition  = goal receptacle

    Only evidence availability/noise/confidence are controlled.
    """
    rng = random.Random(seed)

    start_state = row["start_receptacle"]
    goal_state = row["goal_receptacle"]

    seq = []

    total = pre_steps + post_steps

    for t in range(total):
        truth = start_state if t < pre_steps else goal_state

        # Ensure fair initialization and at least one opportunity immediately
        # after the genuine transition.
        observed = (
            t == 0
            or t == pre_steps
            or rng.random() < observation_rate
        )

        if observed:
            corrupted = rng.random() < corruption_rate
            observed_state = (
                goal_state if truth == start_state else start_state
            ) if corrupted else truth

            raw_conf = sample_confidence(
                correct=(observed_state == truth),
                mode=confidence_mode,
                rng=rng,
            )
        else:
            corrupted = False
            observed_state = None
            raw_conf = None

        seq.append({
            "t": t,
            "true_state": truth,
            "observed": observed,
            "observed_state": observed_state,
            "raw_confidence": raw_conf,
            "corrupted": corrupted,
            "post_transition": t >= pre_steps,
            "early_post": pre_steps <= t < pre_steps + 3,
        })

    return seq


# ---------------------------------------------------------------------
# Pooled confidence calibration
# ---------------------------------------------------------------------

class HistogramCalibrator:
    def __init__(self, bins=12, alpha=4.0, beta=4.0):
        self.bins = bins
        self.alpha = alpha
        self.beta = beta
        self.correct = [0] * bins
        self.total = [0] * bins
        self.global_correct = 0
        self.global_total = 0

    def _bin(self, c):
        c = max(0.0, min(0.999999, float(c)))
        return min(self.bins - 1, int(c * self.bins))

    def fit_one(self, c, correct):
        i = self._bin(c)
        self.correct[i] += int(correct)
        self.total[i] += 1
        self.global_correct += int(correct)
        self.global_total += 1

    def predict(self, c):
        i = self._bin(c)

        if self.total[i] == 0:
            if self.global_total == 0:
                return 0.75
            return (
                self.global_correct + self.alpha
            ) / (
                self.global_total + self.alpha + self.beta
            )

        return (
            self.correct[i] + self.alpha
        ) / (
            self.total[i] + self.alpha + self.beta
        )

    def table(self):
        out = []
        for i in range(self.bins):
            center = (i + 0.5) / self.bins
            out.append({
                "raw_confidence_center": center,
                "estimated_reliability": self.predict(center),
                "count": self.total[i],
            })
        return out


def fit_pooled_calibrator(
    rows,
    observation_rates,
    corruptions,
    confidence_modes,
    pre_steps,
    post_steps,
    bins,
    seed,
):
    """
    ONE calibrator pooled over every confidence regime.

    This is intentionally stricter than our early synthetic diagnostic, which
    used separate calibrators per regime.
    """
    cal = HistogramCalibrator(bins=bins)

    for mode in confidence_modes:
        for obsr in observation_rates:
            for cr in corruptions:
                for row in rows:
                    s = stable_seed(
                        seed,
                        "calibration",
                        row["episode_id"],
                        row["object_handle"],
                        mode,
                        obsr,
                        cr,
                    )

                    seq = make_evidence_sequence(
                        row=row,
                        observation_rate=obsr,
                        corruption_rate=cr,
                        confidence_mode=mode,
                        pre_steps=pre_steps,
                        post_steps=post_steps,
                        seed=s,
                    )

                    for x in seq:
                        if x["observed"]:
                            cal.fit_one(
                                x["raw_confidence"],
                                x["observed_state"] == x["true_state"],
                            )

    return cal


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------

class RecencyTracker:
    def __init__(self, states, decay, use_confidence):
        self.states = list(states)
        self.factor = math.exp(-decay)
        self.use_confidence = use_confidence
        self.scores = {s: 0.0 for s in self.states}

    def step(self, observed_state=None, confidence=None):
        for s in self.states:
            self.scores[s] *= self.factor

        if observed_state is not None:
            w = confidence if self.use_confidence else 1.0
            self.scores[observed_state] += w

        return max(self.scores, key=self.scores.get)


class BinaryStateMem:
    def __init__(
        self,
        states,
        stay_probability,
        calibrator=None,
        use_raw=False,
        adaptive_strength=0.0,
    ):
        assert len(states) == 2
        self.states = list(states)
        self.base_stay = stay_probability
        self.calibrator = calibrator
        self.use_raw = use_raw
        self.adaptive_strength = adaptive_strength
        self.belief = {s: 0.5 for s in self.states}

    def _reliability(self, raw_conf):
        if self.use_raw:
            return max(0.01, min(0.99, float(raw_conf)))
        return max(
            0.01,
            min(0.99, float(self.calibrator.predict(raw_conf))),
        )

    def step(self, observed_state=None, raw_confidence=None):
        stay = self.base_stay

        if (
            observed_state is not None
            and self.adaptive_strength > 0.0
        ):
            current = max(self.belief, key=self.belief.get)
            r = self._reliability(raw_confidence)

            # High-reliability contradiction -> become temporarily less sticky.
            if observed_state != current and r > 0.5:
                contradiction_strength = (r - 0.5) / 0.5
                stay = max(
                    0.50,
                    self.base_stay
                    - self.adaptive_strength * contradiction_strength,
                )

        a, b = self.states

        prior = {
            a: self.belief[a] * stay + self.belief[b] * (1.0 - stay),
            b: self.belief[b] * stay + self.belief[a] * (1.0 - stay),
        }

        if observed_state is None:
            self.belief = prior
            return dict(self.belief)

        r = self._reliability(raw_confidence)

        likelihood = {
            s: r if s == observed_state else (1.0 - r)
            for s in self.states
        }

        post = {
            s: prior[s] * likelihood[s]
            for s in self.states
        }
        z = sum(post.values())

        self.belief = {
            s: post[s] / z
            for s in self.states
        }
        return dict(self.belief)

    def predict(self):
        return max(self.belief, key=self.belief.get)


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def transition_lag(truth, preds, transition_index):
    goal = truth[transition_index]

    for j in range(transition_index, len(truth)):
        if preds[j] == goal:
            return j - transition_index, False

    return len(truth) - transition_index, True


def evaluate_one(
    row,
    seq,
    calibrator,
    calibrated_stay,
    adaptive_stay,
    adaptive_strength,
    decay,
    pre_steps,
):
    states = [
        row["start_receptacle"],
        row["goal_receptacle"],
    ]

    latest = None
    rec = RecencyTracker(states, decay, False)
    recc = RecencyTracker(states, decay, True)

    raw = BinaryStateMem(
        states,
        calibrated_stay,
        use_raw=True,
    )
    cal = BinaryStateMem(
        states,
        calibrated_stay,
        calibrator=calibrator,
    )
    ada = BinaryStateMem(
        states,
        adaptive_stay,
        calibrator=calibrator,
        adaptive_strength=adaptive_strength,
    )

    preds = {m: [] for m in METHODS}
    truth = []

    counts = {
        m: {
            "all": 0,
            "post": 0,
            "early": 0,
            "noise": 0,
        }
        for m in METHODS
    }

    n_all = n_post = n_early = n_noise = 0
    raw_brier = cal_brier = ada_brier = 0.0

    for x in seq:
        if x["observed"]:
            latest = x["observed_state"]
            p_rec = rec.step(
                x["observed_state"],
                x["raw_confidence"],
            )
            p_recc = recc.step(
                x["observed_state"],
                x["raw_confidence"],
            )
            b_raw = raw.step(
                x["observed_state"],
                x["raw_confidence"],
            )
            b_cal = cal.step(
                x["observed_state"],
                x["raw_confidence"],
            )
            b_ada = ada.step(
                x["observed_state"],
                x["raw_confidence"],
            )
        else:
            p_rec = rec.step()
            p_recc = recc.step()
            b_raw = raw.step()
            b_cal = cal.step()
            b_ada = ada.step()

        pm = {
            "latest": latest,
            "recency": p_rec,
            "recency_rawconf": p_recc,
            "statemem_raw": raw.predict(),
            "statemem_calibrated": cal.predict(),
            "statemem_adaptive": ada.predict(),
        }

        y = x["true_state"]
        truth.append(y)

        n_all += 1
        if x["post_transition"]:
            n_post += 1
        if x["early_post"]:
            n_early += 1
        if x["corrupted"]:
            n_noise += 1

        for m, p in pm.items():
            preds[m].append(p)
            ok = int(p == y)
            counts[m]["all"] += ok
            if x["post_transition"]:
                counts[m]["post"] += ok
            if x["early_post"]:
                counts[m]["early"] += ok
            if x["corrupted"]:
                counts[m]["noise"] += ok

        for s in states:
            target = 1.0 if s == y else 0.0
            raw_brier += (b_raw[s] - target) ** 2
            cal_brier += (b_cal[s] - target) ** 2
            ada_brier += (b_ada[s] - target) ** 2

    lag = {}
    miss = {}

    for m in METHODS:
        lag[m], miss[m] = transition_lag(
            truth,
            preds[m],
            pre_steps,
        )

    return {
        "steps": n_all,
        "post_steps": n_post,
        "early_steps": n_early,
        "noise_steps": n_noise,
        "counts": counts,
        "lag": lag,
        "miss": miss,
        "raw_brier_sum": raw_brier,
        "cal_brier_sum": cal_brier,
        "ada_brier_sum": ada_brier,
        "same_category": (
            row.get("start_receptacle_category")
            == row.get("goal_receptacle_category")
        ),
        "displacement_m": row["displacement_m"],
    }


def aggregate(evals):
    n_all = sum(x["steps"] for x in evals)
    n_post = sum(x["post_steps"] for x in evals)
    n_early = sum(x["early_steps"] for x in evals)
    n_noise = sum(x["noise_steps"] for x in evals)

    out = {
        "n_transitions": len(evals),
        "n_steps": n_all,
        "n_post_steps": n_post,
        "n_early_post_steps": n_early,
        "n_corrupted_steps": n_noise,
    }

    for m in METHODS:
        out[f"{m}_acc"] = (
            sum(x["counts"][m]["all"] for x in evals) / n_all
        )
        out[f"{m}_post_acc"] = (
            sum(x["counts"][m]["post"] for x in evals) / n_post
        )
        out[f"{m}_early_post_acc"] = (
            sum(x["counts"][m]["early"] for x in evals) / n_early
        )
        out[f"{m}_noise_acc"] = (
            sum(x["counts"][m]["noise"] for x in evals) / n_noise
            if n_noise else float("nan")
        )
        out[f"{m}_transition_lag"] = mean_valid(
            [x["lag"][m] for x in evals]
        )
        out[f"{m}_transition_miss_rate"] = (
            sum(int(x["miss"][m]) for x in evals) / len(evals)
        )

    out["statemem_raw_brier"] = (
        sum(x["raw_brier_sum"] for x in evals) / n_all
    )
    out["statemem_calibrated_brier"] = (
        sum(x["cal_brier_sum"] for x in evals) / n_all
    )
    out["statemem_adaptive_brier"] = (
        sum(x["ada_brier_sum"] for x in evals) / n_all
    )

    return out


def evaluate_condition(
    rows,
    calibrator,
    observation_rate,
    corruption_rate,
    confidence_mode,
    calibrated_stay,
    adaptive_stay,
    adaptive_strength,
    decay,
    pre_steps,
    post_steps,
    repeat,
    seed,
    split_name,
):
    evals = []

    for row in rows:
        s = stable_seed(
            seed,
            split_name,
            repeat,
            row["episode_id"],
            row["object_handle"],
            observation_rate,
            corruption_rate,
            confidence_mode,
        )

        seq = make_evidence_sequence(
            row=row,
            observation_rate=observation_rate,
            corruption_rate=corruption_rate,
            confidence_mode=confidence_mode,
            pre_steps=pre_steps,
            post_steps=post_steps,
            seed=s,
        )

        evals.append(
            evaluate_one(
                row=row,
                seq=seq,
                calibrator=calibrator,
                calibrated_stay=calibrated_stay,
                adaptive_stay=adaptive_stay,
                adaptive_strength=adaptive_strength,
                decay=decay,
                pre_steps=pre_steps,
            )
        )

    return aggregate(evals)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="real_object_transitions.jsonl",
    )
    ap.add_argument(
        "--out-dir",
        default="findingdory_statemem_results",
    )
    ap.add_argument(
        "--observation-rates",
        default="0.25,0.50,1.00",
    )
    ap.add_argument(
        "--corruptions",
        default="0.10,0.20,0.30",
    )
    ap.add_argument(
        "--confidence-modes",
        default="reliable,overlapping,misleading",
    )
    ap.add_argument(
        "--stay-probs",
        default="0.55,0.65,0.70,0.75,0.80,0.85,0.90,0.95",
    )
    ap.add_argument(
        "--adaptive-strengths",
        default="0.05,0.10,0.15,0.20,0.30,0.40",
    )
    ap.add_argument("--pre-steps", type=int, default=8)
    ap.add_argument("--post-steps", type=int, default=12)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--decay", type=float, default=0.25)
    ap.add_argument("--calibration-bins", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260915)

    args = ap.parse_args()

    observation_rates = parse_float_list(args.observation_rates)
    corruptions = parse_float_list(args.corruptions)
    stay_probs = parse_float_list(args.stay_probs)
    adaptive_strengths = parse_float_list(args.adaptive_strengths)
    confidence_modes = [
        x.strip()
        for x in args.confidence_modes.split(",")
        if x.strip()
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_transitions(args.input)
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "val"]

    train_scenes = {r["scene_id"] for r in train}
    test_scenes = {r["scene_id"] for r in test}
    overlap = train_scenes & test_scenes

    if overlap:
        raise SystemExit(
            f"ERROR: train/val scene leakage detected: {len(overlap)} scenes"
        )

    split = make_train_scene_split(train, args.seed)
    calibration_rows = rows_for_scenes(
        train,
        split["calibration"],
    )
    tuning_rows = rows_for_scenes(
        train,
        split["tuning"],
    )

    split_report = {
        "official_train_transitions": len(train),
        "official_val_transitions": len(test),
        "official_train_scenes": len(train_scenes),
        "official_val_scenes": len(test_scenes),
        "official_train_val_scene_overlap": 0,
        "train_calibration_scenes": len(split["calibration"]),
        "train_tuning_scenes": len(split["tuning"]),
        "train_calibration_transitions": len(calibration_rows),
        "train_tuning_transitions": len(tuning_rows),
        "same_receptacle_category_rate_train": (
            sum(
                r.get("start_receptacle_category")
                == r.get("goal_receptacle_category")
                for r in train
            ) / len(train)
        ),
        "same_receptacle_category_rate_val": (
            sum(
                r.get("start_receptacle_category")
                == r.get("goal_receptacle_category")
                for r in test
            ) / len(test)
        ),
        "calibration_scene_ids": split["calibration"],
        "tuning_scene_ids": split["tuning"],
    }

    with (out_dir / "split_report.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(split_report, f, indent=2)

    print("=" * 94)
    print("FINDINGDORY NATIVE-TRANSITION STATEMEM")
    print("=" * 94)
    print(
        f"Official train: {len(train)} transitions | "
        f"{len(train_scenes)} scenes"
    )
    print(
        f"Official val:   {len(test)} transitions | "
        f"{len(test_scenes)} scenes"
    )
    print("Train/val scene overlap: 0")
    print(
        f"Train calibration: {len(calibration_rows)} transitions | "
        f"{len(split['calibration'])} scenes"
    )
    print(
        f"Train tuning:      {len(tuning_rows)} transitions | "
        f"{len(split['tuning'])} scenes"
    )
    print()

    print("Fitting ONE pooled confidence calibrator on train-calibration scenes...")

    calibrator = fit_pooled_calibrator(
        rows=calibration_rows,
        observation_rates=observation_rates,
        corruptions=corruptions,
        confidence_modes=confidence_modes,
        pre_steps=args.pre_steps,
        post_steps=args.post_steps,
        bins=args.calibration_bins,
        seed=args.seed,
    )

    with (out_dir / "pooled_calibration.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(calibrator.table(), f, indent=2)

    # ---------------------------------------------------------------
    # Tune calibrated fixed persistence on train-tuning scenes.
    # ---------------------------------------------------------------
    print("Tuning calibrated StateMem stay probability...")

    stay_rows = []

    for stay in stay_probs:
        metrics = []

        for mode in confidence_modes:
            for obsr in observation_rates:
                for cr in corruptions:
                    m = evaluate_condition(
                        rows=tuning_rows,
                        calibrator=calibrator,
                        observation_rate=obsr,
                        corruption_rate=cr,
                        confidence_mode=mode,
                        calibrated_stay=stay,
                        adaptive_stay=stay,
                        adaptive_strength=0.0,
                        decay=args.decay,
                        pre_steps=args.pre_steps,
                        post_steps=args.post_steps,
                        repeat=0,
                        seed=args.seed,
                        split_name="tuning-fixed",
                    )
                    metrics.append(m)

        row = {
            "stay_probability": stay,
            "mean_accuracy": mean_valid(
                [x["statemem_calibrated_acc"] for x in metrics]
            ),
            "mean_post_accuracy": mean_valid(
                [x["statemem_calibrated_post_acc"] for x in metrics]
            ),
            "mean_early_post_accuracy": mean_valid(
                [x["statemem_calibrated_early_post_acc"] for x in metrics]
            ),
            "mean_transition_lag": mean_valid(
                [x["statemem_calibrated_transition_lag"] for x in metrics]
            ),
            "mean_brier": mean_valid(
                [x["statemem_calibrated_brier"] for x in metrics]
            ),
        }
        stay_rows.append(row)

        print(
            f"  stay={stay:.2f} | "
            f"acc={100*row['mean_accuracy']:.2f}% | "
            f"post={100*row['mean_post_accuracy']:.2f}% | "
            f"early={100*row['mean_early_post_accuracy']:.2f}% | "
            f"lag={row['mean_transition_lag']:.3f} | "
            f"Brier={row['mean_brier']:.4f}"
        )

    save_csv(
        out_dir / "validation_stay_sweep.csv",
        stay_rows,
    )

    best_fixed = max(
        stay_rows,
        key=lambda x: (
            x["mean_post_accuracy"],
            x["mean_early_post_accuracy"],
            -x["mean_brier"],
        ),
    )
    calibrated_stay = best_fixed["stay_probability"]

    print()
    print(
        f"Selected fixed calibrated stay={calibrated_stay:.2f}"
    )

    # ---------------------------------------------------------------
    # Tune adaptive persistence separately.
    # ---------------------------------------------------------------
    print("Tuning adaptive persistence...")

    adaptive_rows = []

    for stay in stay_probs:
        for strength in adaptive_strengths:
            metrics = []

            for mode in confidence_modes:
                for obsr in observation_rates:
                    for cr in corruptions:
                        m = evaluate_condition(
                            rows=tuning_rows,
                            calibrator=calibrator,
                            observation_rate=obsr,
                            corruption_rate=cr,
                            confidence_mode=mode,
                            calibrated_stay=calibrated_stay,
                            adaptive_stay=stay,
                            adaptive_strength=strength,
                            decay=args.decay,
                            pre_steps=args.pre_steps,
                            post_steps=args.post_steps,
                            repeat=0,
                            seed=args.seed,
                            split_name="tuning-adaptive",
                        )
                        metrics.append(m)

            row = {
                "stay_probability": stay,
                "adaptive_strength": strength,
                "mean_accuracy": mean_valid(
                    [x["statemem_adaptive_acc"] for x in metrics]
                ),
                "mean_post_accuracy": mean_valid(
                    [x["statemem_adaptive_post_acc"] for x in metrics]
                ),
                "mean_early_post_accuracy": mean_valid(
                    [x["statemem_adaptive_early_post_acc"] for x in metrics]
                ),
                "mean_transition_lag": mean_valid(
                    [x["statemem_adaptive_transition_lag"] for x in metrics]
                ),
                "mean_brier": mean_valid(
                    [x["statemem_adaptive_brier"] for x in metrics]
                ),
            }
            adaptive_rows.append(row)

    save_csv(
        out_dir / "validation_adaptive_sweep.csv",
        adaptive_rows,
    )

    best_adaptive = max(
        adaptive_rows,
        key=lambda x: (
            x["mean_post_accuracy"],
            x["mean_early_post_accuracy"],
            -x["mean_brier"],
        ),
    )

    adaptive_stay = best_adaptive["stay_probability"]
    adaptive_strength = best_adaptive["adaptive_strength"]

    print(
        f"Selected adaptive stay={adaptive_stay:.2f}, "
        f"strength={adaptive_strength:.2f}"
    )
    print()

    # ---------------------------------------------------------------
    # FINAL official FindingDory VAL test.
    # ---------------------------------------------------------------
    print("Running untouched official FindingDory VAL transitions...")

    test_rows = []

    total_runs = (
        len(confidence_modes)
        * len(observation_rates)
        * len(corruptions)
        * args.repeats
    )
    done = 0

    for mode in confidence_modes:
        for obsr in observation_rates:
            for cr in corruptions:
                for rep in range(args.repeats):
                    m = evaluate_condition(
                        rows=test,
                        calibrator=calibrator,
                        observation_rate=obsr,
                        corruption_rate=cr,
                        confidence_mode=mode,
                        calibrated_stay=calibrated_stay,
                        adaptive_stay=adaptive_stay,
                        adaptive_strength=adaptive_strength,
                        decay=args.decay,
                        pre_steps=args.pre_steps,
                        post_steps=args.post_steps,
                        repeat=rep,
                        seed=args.seed,
                        split_name="official-val",
                    )

                    test_rows.append({
                        "confidence_mode": mode,
                        "observation_rate": obsr,
                        "corruption_rate": cr,
                        "repeat": rep,
                        "calibrated_stay": calibrated_stay,
                        "adaptive_stay": adaptive_stay,
                        "adaptive_strength": adaptive_strength,
                        **m,
                    })

                    done += 1
                    if done % 10 == 0 or done == total_runs:
                        print(f"  completed {done}/{total_runs}")

    save_csv(
        out_dir / "test_results.csv",
        test_rows,
    )

    labels = [
        ("Latest", "latest"),
        ("Recency", "recency"),
        ("Recency x RawConf", "recency_rawconf"),
        ("StateMem-Raw", "statemem_raw"),
        ("StateMem-Calibrated", "statemem_calibrated"),
        ("StateMem-Adaptive", "statemem_adaptive"),
    ]

    overall = {
        "selected_fixed_stay": calibrated_stay,
        "selected_adaptive_stay": adaptive_stay,
        "selected_adaptive_strength": adaptive_strength,
        "num_official_val_transitions": len(test),
        "methods": {},
    }

    print()
    print("=" * 126)
    print("OFFICIAL FINDINGDORY VAL — NATIVE OBJECT-TRANSITION DIAGNOSTIC")
    print("=" * 126)
    print(
        f"{'Method':<24} | "
        f"{'All Acc':>8} | "
        f"{'Post':>8} | "
        f"{'Early Post':>10} | "
        f"{'Noise':>8} | "
        f"{'Lag':>7} | "
        f"{'Brier':>8}"
    )
    print("-" * 126)

    for label, key in labels:
        entry = {
            "accuracy": mean_valid(
                [r[f"{key}_acc"] for r in test_rows]
            ),
            "post_accuracy": mean_valid(
                [r[f"{key}_post_acc"] for r in test_rows]
            ),
            "early_post_accuracy": mean_valid(
                [r[f"{key}_early_post_acc"] for r in test_rows]
            ),
            "noise_accuracy": mean_valid(
                [r[f"{key}_noise_acc"] for r in test_rows]
            ),
            "transition_lag": mean_valid(
                [r[f"{key}_transition_lag"] for r in test_rows]
            ),
            "transition_miss_rate": mean_valid(
                [r[f"{key}_transition_miss_rate"] for r in test_rows]
            ),
        }

        bk = f"{key}_brier"
        if bk in test_rows[0]:
            entry["brier"] = mean_valid(
                [r[bk] for r in test_rows]
            )
        else:
            entry["brier"] = None

        overall["methods"][label] = entry

        bt = (
            f"{entry['brier']:.4f}"
            if entry["brier"] is not None
            else "-"
        )

        print(
            f"{label:<24} | "
            f"{100*entry['accuracy']:7.2f}% | "
            f"{100*entry['post_accuracy']:7.2f}% | "
            f"{100*entry['early_post_accuracy']:9.2f}% | "
            f"{100*entry['noise_accuracy']:7.2f}% | "
            f"{entry['transition_lag']:7.3f} | "
            f"{bt:>8}"
        )

    print("=" * 126)

    with (out_dir / "test_overall.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(overall, f, indent=2)

    # ---------------------------------------------------------------
    # Native FindingDory slices:
    # same receptacle category vs changed category;
    # displacement quartiles.
    # ---------------------------------------------------------------
    ds = sorted(r["displacement_m"] for r in test)

    def q(p):
        if not ds:
            return 0.0
        i = int(round((len(ds) - 1) * p))
        return ds[i]

    q25, q50, q75 = q(0.25), q(0.50), q(0.75)

    slices = {
        "displacement_quartile_thresholds_m": {
            "q25": q25,
            "q50": q50,
            "q75": q75,
        },
        "transition_counts": {
            "same_receptacle_category": sum(
                r["start_receptacle_category"]
                == r["goal_receptacle_category"]
                for r in test
            ),
            "changed_receptacle_category": sum(
                r["start_receptacle_category"]
                != r["goal_receptacle_category"]
                for r in test
            ),
            "Q1_displacement": sum(
                r["displacement_m"] <= q25 for r in test
            ),
            "Q2_displacement": sum(
                q25 < r["displacement_m"] <= q50 for r in test
            ),
            "Q3_displacement": sum(
                q50 < r["displacement_m"] <= q75 for r in test
            ),
            "Q4_displacement": sum(
                r["displacement_m"] > q75 for r in test
            ),
        },
    }

    with (out_dir / "test_slices.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(slices, f, indent=2)

    config = {
        "input": args.input,
        "native_transition_source": (
            "FindingDory/Habitat rigid-object initial pose + target goal pose "
            "and start/goal receptacle metadata"
        ),
        "controlled_components": [
            "transition timing within the diagnostic sequence",
            "observation availability",
            "observation corruption",
            "raw confidence",
        ],
        "official_train_transitions": len(train),
        "official_val_transitions": len(test),
        "official_train_scenes": len(train_scenes),
        "official_val_scenes": len(test_scenes),
        "train_val_scene_overlap": 0,
        "observation_rates": observation_rates,
        "corruption_rates": corruptions,
        "confidence_modes": confidence_modes,
        "candidate_stay_probabilities": stay_probs,
        "candidate_adaptive_strengths": adaptive_strengths,
        "selected_fixed_stay": calibrated_stay,
        "selected_adaptive_stay": adaptive_stay,
        "selected_adaptive_strength": adaptive_strength,
        "pre_steps": args.pre_steps,
        "post_steps": args.post_steps,
        "repeats": args.repeats,
        "decay": args.decay,
        "seed": args.seed,
        "important_limitation": (
            "The object relocation endpoints are native FindingDory ground truth, "
            "but this diagnostic does not yet use RGB/VLM perception or the exact "
            "runtime interaction timestep. Those are the next integration step."
        ),
    }

    with (out_dir / "experiment_config.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(config, f, indent=2)

    print()
    print("Saved:")
    for name in [
        "split_report.json",
        "pooled_calibration.json",
        "validation_stay_sweep.csv",
        "validation_adaptive_sweep.csv",
        "test_results.csv",
        "test_overall.json",
        "test_slices.json",
        "experiment_config.json",
    ]:
        print(f"  {out_dir / name}")


if __name__ == "__main__":
    main()
