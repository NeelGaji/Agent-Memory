#!/usr/bin/env python3
"""
LMEE-State controlled temporal-belief experiment
================================================

This is the first StateMem experiment that uses the FULL released LMEE test
trajectory timelines as the temporal scaffold.

What is real from LMEE:
- scene/task split
- episode lengths
- simulator step sequence
- trial/action structure
- target metadata

What is controlled/synthetic:
- latent 3-state world variable
- when that state changes
- whether the state is observable at a timestep
- observation corruption
- perception confidence

This distinction is intentional. LMEE does not natively provide dynamic
per-timestep object-state ground truth, so this script does NOT claim otherwise.

Protocol:
1. Collapse LMEE's duplicate terminal move/stop records into unique sim steps.
2. Split by SCENE, never by task:
      calibration scenes -> confidence calibration only
      validation scenes  -> select ONE global StateMem stay probability
      test scenes        -> final evaluation only
3. Fit confidence calibration on calibration scenes.
4. Tune stay probability on validation scenes.
5. Freeze everything and report independent test results.

Methods:
- Latest
- Recency
- Recency x Raw Confidence
- StateMem-Raw
- StateMem-Calibrated

Metrics:
- all-step current-state accuracy
- observed-step accuracy
- unobserved/stale-step accuracy
- corrupted-observation recovery accuracy
- transition detection lag
- Brier score for StateMem variants

No third-party packages required.

Run:
    cd ~/Downloads
    python3 lmee_state_experiment.py \
      --input lmee_full_statemem/lmee_trajectory_skeleton_full.jsonl

Fast smoke test:
    python3 lmee_state_experiment.py \
      --input lmee_full_statemem/lmee_trajectory_skeleton_full.jsonl \
      --transition-rates 0.05 \
      --observation-rates 0.5 \
      --corruptions 0.2 \
      --confidence-modes reliable,misleading \
      --repeats 1

Outputs:
    lmee_state_results/
      scene_split.json
      calibration_tables.json
      validation_stay_sweep.csv
      test_results.csv
      test_overall.json
      experiment_config.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

STATES = ["state_A", "state_B", "state_C"]
METHODS = [
    "latest",
    "recency",
    "recency_rawconf",
    "statemem_raw",
    "statemem_calibrated",
]


# ---------------------------------------------------------------------
# Loading real LMEE trajectories
# ---------------------------------------------------------------------

def load_lmee_timelines(path: Path):
    """
    Load rows and collapse duplicate LMEE terminal records that share sim_step.
    At duplicated steps, prefer the STOP record because that is the actual
    checkpoint observation.
    """
    by_task_step = {}
    task_meta = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)
            task_id = row["task_id"]
            sim_step = row.get("sim_step")

            if sim_step is None:
                continue

            task_meta[task_id] = {
                "scene": row.get("scene"),
                "difficulty": row.get("difficulty"),
            }

            key = (task_id, sim_step)

            if key not in by_task_step:
                by_task_step[key] = row
            else:
                # LMEE has one duplicate at subtask terminal steps:
                # e.g. 93_move_forward and 93_stop. Prefer STOP.
                if row.get("is_stop") and not by_task_step[key].get("is_stop"):
                    by_task_step[key] = row

    timelines = defaultdict(list)

    for (task_id, sim_step), row in by_task_step.items():
        timelines[task_id].append({
            "sim_step": sim_step,
            "action": row.get("action"),
            "trial_name": row.get("trial_name"),
            "subtask_index": row.get("subtask_index"),
            "target_object_id": row.get("target_object_id"),
            "target_object_class": row.get("target_object_class"),
            "position": row.get("position"),
            "qa_checkpoint": row.get("qa_checkpoint", False),
        })

    for task_id in timelines:
        timelines[task_id].sort(key=lambda x: x["sim_step"])

    # Sanity check monotonicity
    bad = []
    for task_id, seq in timelines.items():
        steps = [x["sim_step"] for x in seq]
        if any(b <= a for a, b in zip(steps, steps[1:])):
            bad.append(task_id)

    if bad:
        raise ValueError(
            f"{len(bad)} tasks remain non-monotonic after step collapsing."
        )

    return dict(timelines), task_meta


# ---------------------------------------------------------------------
# Deterministic random seeds
# ---------------------------------------------------------------------

def stable_seed(base_seed, *parts):
    text = "|".join([str(base_seed)] + [str(x) for x in parts])
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


# ---------------------------------------------------------------------
# Scene split
# ---------------------------------------------------------------------

def make_scene_split(timelines, task_meta, seed):
    scenes = sorted({
        task_meta[t]["scene"]
        for t in timelines
        if task_meta[t].get("scene") is not None
    })

    rng = random.Random(seed)
    rng.shuffle(scenes)

    n = len(scenes)
    n_cal = max(1, round(0.20 * n))
    n_val = max(1, round(0.20 * n))

    # Ensure at least one test scene.
    if n_cal + n_val >= n:
        n_val = max(1, n - n_cal - 1)

    calibration = set(scenes[:n_cal])
    validation = set(scenes[n_cal:n_cal+n_val])
    test = set(scenes[n_cal+n_val:])

    split = {
        "calibration": sorted(calibration),
        "validation": sorted(validation),
        "test": sorted(test),
    }

    return split


def tasks_for_scenes(timelines, task_meta, scenes):
    scenes = set(scenes)
    return {
        task_id: seq
        for task_id, seq in timelines.items()
        if task_meta[task_id]["scene"] in scenes
    }


def split_stats(tasks, task_meta):
    diffs = Counter()
    scenes = Counter()
    steps = 0

    for task_id, seq in tasks.items():
        diffs[str(task_meta[task_id].get("difficulty"))] += 1
        scenes[str(task_meta[task_id].get("scene"))] += 1
        steps += len(seq)

    return {
        "tasks": len(tasks),
        "unique_scenes": len(scenes),
        "timesteps": steps,
        "difficulty_task_counts": dict(diffs),
    }


# ---------------------------------------------------------------------
# Controlled dynamic-state overlay
# ---------------------------------------------------------------------

@dataclass
class OverlayStep:
    true_state: str
    observed: bool
    observed_state: str | None
    raw_confidence: float | None
    corrupted: bool


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
        # Intentionally adversarial: wrong observations may look more confident.
        return (
            rng.uniform(0.45, 0.85)
            if correct
            else rng.uniform(0.55, 0.95)
        )

    raise ValueError(f"Unknown confidence mode: {mode}")


def make_overlay(
    timeline,
    transition_rate,
    observation_rate,
    corruption_rate,
    confidence_mode,
    seed,
):
    rng = random.Random(seed)
    current = rng.choice(STATES)
    out = []

    for i, _ in enumerate(timeline):
        if i > 0 and rng.random() < transition_rate:
            current = rng.choice([s for s in STATES if s != current])

        # Always expose the first state so methods are initialized fairly.
        observed = (i == 0) or (rng.random() < observation_rate)

        if observed:
            corrupted = rng.random() < corruption_rate

            if corrupted:
                obs_state = rng.choice([s for s in STATES if s != current])
            else:
                obs_state = current

            conf = sample_confidence(
                correct=not corrupted,
                mode=confidence_mode,
                rng=rng,
            )
        else:
            corrupted = False
            obs_state = None
            conf = None

        out.append(
            OverlayStep(
                true_state=current,
                observed=observed,
                observed_state=obs_state,
                raw_confidence=conf,
                corrupted=corrupted,
            )
        )

    return out


# ---------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------

class HistogramCalibrator:
    def __init__(self, bins=10, alpha=2.0, beta=2.0):
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

    def fit(self, confidence, correct):
        i = self._bin(confidence)
        self.total[i] += 1
        self.correct[i] += int(correct)
        self.global_total += 1
        self.global_correct += int(correct)

    def predict(self, confidence):
        i = self._bin(confidence)

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


def fit_calibrators(
    calibration_tasks,
    transition_rates,
    corruptions,
    confidence_modes,
    bins,
    seed,
):
    """
    One calibrator per simulated perception regime.

    We pool several corruption and transition settings on CALIBRATION scenes.
    The mapping is therefore not fit on validation or test scenes.
    """
    calibrators = {
        mode: HistogramCalibrator(bins=bins)
        for mode in confidence_modes
    }

    representative_obs_rate = 1.0

    for mode in confidence_modes:
        cal = calibrators[mode]

        for transition_rate in transition_rates:
            for corruption_rate in corruptions:
                for task_id, timeline in calibration_tasks.items():
                    s = stable_seed(
                        seed,
                        "calibration",
                        mode,
                        transition_rate,
                        corruption_rate,
                        task_id,
                    )

                    overlay = make_overlay(
                        timeline=timeline,
                        transition_rate=transition_rate,
                        observation_rate=representative_obs_rate,
                        corruption_rate=corruption_rate,
                        confidence_mode=mode,
                        seed=s,
                    )

                    for x in overlay:
                        if x.observed:
                            cal.fit(
                                x.raw_confidence,
                                x.observed_state == x.true_state,
                            )

    return calibrators


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------

class RecencyTracker:
    def __init__(self, decay, use_confidence):
        self.factor = math.exp(-decay)
        self.use_confidence = use_confidence
        self.scores = {s: 0.0 for s in STATES}

    def step(self, observed_state=None, confidence=None):
        for s in STATES:
            self.scores[s] *= self.factor

        if observed_state is not None:
            w = confidence if self.use_confidence else 1.0
            self.scores[observed_state] += w

        return max(self.scores, key=self.scores.get)


class StateMem:
    def __init__(self, stay_probability, calibrator=None, use_raw=False):
        self.stay_probability = stay_probability
        self.calibrator = calibrator
        self.use_raw = use_raw
        self.belief = {s: 1.0 / len(STATES) for s in STATES}

    def prior_step(self):
        n = len(STATES)
        change_each = (1.0 - self.stay_probability) / (n - 1)

        prior = {s: 0.0 for s in STATES}

        for prev, pp in self.belief.items():
            for nxt in STATES:
                trans = (
                    self.stay_probability
                    if nxt == prev
                    else change_each
                )
                prior[nxt] += pp * trans

        return prior

    def step(self, observed_state=None, raw_confidence=None):
        prior = self.prior_step()

        # No new evidence: carry the predicted distribution forward.
        if observed_state is None:
            self.belief = prior
            return dict(self.belief)

        if self.use_raw:
            reliability = raw_confidence
        else:
            reliability = self.calibrator.predict(raw_confidence)

        r = max(0.01, min(0.99, float(reliability)))
        n = len(STATES)

        likelihood = {
            s: (
                r
                if s == observed_state
                else (1.0 - r) / (n - 1)
            )
            for s in STATES
        }

        posterior = {
            s: prior[s] * likelihood[s]
            for s in STATES
        }

        z = sum(posterior.values())
        self.belief = {
            s: posterior[s] / z
            for s in STATES
        }

        return dict(self.belief)

    def predict(self):
        return max(self.belief, key=self.belief.get)


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def transition_lag(truths, predictions):
    change_indices = [
        i
        for i in range(1, len(truths))
        if truths[i] != truths[i-1]
    ]

    lags = []
    misses = 0

    for k, start in enumerate(change_indices):
        target = truths[start]
        end = (
            change_indices[k+1]
            if k+1 < len(change_indices)
            else len(truths)
        )

        for j in range(start, end):
            if predictions[j] == target:
                lags.append(j - start)
                break
        else:
            misses += 1

    return (
        sum(lags) / len(lags) if lags else float("nan"),
        len(lags),
        misses,
    )


def evaluate_task(
    timeline,
    overlay,
    stay_probability,
    calibrator,
    decay,
):
    latest = None
    recency = RecencyTracker(decay=decay, use_confidence=False)
    recconf = RecencyTracker(decay=decay, use_confidence=True)
    raw = StateMem(
        stay_probability=stay_probability,
        use_raw=True,
    )
    calibrated = StateMem(
        stay_probability=stay_probability,
        calibrator=calibrator,
        use_raw=False,
    )

    preds = {m: [] for m in METHODS}
    truths = []

    counts = {
        m: {
            "correct": 0,
            "observed_correct": 0,
            "stale_correct": 0,
            "noise_correct": 0,
        }
        for m in METHODS
    }

    observed_n = 0
    stale_n = 0
    noise_n = 0

    brier_raw = 0.0
    brier_cal = 0.0

    for lm, x in zip(timeline, overlay):
        if x.observed:
            latest = x.observed_state
            rec_pred = recency.step(
                x.observed_state,
                x.raw_confidence,
            )
            rc_pred = recconf.step(
                x.observed_state,
                x.raw_confidence,
            )
            br = raw.step(
                x.observed_state,
                x.raw_confidence,
            )
            bc = calibrated.step(
                x.observed_state,
                x.raw_confidence,
            )
        else:
            rec_pred = recency.step()
            rc_pred = recconf.step()
            br = raw.step()
            bc = calibrated.step()

        pred_map = {
            "latest": latest,
            "recency": rec_pred,
            "recency_rawconf": rc_pred,
            "statemem_raw": raw.predict(),
            "statemem_calibrated": calibrated.predict(),
        }

        truth = x.true_state
        truths.append(truth)

        if x.observed:
            observed_n += 1
        else:
            stale_n += 1

        if x.corrupted:
            noise_n += 1

        for m, pred in pred_map.items():
            preds[m].append(pred)
            ok = int(pred == truth)
            counts[m]["correct"] += ok

            if x.observed:
                counts[m]["observed_correct"] += ok
            else:
                counts[m]["stale_correct"] += ok

            if x.corrupted:
                counts[m]["noise_correct"] += ok

        for s in STATES:
            y = 1.0 if s == truth else 0.0
            brier_raw += (br[s] - y) ** 2
            brier_cal += (bc[s] - y) ** 2

    lag = {}

    for m in METHODS:
        lag[m] = transition_lag(truths, preds[m])

    return {
        "steps": len(timeline),
        "observed_steps": observed_n,
        "stale_steps": stale_n,
        "noise_steps": noise_n,
        "counts": counts,
        "lag": lag,
        "brier_raw_sum": brier_raw,
        "brier_cal_sum": brier_cal,
    }


def aggregate_evaluations(evals):
    total_steps = sum(x["steps"] for x in evals)
    observed_steps = sum(x["observed_steps"] for x in evals)
    stale_steps = sum(x["stale_steps"] for x in evals)
    noise_steps = sum(x["noise_steps"] for x in evals)

    out = {
        "n_steps": total_steps,
        "n_observed_steps": observed_steps,
        "n_stale_steps": stale_steps,
        "n_corrupted_observed_steps": noise_steps,
    }

    for m in METHODS:
        correct = sum(x["counts"][m]["correct"] for x in evals)
        obs_correct = sum(
            x["counts"][m]["observed_correct"]
            for x in evals
        )
        stale_correct = sum(
            x["counts"][m]["stale_correct"]
            for x in evals
        )
        noise_correct = sum(
            x["counts"][m]["noise_correct"]
            for x in evals
        )

        out[f"{m}_acc"] = correct / total_steps
        out[f"{m}_observed_acc"] = (
            obs_correct / observed_steps
            if observed_steps
            else float("nan")
        )
        out[f"{m}_stale_acc"] = (
            stale_correct / stale_steps
            if stale_steps
            else float("nan")
        )
        out[f"{m}_noise_acc"] = (
            noise_correct / noise_steps
            if noise_steps
            else float("nan")
        )

        lag_weighted = 0.0
        lag_success = 0
        lag_miss = 0

        for x in evals:
            mean_lag, successes, misses = x["lag"][m]
            if successes and not math.isnan(mean_lag):
                lag_weighted += mean_lag * successes
                lag_success += successes
            lag_miss += misses

        out[f"{m}_transition_lag"] = (
            lag_weighted / lag_success
            if lag_success
            else float("nan")
        )

        denom = lag_success + lag_miss
        out[f"{m}_transition_miss_rate"] = (
            lag_miss / denom
            if denom
            else 0.0
        )

    out["statemem_raw_brier"] = (
        sum(x["brier_raw_sum"] for x in evals)
        / total_steps
    )

    out["statemem_calibrated_brier"] = (
        sum(x["brier_cal_sum"] for x in evals)
        / total_steps
    )

    return out


# ---------------------------------------------------------------------
# Running conditions
# ---------------------------------------------------------------------

def evaluate_condition(
    tasks,
    calibrator,
    transition_rate,
    observation_rate,
    corruption_rate,
    confidence_mode,
    stay_probability,
    decay,
    repeat,
    seed,
    split_name,
):
    evals = []

    for task_id, timeline in tasks.items():
        s = stable_seed(
            seed,
            split_name,
            repeat,
            task_id,
            transition_rate,
            observation_rate,
            corruption_rate,
            confidence_mode,
        )

        overlay = make_overlay(
            timeline=timeline,
            transition_rate=transition_rate,
            observation_rate=observation_rate,
            corruption_rate=corruption_rate,
            confidence_mode=confidence_mode,
            seed=s,
        )

        evals.append(
            evaluate_task(
                timeline=timeline,
                overlay=overlay,
                stay_probability=stay_probability,
                calibrator=calibrator,
                decay=decay,
            )
        )

    return aggregate_evaluations(evals)


def save_csv(path, rows):
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        w.writeheader()
        w.writerows(rows)


def mean_valid(values):
    vals = [
        x for x in values
        if not (isinstance(x, float) and math.isnan(x))
    ]
    return sum(vals) / len(vals) if vals else float("nan")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_float_list(text):
    return [
        float(x.strip())
        for x in text.split(",")
        if x.strip()
    ]


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input",
        required=True,
        help="Path to lmee_trajectory_skeleton_full.jsonl",
    )
    ap.add_argument(
        "--out-dir",
        default="lmee_state_results",
    )
    ap.add_argument(
        "--transition-rates",
        default="0.01,0.05,0.10",
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
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--decay", type=float, default=0.25)
    ap.add_argument("--calibration-bins", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260915)

    args = ap.parse_args()

    transition_rates = parse_float_list(args.transition_rates)
    observation_rates = parse_float_list(args.observation_rates)
    corruptions = parse_float_list(args.corruptions)
    stay_probs = parse_float_list(args.stay_probs)
    confidence_modes = [
        x.strip()
        for x in args.confidence_modes.split(",")
        if x.strip()
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and collapsing LMEE trajectories...")
    timelines, task_meta = load_lmee_timelines(
        Path(args.input)
    )

    raw_rows = sum(
        1
        for _ in Path(args.input).open("r", encoding="utf-8")
        if _.strip()
    )
    unique_steps = sum(len(x) for x in timelines.values())

    print(f"Tasks:             {len(timelines)}")
    print(f"Raw LMEE events:   {raw_rows}")
    print(f"Unique sim steps:  {unique_steps}")
    print(f"Collapsed records: {raw_rows - unique_steps}")
    print()

    scene_split = make_scene_split(
        timelines=timelines,
        task_meta=task_meta,
        seed=args.seed,
    )

    calibration_tasks = tasks_for_scenes(
        timelines,
        task_meta,
        scene_split["calibration"],
    )
    validation_tasks = tasks_for_scenes(
        timelines,
        task_meta,
        scene_split["validation"],
    )
    test_tasks = tasks_for_scenes(
        timelines,
        task_meta,
        scene_split["test"],
    )

    split_report = {
        "seed": args.seed,
        "scenes": scene_split,
        "stats": {
            "calibration": split_stats(
                calibration_tasks,
                task_meta,
            ),
            "validation": split_stats(
                validation_tasks,
                task_meta,
            ),
            "test": split_stats(
                test_tasks,
                task_meta,
            ),
        },
    }

    with (out_dir / "scene_split.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(split_report, f, indent=2)

    print("Scene-disjoint split:")
    for name in ["calibration", "validation", "test"]:
        st = split_report["stats"][name]
        print(
            f"  {name:<11}: "
            f"{st['unique_scenes']:2d} scenes | "
            f"{st['tasks']:3d} tasks | "
            f"{st['timesteps']:5d} steps"
        )

    print()
    print("Fitting confidence calibration on CALIBRATION scenes only...")

    calibrators = fit_calibrators(
        calibration_tasks=calibration_tasks,
        transition_rates=transition_rates,
        corruptions=corruptions,
        confidence_modes=confidence_modes,
        bins=args.calibration_bins,
        seed=args.seed,
    )

    cal_json = {
        mode: cal.table()
        for mode, cal in calibrators.items()
    }

    with (out_dir / "calibration_tables.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(cal_json, f, indent=2)

    print("Tuning ONE global stay_probability on VALIDATION scenes...")

    sweep_rows = []

    for stay in stay_probs:
        condition_metrics = []

        for mode in confidence_modes:
            for tr in transition_rates:
                for obsr in observation_rates:
                    for cr in corruptions:
                        # One validation repeat per condition is enough for
                        # selection; all final test repeats remain untouched.
                        metrics = evaluate_condition(
                            tasks=validation_tasks,
                            calibrator=calibrators[mode],
                            transition_rate=tr,
                            observation_rate=obsr,
                            corruption_rate=cr,
                            confidence_mode=mode,
                            stay_probability=stay,
                            decay=args.decay,
                            repeat=0,
                            seed=args.seed,
                            split_name="validation",
                        )
                        condition_metrics.append(metrics)

        mean_acc = mean_valid([
            x["statemem_calibrated_acc"]
            for x in condition_metrics
        ])
        mean_brier = mean_valid([
            x["statemem_calibrated_brier"]
            for x in condition_metrics
        ])
        mean_lag = mean_valid([
            x["statemem_calibrated_transition_lag"]
            for x in condition_metrics
        ])

        sweep_rows.append({
            "stay_probability": stay,
            "mean_calibrated_accuracy": mean_acc,
            "mean_calibrated_brier": mean_brier,
            "mean_transition_lag": mean_lag,
        })

        print(
            f"  stay={stay:.2f} | "
            f"acc={100*mean_acc:.2f}% | "
            f"Brier={mean_brier:.4f} | "
            f"lag={mean_lag:.3f}"
        )

    save_csv(
        out_dir / "validation_stay_sweep.csv",
        sweep_rows,
    )

    best = max(
        sweep_rows,
        key=lambda r: (
            r["mean_calibrated_accuracy"],
            -r["mean_calibrated_brier"],
        ),
    )
    chosen_stay = best["stay_probability"]

    print()
    print(
        f"Frozen stay_probability for TEST = {chosen_stay:.2f}"
    )
    print()

    print("Running independent TEST scenes...")

    test_rows = []
    total_conditions = (
        len(confidence_modes)
        * len(transition_rates)
        * len(observation_rates)
        * len(corruptions)
        * args.repeats
    )
    done = 0

    for mode in confidence_modes:
        for tr in transition_rates:
            for obsr in observation_rates:
                for cr in corruptions:
                    for repeat in range(args.repeats):
                        metrics = evaluate_condition(
                            tasks=test_tasks,
                            calibrator=calibrators[mode],
                            transition_rate=tr,
                            observation_rate=obsr,
                            corruption_rate=cr,
                            confidence_mode=mode,
                            stay_probability=chosen_stay,
                            decay=args.decay,
                            repeat=repeat,
                            seed=args.seed,
                            split_name="test",
                        )

                        test_rows.append({
                            "confidence_mode": mode,
                            "transition_rate": tr,
                            "observation_rate": obsr,
                            "corruption_rate": cr,
                            "repeat": repeat,
                            "stay_probability": chosen_stay,
                            **metrics,
                        })

                        done += 1
                        if done % 25 == 0 or done == total_conditions:
                            print(
                                f"  completed {done}/{total_conditions}"
                            )

    save_csv(
        out_dir / "test_results.csv",
        test_rows,
    )

    overall = {
        "selected_stay_probability": chosen_stay,
        "num_test_condition_runs": len(test_rows),
        "methods": {},
    }

    labels = [
        ("Latest", "latest"),
        ("Recency", "recency"),
        ("Recency x RawConf", "recency_rawconf"),
        ("StateMem-Raw", "statemem_raw"),
        ("StateMem-Calibrated", "statemem_calibrated"),
    ]

    print()
    print("=" * 118)
    print("LMEE-CONDITIONED DYNAMIC-STATE TEST RESULTS")
    print("=" * 118)
    print(
        f"{'Method':<24} | "
        f"{'All Acc':>8} | "
        f"{'Observed':>8} | "
        f"{'Stale':>8} | "
        f"{'Noise':>8} | "
        f"{'Trans Lag':>9} | "
        f"{'Brier':>8}"
    )
    print("-" * 118)

    for label, key in labels:
        entry = {
            "accuracy": mean_valid([
                r[f"{key}_acc"]
                for r in test_rows
            ]),
            "observed_accuracy": mean_valid([
                r[f"{key}_observed_acc"]
                for r in test_rows
            ]),
            "stale_accuracy": mean_valid([
                r[f"{key}_stale_acc"]
                for r in test_rows
            ]),
            "noise_accuracy": mean_valid([
                r[f"{key}_noise_acc"]
                for r in test_rows
            ]),
            "transition_lag": mean_valid([
                r[f"{key}_transition_lag"]
                for r in test_rows
            ]),
        }

        brier_key = f"{key}_brier"

        if brier_key in test_rows[0]:
            entry["brier"] = mean_valid([
                r[brier_key]
                for r in test_rows
            ])
        else:
            entry["brier"] = None

        overall["methods"][label] = entry

        brier_text = (
            f"{entry['brier']:.4f}"
            if entry["brier"] is not None
            else "-"
        )

        print(
            f"{label:<24} | "
            f"{100*entry['accuracy']:7.2f}% | "
            f"{100*entry['observed_accuracy']:7.2f}% | "
            f"{100*entry['stale_accuracy']:7.2f}% | "
            f"{100*entry['noise_accuracy']:7.2f}% | "
            f"{entry['transition_lag']:9.3f} | "
            f"{brier_text:>8}"
        )

    print("=" * 118)

    with (out_dir / "test_overall.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(overall, f, indent=2)

    config = {
        "input": str(args.input),
        "real_lmee_tasks": len(timelines),
        "real_lmee_raw_events": raw_rows,
        "real_lmee_unique_sim_steps": unique_steps,
        "scene_split": split_report,
        "transition_rates": transition_rates,
        "observation_rates": observation_rates,
        "corruption_rates": corruptions,
        "confidence_modes": confidence_modes,
        "candidate_stay_probabilities": stay_probs,
        "selected_stay_probability": chosen_stay,
        "repeats": args.repeats,
        "decay": args.decay,
        "seed": args.seed,
        "interpretation_warning": (
            "LMEE supplies the real embodied trajectory timeline. "
            "Dynamic latent states, observation availability, corruption, "
            "and confidence are controlled overlays and are NOT native "
            "LMEE ground-truth state annotations."
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
        "scene_split.json",
        "calibration_tables.json",
        "validation_stay_sweep.csv",
        "test_results.csv",
        "test_overall.json",
        "experiment_config.json",
    ]:
        print(f"  {out_dir / name}")


if __name__ == "__main__":
    main()
