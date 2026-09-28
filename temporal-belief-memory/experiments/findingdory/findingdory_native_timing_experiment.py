#!/usr/bin/env python3
"""
FindingDory StateMem — native interaction-order / placement-time experiment
===========================================================================

This experiment removes the previous *controlled transition-timing* assumption.

Native FindingDory signals used:
- ordered target objects in each episode
- place_oracle_action_seq[i] for target i
- the exact DESNAP_OBJECT action index inside each placement sequence
- native start and goal receptacle instances

We construct stable-world checkpoints:
    checkpoint 0: before any placement
    checkpoint 1: immediately after target 0's DESNAP_OBJECT
    checkpoint 2: immediately after target 1's DESNAP_OBJECT
    ...
At each checkpoint, already-placed objects are at their native goal
receptacles and not-yet-placed objects are at their native start receptacles.

Important:
- native interaction ORDER and native action-step placement EVENT TIMES are real.
- observation availability, corruption, and raw confidence remain controlled.
- RGB/VLM perception is still not used.

Protocol:
TRAIN scenes only:
  - calibration scenes -> one pooled confidence calibrator
  - tuning scenes      -> choose StateMem hyperparameters
Official FindingDory VAL scenes:
  - untouched final test

Expected existing files:
  findingdory_probe/metadata/findingdory/train/episodes.json.gz
  findingdory_probe/metadata/findingdory/val/episodes.json.gz
  findingdory_transitions/real_object_transitions.jsonl

Run:
    cd ~/Downloads
    python3 findingdory_native_timing_experiment.py

Outputs:
    findingdory_native_timing_results/
      native_event_verification.json
      split_report.json
      pooled_calibration.json
      validation_stay_sweep.csv
      validation_adaptive_sweep.csv
      test_results.csv
      test_overall.json
      experiment_config.json

No third-party packages required.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
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


def parse_float_list(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def normalize_recep(x):
    if isinstance(x, list) and x:
        return x[0]
    if isinstance(x, str):
        return x
    return None


# ---------------------------------------------------------------------
# Load FindingDory
# ---------------------------------------------------------------------

def load_episodes(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)["episodes"]


def load_transitions(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)

            if not r.get("start_receptacle") or not r.get("goal_receptacle"):
                continue

            # Instance-level move must actually change.
            if r["start_receptacle"] == r["goal_receptacle"]:
                continue

            rows.append(r)
    return rows


def build_native_events(episodes_by_split, transition_rows):
    """
    Map each FindingDory target object to its ordered placement action sequence
    and exact DESNAP_OBJECT index.

    The function is intentionally strict and records every violation.
    """
    transition_map = {
        (r["split"], str(r["episode_id"]), r["object_handle"]): r
        for r in transition_rows
    }

    verification = {
        "episodes_checked": 0,
        "targets_checked": 0,
        "events_built": 0,
        "episodes_with_exact_seq_keys": 0,
        "target_label_order_matches": 0,
        "exactly_one_desnap": 0,
        "zero_snap_object": 0,
        "stop_is_last": 0,
        "start_receptacle_matches": 0,
        "goal_receptacle_matches": 0,
        "missing_transition_rows": 0,
        "violations": [],
    }

    event_episodes = {"train": [], "val": []}
    all_native_gaps = []

    for split, episodes in episodes_by_split.items():
        for ep in episodes:
            verification["episodes_checked"] += 1

            eid = str(ep["episode_id"])
            scene = ep["scene_id"]
            targets = list((ep.get("targets") or {}).keys())
            seqs = ep.get("place_oracle_action_seq") or {}
            labels = (ep.get("info") or {}).get("object_labels") or {}
            target_receps = ep.get("target_receptacles") or []
            goal_receps = ep.get("goal_receptacles") or []

            expected_keys = {str(i) for i in range(len(targets))}
            if set(seqs.keys()) == expected_keys:
                verification["episodes_with_exact_seq_keys"] += 1
            else:
                verification["violations"].append({
                    "split": split,
                    "episode_id": eid,
                    "type": "sequence_keys_mismatch",
                    "expected": sorted(expected_keys),
                    "actual": sorted(seqs.keys()),
                })
                continue

            cumulative = 0
            events = []
            prev_desnap = None

            for i, handle in enumerate(targets):
                verification["targets_checked"] += 1

                if labels.get(handle) == f"hab2|{i}":
                    verification["target_label_order_matches"] += 1
                else:
                    verification["violations"].append({
                        "split": split,
                        "episode_id": eid,
                        "object_handle": handle,
                        "type": "target_label_order_mismatch",
                        "label": labels.get(handle),
                        "expected": f"hab2|{i}",
                    })

                key = (split, eid, handle)
                tr = transition_map.get(key)

                if tr is None:
                    verification["missing_transition_rows"] += 1
                    continue

                actions = seqs[str(i)]
                names = [
                    x[0]
                    for x in actions
                    if isinstance(x, list) and x
                ]

                desnap = [
                    j for j, name in enumerate(names)
                    if name == "DESNAP_OBJECT"
                ]

                snaps = sum(name == "SNAP_OBJECT" for name in names)

                if len(desnap) == 1:
                    verification["exactly_one_desnap"] += 1
                else:
                    verification["violations"].append({
                        "split": split,
                        "episode_id": eid,
                        "object_handle": handle,
                        "type": "desnap_count",
                        "count": len(desnap),
                    })
                    cumulative += len(actions)
                    continue

                if snaps == 0:
                    verification["zero_snap_object"] += 1
                else:
                    verification["violations"].append({
                        "split": split,
                        "episode_id": eid,
                        "object_handle": handle,
                        "type": "snap_object_count",
                        "count": snaps,
                    })

                if names and names[-1] == "STOP":
                    verification["stop_is_last"] += 1
                else:
                    verification["violations"].append({
                        "split": split,
                        "episode_id": eid,
                        "object_handle": handle,
                        "type": "stop_not_last",
                    })

                start_meta = (
                    normalize_recep(target_receps[i])
                    if i < len(target_receps)
                    else None
                )
                goal_meta = (
                    normalize_recep(goal_receps[i])
                    if i < len(goal_receps)
                    else None
                )

                if start_meta == tr["start_receptacle"]:
                    verification["start_receptacle_matches"] += 1
                else:
                    verification["violations"].append({
                        "split": split,
                        "episode_id": eid,
                        "object_handle": handle,
                        "type": "start_receptacle_mismatch",
                    })

                if goal_meta == tr["goal_receptacle"]:
                    verification["goal_receptacle_matches"] += 1
                else:
                    verification["violations"].append({
                        "split": split,
                        "episode_id": eid,
                        "object_handle": handle,
                        "type": "goal_receptacle_mismatch",
                    })

                local_desnap = desnap[0]
                global_start = cumulative
                global_desnap = cumulative + local_desnap
                global_end = cumulative + len(actions) - 1

                if prev_desnap is not None:
                    all_native_gaps.append(global_desnap - prev_desnap)
                prev_desnap = global_desnap

                event = {
                    **tr,
                    "interaction_index": i,
                    "interaction_global_start": global_start,
                    "desnap_local_action_index": local_desnap,
                    "desnap_global_action_index": global_desnap,
                    "interaction_global_end": global_end,
                    "interaction_action_count": len(actions),
                }

                events.append(event)
                verification["events_built"] += 1

                cumulative += len(actions)

            # Keep only episodes where every target produced an event.
            if len(events) == len(targets) and events:
                event_episodes[split].append({
                    "episode_id": eid,
                    "scene_id": scene,
                    "total_native_action_steps": cumulative,
                    "events": events,
                })

    verification["native_gap_action_steps"] = {
        "count": len(all_native_gaps),
        "median": (
            statistics.median(all_native_gaps)
            if all_native_gaps else None
        ),
        "mean": (
            statistics.mean(all_native_gaps)
            if all_native_gaps else None
        ),
        "min": min(all_native_gaps) if all_native_gaps else None,
        "max": max(all_native_gaps) if all_native_gaps else None,
    }

    return event_episodes, verification


# ---------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------

def make_train_scene_split(train_eps, seed):
    scenes = sorted({ep["scene_id"] for ep in train_eps})
    rng = random.Random(seed)
    rng.shuffle(scenes)
    cut = max(1, len(scenes) // 2)
    return {
        "calibration": set(scenes[:cut]),
        "tuning": set(scenes[cut:]),
    }


def filter_eps_by_scene(eps, scene_set):
    return [ep for ep in eps if ep["scene_id"] in scene_set]


# ---------------------------------------------------------------------
# Controlled perception over NATIVE checkpoint timeline
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


def build_checkpoint_stream(
    ep,
    observation_rate,
    corruption_rate,
    confidence_mode,
    seed,
):
    """
    checkpoint 0: before any placement (native time=0)
    checkpoint k+1: immediately after event k's native DESNAP_OBJECT

    For each object at each stable checkpoint:
      current native state = start receptacle until its placement checkpoint,
                             goal receptacle afterwards.

    We guarantee an observation:
      - at initial checkpoint
      - for the object that was just placed at its own checkpoint
    Other observations are sampled with observation_rate.
    """
    rng = random.Random(seed)
    events = ep["events"]

    times = [0] + [
        e["desnap_global_action_index"]
        for e in events
    ]

    stream = []

    for checkpoint_idx, native_time in enumerate(times):
        moved_idx = checkpoint_idx - 1

        observations = []

        for obj_idx, e in enumerate(events):
            truth = (
                e["goal_receptacle"]
                if obj_idx < checkpoint_idx
                else e["start_receptacle"]
            )

            forced = (
                checkpoint_idx == 0
                or obj_idx == moved_idx
            )

            observed = forced or (rng.random() < observation_rate)

            if observed:
                corrupted = rng.random() < corruption_rate
                wrong = (
                    e["start_receptacle"]
                    if truth == e["goal_receptacle"]
                    else e["goal_receptacle"]
                )
                obs_state = wrong if corrupted else truth
                conf = sample_confidence(
                    obs_state == truth,
                    confidence_mode,
                    rng,
                )
            else:
                corrupted = False
                obs_state = None
                conf = None

            observations.append({
                "object_index": obj_idx,
                "object_handle": e["object_handle"],
                "true_state": truth,
                "observed": observed,
                "observed_state": obs_state,
                "raw_confidence": conf,
                "corrupted": corrupted,
                "just_transitioned": obj_idx == moved_idx,
                "post_transition": obj_idx < checkpoint_idx,
            })

        stream.append({
            "checkpoint_index": checkpoint_idx,
            "native_action_time": native_time,
            "observations": observations,
        })

    return stream


# ---------------------------------------------------------------------
# Calibration
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
    episodes,
    observation_rates,
    corruptions,
    confidence_modes,
    bins,
    seed,
):
    cal = HistogramCalibrator(bins=bins)

    for mode in confidence_modes:
        for obsr in observation_rates:
            for cr in corruptions:
                for ep in episodes:
                    s = stable_seed(
                        seed,
                        "calibration",
                        ep["episode_id"],
                        mode,
                        obsr,
                        cr,
                    )
                    stream = build_checkpoint_stream(
                        ep,
                        observation_rate=obsr,
                        corruption_rate=cr,
                        confidence_mode=mode,
                        seed=s,
                    )

                    for cp in stream:
                        for x in cp["observations"]:
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
        self.decay = decay
        self.use_confidence = use_confidence
        self.scores = {s: 0.0 for s in self.states}

    def step(
        self,
        delta_units,
        observed_state=None,
        confidence=None,
    ):
        factor = math.exp(-self.decay * delta_units)

        for s in self.states:
            self.scores[s] *= factor

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

    def reliability(self, raw_conf):
        if self.use_raw:
            r = raw_conf
        else:
            r = self.calibrator.predict(raw_conf)
        return max(0.01, min(0.99, float(r)))

    @staticmethod
    def effective_stay(p, delta_units):
        # Symmetric 2-state Markov-chain power for potentially fractional dt.
        eig = max(0.0, min(1.0, 2.0 * p - 1.0))
        return 0.5 + 0.5 * (eig ** delta_units)

    def step(
        self,
        delta_units,
        observed_state=None,
        raw_confidence=None,
    ):
        base = self.base_stay

        if observed_state is not None and self.adaptive_strength > 0:
            current = max(self.belief, key=self.belief.get)
            r = self.reliability(raw_confidence)

            if observed_state != current and r > 0.5:
                strength = (r - 0.5) / 0.5
                base = max(
                    0.50,
                    base - self.adaptive_strength * strength,
                )

        stay = self.effective_stay(base, delta_units)
        a, b = self.states

        prior = {
            a: self.belief[a] * stay + self.belief[b] * (1.0 - stay),
            b: self.belief[b] * stay + self.belief[a] * (1.0 - stay),
        }

        if observed_state is None:
            self.belief = prior
            return dict(self.belief)

        r = self.reliability(raw_confidence)
        like = {
            s: r if s == observed_state else 1.0 - r
            for s in self.states
        }

        post = {
            s: prior[s] * like[s]
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

def evaluate_episode(
    ep,
    stream,
    calibrator,
    fixed_stay,
    adaptive_stay,
    adaptive_strength,
    decay,
    time_scale,
):
    events = ep["events"]

    models = {}

    for i, e in enumerate(events):
        states = [
            e["start_receptacle"],
            e["goal_receptacle"],
        ]
        models[i] = {
            "latest": None,
            "recency": RecencyTracker(states, decay, False),
            "recency_rawconf": RecencyTracker(states, decay, True),
            "raw": BinaryStateMem(
                states,
                fixed_stay,
                use_raw=True,
            ),
            "cal": BinaryStateMem(
                states,
                fixed_stay,
                calibrator=calibrator,
            ),
            "adaptive": BinaryStateMem(
                states,
                adaptive_stay,
                calibrator=calibrator,
                adaptive_strength=adaptive_strength,
            ),
        }

    counts = {
        m: {
            "all": 0,
            "post": 0,
            "immediate": 0,
            "noise": 0,
        }
        for m in METHODS
    }

    n_all = n_post = n_immediate = n_noise = 0
    raw_brier = cal_brier = ada_brier = 0.0

    # Track transition lag in checkpoint units.
    lag_found = {m: {} for m in METHODS}
    transitioned_at = {}

    prev_time = stream[0]["native_action_time"]

    for cp in stream:
        native_time = cp["native_action_time"]
        delta_units = max(
            0.0,
            (native_time - prev_time) / time_scale,
        )

        for x in cp["observations"]:
            i = x["object_index"]
            m = models[i]

            if x["observed"]:
                m["latest"] = x["observed_state"]
                p_rec = m["recency"].step(
                    delta_units,
                    x["observed_state"],
                    x["raw_confidence"],
                )
                p_recc = m["recency_rawconf"].step(
                    delta_units,
                    x["observed_state"],
                    x["raw_confidence"],
                )
                b_raw = m["raw"].step(
                    delta_units,
                    x["observed_state"],
                    x["raw_confidence"],
                )
                b_cal = m["cal"].step(
                    delta_units,
                    x["observed_state"],
                    x["raw_confidence"],
                )
                b_ada = m["adaptive"].step(
                    delta_units,
                    x["observed_state"],
                    x["raw_confidence"],
                )
            else:
                p_rec = m["recency"].step(delta_units)
                p_recc = m["recency_rawconf"].step(delta_units)
                b_raw = m["raw"].step(delta_units)
                b_cal = m["cal"].step(delta_units)
                b_ada = m["adaptive"].step(delta_units)

            preds = {
                "latest": m["latest"],
                "recency": p_rec,
                "recency_rawconf": p_recc,
                "statemem_raw": m["raw"].predict(),
                "statemem_calibrated": m["cal"].predict(),
                "statemem_adaptive": m["adaptive"].predict(),
            }

            y = x["true_state"]

            n_all += 1
            if x["post_transition"]:
                n_post += 1
            if x["just_transitioned"]:
                n_immediate += 1
                transitioned_at[i] = cp["checkpoint_index"]
            if x["corrupted"]:
                n_noise += 1

            for method, pred in preds.items():
                ok = int(pred == y)
                counts[method]["all"] += ok

                if x["post_transition"]:
                    counts[method]["post"] += ok
                if x["just_transitioned"]:
                    counts[method]["immediate"] += ok
                if x["corrupted"]:
                    counts[method]["noise"] += ok

                if (
                    i in transitioned_at
                    and method not in lag_found
                ):
                    pass

                if i in transitioned_at and i not in lag_found[method]:
                    goal = events[i]["goal_receptacle"]
                    if pred == goal:
                        lag_found[method][i] = (
                            cp["checkpoint_index"]
                            - transitioned_at[i]
                        )

            states = [
                events[i]["start_receptacle"],
                events[i]["goal_receptacle"],
            ]

            for s in states:
                target = 1.0 if s == y else 0.0
                raw_brier += (b_raw[s] - target) ** 2
                cal_brier += (b_cal[s] - target) ** 2
                ada_brier += (b_ada[s] - target) ** 2

        prev_time = native_time

    lag = {}
    miss = {}

    n_objects = len(events)

    for method in METHODS:
        vals = list(lag_found[method].values())
        lag[method] = mean_valid(vals)
        miss[method] = (
            (n_objects - len(vals)) / n_objects
            if n_objects else 0.0
        )

    return {
        "steps": n_all,
        "post_steps": n_post,
        "immediate_steps": n_immediate,
        "noise_steps": n_noise,
        "counts": counts,
        "lag": lag,
        "miss": miss,
        "raw_brier_sum": raw_brier,
        "cal_brier_sum": cal_brier,
        "ada_brier_sum": ada_brier,
    }


def aggregate(evals):
    n_all = sum(x["steps"] for x in evals)
    n_post = sum(x["post_steps"] for x in evals)
    n_immediate = sum(x["immediate_steps"] for x in evals)
    n_noise = sum(x["noise_steps"] for x in evals)

    out = {
        "n_episodes": len(evals),
        "n_object_checkpoints": n_all,
        "n_post_transition_checkpoints": n_post,
        "n_immediate_transition_checkpoints": n_immediate,
        "n_corrupted_observations": n_noise,
    }

    for m in METHODS:
        out[f"{m}_acc"] = (
            sum(x["counts"][m]["all"] for x in evals) / n_all
        )
        out[f"{m}_post_acc"] = (
            sum(x["counts"][m]["post"] for x in evals) / n_post
            if n_post else float("nan")
        )
        out[f"{m}_immediate_acc"] = (
            sum(x["counts"][m]["immediate"] for x in evals)
            / n_immediate
            if n_immediate else float("nan")
        )
        out[f"{m}_noise_acc"] = (
            sum(x["counts"][m]["noise"] for x in evals) / n_noise
            if n_noise else float("nan")
        )
        out[f"{m}_transition_lag"] = mean_valid(
            [x["lag"][m] for x in evals]
        )
        out[f"{m}_transition_miss_rate"] = mean_valid(
            [x["miss"][m] for x in evals]
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
    episodes,
    calibrator,
    observation_rate,
    corruption_rate,
    confidence_mode,
    fixed_stay,
    adaptive_stay,
    adaptive_strength,
    decay,
    time_scale,
    repeat,
    seed,
    split_name,
):
    evals = []

    for ep in episodes:
        s = stable_seed(
            seed,
            split_name,
            repeat,
            ep["episode_id"],
            observation_rate,
            corruption_rate,
            confidence_mode,
        )

        stream = build_checkpoint_stream(
            ep,
            observation_rate,
            corruption_rate,
            confidence_mode,
            s,
        )

        evals.append(
            evaluate_episode(
                ep,
                stream,
                calibrator,
                fixed_stay,
                adaptive_stay,
                adaptive_strength,
                decay,
                time_scale,
            )
        )

    return aggregate(evals)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--metadata-root",
        default="findingdory_probe/metadata/findingdory",
    )
    ap.add_argument(
        "--transitions",
        default="findingdory_transitions/real_object_transitions.jsonl",
    )
    ap.add_argument(
        "--out-dir",
        default="findingdory_native_timing_results",
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
        default="0.05,0.10,0.15,0.20,0.30",
    )
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

    root = Path(args.metadata_root)

    episodes_by_split = {
        "train": load_episodes(root / "train" / "episodes.json.gz"),
        "val": load_episodes(root / "val" / "episodes.json.gz"),
    }

    transitions = load_transitions(args.transitions)

    print("Verifying native FindingDory event mapping across the full dataset...")

    event_eps, verification = build_native_events(
        episodes_by_split,
        transitions,
    )

    with (out_dir / "native_event_verification.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(verification, f, indent=2)

    critical = [
        v for v in verification["violations"]
        if v["type"] in {
            "sequence_keys_mismatch",
            "target_label_order_mismatch",
            "desnap_count",
            "start_receptacle_mismatch",
            "goal_receptacle_mismatch",
        }
    ]

    print(
        f"  episodes checked: {verification['episodes_checked']}"
    )
    print(
        f"  targets checked:  {verification['targets_checked']}"
    )
    print(
        f"  native events:    {verification['events_built']}"
    )
    print(
        f"  critical violations: {len(critical)}"
    )
    print(
        "  native DESNAP gap median: "
        f"{verification['native_gap_action_steps']['median']} action steps"
    )

    if critical:
        raise SystemExit(
            "Native event mapping has critical violations. "
            "Inspect native_event_verification.json before continuing."
        )

    train_eps = event_eps["train"]
    test_eps = event_eps["val"]

    train_scenes = {e["scene_id"] for e in train_eps}
    test_scenes = {e["scene_id"] for e in test_eps}

    overlap = train_scenes & test_scenes
    if overlap:
        raise SystemExit(
            f"ERROR: official train/val scene overlap: {len(overlap)}"
        )

    split = make_train_scene_split(train_eps, args.seed)

    calibration_eps = filter_eps_by_scene(
        train_eps,
        split["calibration"],
    )
    tuning_eps = filter_eps_by_scene(
        train_eps,
        split["tuning"],
    )

    # Native time normalization from TRAIN only.
    gaps = []
    for ep in train_eps:
        t = [
            e["desnap_global_action_index"]
            for e in ep["events"]
        ]
        gaps.extend(
            b - a
            for a, b in zip(t, t[1:])
            if b > a
        )

    time_scale = (
        statistics.median(gaps)
        if gaps else 1.0
    )

    split_report = {
        "official_train_episodes": len(train_eps),
        "official_val_episodes": len(test_eps),
        "official_train_scenes": len(train_scenes),
        "official_val_scenes": len(test_scenes),
        "official_train_val_scene_overlap": 0,
        "calibration_train_scenes": len(split["calibration"]),
        "tuning_train_scenes": len(split["tuning"]),
        "calibration_episodes": len(calibration_eps),
        "tuning_episodes": len(tuning_eps),
        "native_time_scale_median_action_gap": time_scale,
    }

    with (out_dir / "split_report.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(split_report, f, indent=2)

    print()
    print("Fitting ONE pooled calibrator on train-calibration scenes...")

    calibrator = fit_pooled_calibrator(
        calibration_eps,
        observation_rates,
        corruptions,
        confidence_modes,
        args.calibration_bins,
        args.seed,
    )

    with (out_dir / "pooled_calibration.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(calibrator.table(), f, indent=2)

    # -------------------------------------------------------------
    # Fixed StateMem tune.
    # -------------------------------------------------------------
    print("Tuning fixed persistence using native event checkpoints...")

    stay_rows = []

    for stay in stay_probs:
        ms = []

        for mode in confidence_modes:
            for obsr in observation_rates:
                for cr in corruptions:
                    ms.append(
                        evaluate_condition(
                            tuning_eps,
                            calibrator,
                            obsr,
                            cr,
                            mode,
                            fixed_stay=stay,
                            adaptive_stay=stay,
                            adaptive_strength=0.0,
                            decay=args.decay,
                            time_scale=time_scale,
                            repeat=0,
                            seed=args.seed,
                            split_name="tuning-fixed",
                        )
                    )

        row = {
            "stay_probability": stay,
            "mean_accuracy": mean_valid(
                [x["statemem_calibrated_acc"] for x in ms]
            ),
            "mean_post_accuracy": mean_valid(
                [x["statemem_calibrated_post_acc"] for x in ms]
            ),
            "mean_immediate_accuracy": mean_valid(
                [x["statemem_calibrated_immediate_acc"] for x in ms]
            ),
            "mean_transition_lag": mean_valid(
                [x["statemem_calibrated_transition_lag"] for x in ms]
            ),
            "mean_brier": mean_valid(
                [x["statemem_calibrated_brier"] for x in ms]
            ),
        }
        stay_rows.append(row)

        print(
            f"  stay={stay:.2f} | "
            f"acc={100*row['mean_accuracy']:.2f}% | "
            f"post={100*row['mean_post_accuracy']:.2f}% | "
            f"immediate={100*row['mean_immediate_accuracy']:.2f}% | "
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
            x["mean_immediate_accuracy"],
            -x["mean_brier"],
        ),
    )

    fixed_stay = best_fixed["stay_probability"]
    print(f"Selected fixed stay={fixed_stay:.2f}")
    print()

    # -------------------------------------------------------------
    # Adaptive tune.
    # -------------------------------------------------------------
    print("Tuning adaptive persistence...")

    adaptive_rows = []

    for stay in stay_probs:
        for strength in adaptive_strengths:
            ms = []

            for mode in confidence_modes:
                for obsr in observation_rates:
                    for cr in corruptions:
                        ms.append(
                            evaluate_condition(
                                tuning_eps,
                                calibrator,
                                obsr,
                                cr,
                                mode,
                                fixed_stay=fixed_stay,
                                adaptive_stay=stay,
                                adaptive_strength=strength,
                                decay=args.decay,
                                time_scale=time_scale,
                                repeat=0,
                                seed=args.seed,
                                split_name="tuning-adaptive",
                            )
                        )

            adaptive_rows.append({
                "stay_probability": stay,
                "adaptive_strength": strength,
                "mean_accuracy": mean_valid(
                    [x["statemem_adaptive_acc"] for x in ms]
                ),
                "mean_post_accuracy": mean_valid(
                    [x["statemem_adaptive_post_acc"] for x in ms]
                ),
                "mean_immediate_accuracy": mean_valid(
                    [x["statemem_adaptive_immediate_acc"] for x in ms]
                ),
                "mean_transition_lag": mean_valid(
                    [x["statemem_adaptive_transition_lag"] for x in ms]
                ),
                "mean_brier": mean_valid(
                    [x["statemem_adaptive_brier"] for x in ms]
                ),
            })

    save_csv(
        out_dir / "validation_adaptive_sweep.csv",
        adaptive_rows,
    )

    best_adaptive = max(
        adaptive_rows,
        key=lambda x: (
            x["mean_post_accuracy"],
            x["mean_immediate_accuracy"],
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

    # -------------------------------------------------------------
    # Official VAL.
    # -------------------------------------------------------------
    print("Running untouched official FindingDory VAL scenes...")

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
                        test_eps,
                        calibrator,
                        obsr,
                        cr,
                        mode,
                        fixed_stay=fixed_stay,
                        adaptive_stay=adaptive_stay,
                        adaptive_strength=adaptive_strength,
                        decay=args.decay,
                        time_scale=time_scale,
                        repeat=rep,
                        seed=args.seed,
                        split_name="official-val-native-time",
                    )

                    test_rows.append({
                        "confidence_mode": mode,
                        "observation_rate": obsr,
                        "corruption_rate": cr,
                        "repeat": rep,
                        "fixed_stay": fixed_stay,
                        "adaptive_stay": adaptive_stay,
                        "adaptive_strength": adaptive_strength,
                        **m,
                    })

                    done += 1
                    if done % 10 == 0 or done == total_runs:
                        print(f"  completed {done}/{total_runs}")

    save_csv(out_dir / "test_results.csv", test_rows)

    labels = [
        ("Latest", "latest"),
        ("Recency", "recency"),
        ("Recency x RawConf", "recency_rawconf"),
        ("StateMem-Raw", "statemem_raw"),
        ("StateMem-Calibrated", "statemem_calibrated"),
        ("StateMem-Adaptive", "statemem_adaptive"),
    ]

    overall = {
        "selected_fixed_stay": fixed_stay,
        "selected_adaptive_stay": adaptive_stay,
        "selected_adaptive_strength": adaptive_strength,
        "native_time_scale_median_action_gap": time_scale,
        "official_val_episodes": len(test_eps),
        "methods": {},
    }

    print()
    print("=" * 132)
    print("FINDINGDORY OFFICIAL VAL — NATIVE ORDER + NATIVE DESNAP TIMING")
    print("=" * 132)
    print(
        f"{'Method':<24} | "
        f"{'All Acc':>8} | "
        f"{'Post':>8} | "
        f"{'Immediate':>10} | "
        f"{'Noise':>8} | "
        f"{'Lag(cp)':>8} | "
        f"{'Brier':>8}"
    )
    print("-" * 132)

    for label, key in labels:
        entry = {
            "accuracy": mean_valid(
                [r[f"{key}_acc"] for r in test_rows]
            ),
            "post_accuracy": mean_valid(
                [r[f"{key}_post_acc"] for r in test_rows]
            ),
            "immediate_transition_accuracy": mean_valid(
                [r[f"{key}_immediate_acc"] for r in test_rows]
            ),
            "noise_accuracy": mean_valid(
                [r[f"{key}_noise_acc"] for r in test_rows]
            ),
            "transition_lag_checkpoints": mean_valid(
                [r[f"{key}_transition_lag"] for r in test_rows]
            ),
            "transition_miss_rate": mean_valid(
                [r[f"{key}_transition_miss_rate"] for r in test_rows]
            ),
        }

        bk = f"{key}_brier"
        entry["brier"] = (
            mean_valid([r[bk] for r in test_rows])
            if bk in test_rows[0]
            else None
        )

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
            f"{100*entry['immediate_transition_accuracy']:9.2f}% | "
            f"{100*entry['noise_accuracy']:7.2f}% | "
            f"{entry['transition_lag_checkpoints']:8.3f} | "
            f"{bt:>8}"
        )

    print("=" * 132)

    with (out_dir / "test_overall.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(overall, f, indent=2)

    config = {
        "metadata_root": args.metadata_root,
        "transitions": args.transitions,
        "native_signals": [
            "FindingDory target interaction order",
            "per-target place_oracle_action_seq",
            "native DESNAP_OBJECT placement action index",
            "native start receptacle instance",
            "native goal receptacle instance",
        ],
        "controlled_components_remaining": [
            "observation availability at stable checkpoints",
            "observation corruption",
            "raw perception confidence",
        ],
        "rgb_vlm_used": False,
        "candidate_stay_probabilities": stay_probs,
        "candidate_adaptive_strengths": adaptive_strengths,
        "selected_fixed_stay": fixed_stay,
        "selected_adaptive_stay": adaptive_stay,
        "selected_adaptive_strength": adaptive_strength,
        "native_time_scale_median_action_gap": time_scale,
        "repeats": args.repeats,
        "seed": args.seed,
        "important_limitation": (
            "Native FindingDory interaction order and placement-event timing are "
            "used. Perception is still controlled rather than RGB/VLM-derived."
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
        "native_event_verification.json",
        "split_report.json",
        "pooled_calibration.json",
        "validation_stay_sweep.csv",
        "validation_adaptive_sweep.csv",
        "test_results.csv",
        "test_overall.json",
        "experiment_config.json",
    ]:
        print(f"  {out_dir / name}")


if __name__ == "__main__":
    main()
