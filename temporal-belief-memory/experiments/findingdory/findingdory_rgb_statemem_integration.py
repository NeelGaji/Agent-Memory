#!/usr/bin/env python3
"""
FindingDory real-RGB -> StateMem integration
============================================

This is the main mechanism experiment after the RGB grounding pilots.

Research question
-----------------
Does calibrated temporal belief reconciliation improve current-state
estimation when the observations themselves come from a real egocentric VLM,
rather than from synthetic corruption?

Protocol
--------
TRAIN ONLY
  1. Official FindingDory train scenes are split deterministically into:
       - calibration scenes
       - tuning scenes
  2. Real RGB evidence is generated with Qwen2.5-VL-7B-Instruct-4bit.
  3. A/B candidate order is deterministically randomized per observation.
  4. Observation reliability is learned ONLY from calibration scenes.
  5. StateMem persistence and the recency baseline are tuned ONLY on tuning
     scenes.

OFFICIAL VALIDATION
  6. Calibration, persistence, and recency settings are frozen.
  7. Official FindingDory validation episodes are evaluated untouched.

RGB supervision
---------------
This experiment uses FindingDory task_58/task_59 answer-frame groups to select
informative RGB evidence:
  task_58 -> picked-from receptacle -> START evidence
  task_59 -> placed-on receptacle   -> GOAL evidence

Therefore this is an ORACLE-FRAME RGB StateMem experiment. The observations
are real RGB/VLM outputs and their errors are real model errors, but the frame
selection is benchmark-annotated rather than end-to-end.

Each state event is represented by up to two bundles of nearby official RGB
frames. Each bundle gets ONE neutral A/B VLM classification. Candidate order
is randomized, so the model never sees the words START or GOAL.

Observable calibration features:
  - chosen option position A/B
  - predicted semantic state START/GOAL
  - number of RGB images in the bundle

No self-reported VLM confidence is used.

Baselines
---------
Latest
Majority
Recency-Calibrated
Bayes-NoDynamics
StateMem-Uniform
StateMem-Calibrated

Recommended first run (pipeline pilot)
--------------------------------------
cd ~/Downloads
python3 findingdory_rgb_statemem_integration.py --pilot

Recommended final run
---------------------
python3 findingdory_rgb_statemem_integration.py --full

The script is resumable. VLM calls are cached in:
  findingdory_rgb_statemem/vlm_cache.jsonl
so an interrupted run can simply be restarted.

Outputs
-------
findingdory_rgb_statemem/
  experiment_config.json
  protocol_report.json
  calibration_table.json
  tuning_stay_sweep.csv
  tuning_recency_sweep.csv
  test_overall.json
  test_transition_results.csv
  test_step_results.csv
  vlm_cache.jsonl
"""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

DATASET_REPO = "yali30/findingdory-subsampled-96"
MODEL_ID = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"

ORDINAL_TO_INDEX = {
    "first": 0,
    "second": 1,
    "third": 2,
    "fourth": 3,
    "fifth": 4,
    "sixth": 5,
    "seventh": 6,
    "eighth": 7,
    "ninth": 8,
    "tenth": 9,
    "eleventh": 10,
}

STAY_CANDIDATES = [0.55, 0.65, 0.75, 0.80, 0.85, 0.90, 0.95]
RECENCY_DECAYS = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00]


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def stable_int(*parts, modulo=None):
    raw = "|".join(map(str, parts)).encode("utf-8")
    x = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")
    return x if modulo is None else x % modulo


def normalize_category(x):
    return str(x or "").replace("_", " ").strip().lower()


def load_json_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def action_name(item):
    if isinstance(item, list) and item:
        return str(item[0])
    return ""


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


def mean_or_none(xs):
    return sum(xs) / len(xs) if xs else None


# ---------------------------------------------------------------------
# Native FindingDory metadata and transition timing
# ---------------------------------------------------------------------

def load_transitions(path: Path):
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            rows[(r["split"], str(r["episode_id"]), r["object_handle"])] = r
    return rows


def candidate_category_map(ep):
    out = {}
    for key in ("candidate_objects", "candidate_objects_noninteracted"):
        for x in ep.get(key) or []:
            if x.get("object_name") and x.get("object_category"):
                out[x["object_name"]] = x["object_category"]
    return out


def build_native_events(metadata_root: Path, transitions_path: Path):
    transitions = load_transitions(transitions_path)
    result = {"train": {}, "val": {}}
    verification = Counter()

    for split in ("train", "val"):
        data = load_json_gz(metadata_root / split / "episodes.json.gz")
        for ep in data["episodes"]:
            eid = str(ep["episode_id"])
            targets = list((ep.get("targets") or {}).keys())
            seqs = ep.get("place_oracle_action_seq") or {}
            cmap = candidate_category_map(ep)

            if set(seqs) != {str(i) for i in range(len(targets))}:
                verification[f"{split}_bad_seq_keys"] += 1
                continue

            events = []
            ok = True

            for i, handle in enumerate(targets):
                tr = transitions.get((split, eid, handle))
                if tr is None:
                    verification[f"{split}_missing_transition"] += 1
                    ok = False
                    break

                seq = seqs[str(i)]
                names = [action_name(x) for x in seq]
                des = [j for j, n in enumerate(names) if n == "DESNAP_OBJECT"]

                if len(des) != 1:
                    verification[f"{split}_bad_desnap"] += 1
                    ok = False
                    break

                obj_cat = cmap.get(tr.get("object_config"))
                if not obj_cat:
                    obj_cat = (
                        str(tr.get("object_config", "object"))
                        .replace(".object_config.json", "")
                        .replace("_", " ")
                    )

                events.append({
                    **tr,
                    "interaction_index": i,
                    "object_category": normalize_category(obj_cat),
                    "local_desnap_action_index": des[0],
                    "interaction_action_count": len(seq),
                })

            if ok and len(events) == len(targets) and events:
                result[split][eid] = {
                    "episode_id": eid,
                    "scene_id": ep["scene_id"],
                    "events": events,
                }
                verification[f"{split}_good_episodes"] += 1
                verification[f"{split}_good_events"] += len(events)

    return result, dict(verification)


# ---------------------------------------------------------------------
# Official task_58 / task_59 RGB supervision + exact video paths
# ---------------------------------------------------------------------

def parse_answer(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return None
    return None


def parse_order(question):
    m = re.search(
        r"order\s+to\s+revisit\s+them\s+is\s*:\s*([^\.]+)",
        str(question).lower(),
    )
    if not m:
        return None

    words = re.findall(
        r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh)\b",
        m.group(1),
    )
    if not words:
        return None
    return [ORDINAL_TO_INDEX[w] for w in words]


def clean_frame_group(group):
    if not isinstance(group, list):
        return []
    return sorted(set(
        int(x)
        for x in group
        if isinstance(x, (int, float)) and 0 <= int(x) <= 95
    ))


def extract_video_relpath(v):
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        for key in ("path", "filename", "file_name"):
            if isinstance(v.get(key), str):
                return v[key]
    p = getattr(v, "path", None)
    if isinstance(p, str):
        return p
    return None


def load_hf_supervision():
    from datasets import load_dataset

    all_out = {}
    stats = {}

    for hf_split, habitat_split in (("train", "train"), ("validation", "val")):
        print(f"Loading FindingDory {hf_split} task metadata...")
        ds = load_dataset(DATASET_REPO, split=hf_split)

        by_ep = defaultdict(lambda: {
            "start": {},
            "goal": {},
            "video_relpath": None,
        })
        c = Counter()

        for row in ds:
            task = str(row["task_id"])
            if task not in {"task_58", "task_59"}:
                continue

            eid = str(row["ep_id"])
            if eid.startswith("ep_"):
                eid = eid[3:]

            rel = extract_video_relpath(row.get("video"))
            if rel:
                by_ep[eid]["video_relpath"] = rel

            order = parse_order(row["question"])
            ans = parse_answer(row["answer"])

            if order is None:
                c["order_parse_failed"] += 1
                continue
            if not isinstance(ans, list):
                c["answer_parse_failed"] += 1
                continue
            if len(order) != len(ans):
                c["order_answer_length_mismatch"] += 1
                continue

            side = "start" if task == "task_58" else "goal"

            for interaction_idx, group in zip(order, ans):
                frames = clean_frame_group(group)
                if not frames:
                    c["empty_group"] += 1
                    continue
                cur = set(by_ep[eid][side].get(interaction_idx, []))
                cur.update(frames)
                by_ep[eid][side][interaction_idx] = sorted(cur)

            c[f"{side}_rows"] += 1
            c[f"{side}_mapped_groups"] += len(order)

        all_out[habitat_split] = dict(by_ep)
        stats[habitat_split] = dict(c)

    return all_out, stats


def build_video_file_list(cache_dir: Path):
    return list(cache_dir.rglob("*.mp4"))


def resolve_video_path(cache_dir: Path, files, relpath, split, eid):
    candidates = []

    if relpath:
        rel = Path(str(relpath))
        candidates.extend([
            cache_dir / rel,
            cache_dir / "videos" / rel,
        ])

    split_dir = "train" if split == "train" else "val"
    candidates.extend([
        cache_dir / "videos" / split_dir / f"ep_{eid}.mp4",
        cache_dir / split_dir / f"ep_{eid}.mp4",
    ])

    for p in candidates:
        if p.exists():
            return p

    # Exact suffix match only. Never resolve by basename alone because train/val
    # can reuse ep_*.mp4 names.
    suffixes = []
    if relpath:
        suffixes.append(str(Path(str(relpath))).replace("\\", "/"))
    suffixes.append(f"videos/{split_dir}/ep_{eid}.mp4")
    suffixes.append(f"{split_dir}/ep_{eid}.mp4")

    hits = []
    for p in files:
        s = str(p).replace("\\", "/")
        if any(s.endswith(suf) for suf in suffixes):
            hits.append(p)

    unique = list(dict.fromkeys(hits))
    if len(unique) == 1:
        return unique[0]
    return None


# ---------------------------------------------------------------------
# Eligible events and split selection
# ---------------------------------------------------------------------

def attach_rgb_supervision(native, supervision, cache_dir: Path):
    video_files = build_video_file_list(cache_dir)
    result = {"train": {}, "val": {}}
    counts = {"train": Counter(), "val": Counter()}

    for split in ("train", "val"):
        for eid, ep in native[split].items():
            sup = supervision[split].get(eid)
            if sup is None:
                counts[split]["missing_supervision_episode"] += 1
                continue

            video = resolve_video_path(
                cache_dir,
                video_files,
                sup.get("video_relpath"),
                split,
                eid,
            )
            if video is None:
                counts[split]["missing_exact_video_path"] += 1
                continue

            usable = []
            for ev in ep["events"]:
                if normalize_category(ev["start_receptacle_category"]) == normalize_category(
                    ev["goal_receptacle_category"]
                ):
                    counts[split]["same_category_skipped"] += 1
                    continue

                i = ev["interaction_index"]
                sf = sup["start"].get(i, [])
                gf = sup["goal"].get(i, [])

                if not sf:
                    counts[split]["missing_start_group"] += 1
                if not gf:
                    counts[split]["missing_goal_group"] += 1
                if not sf or not gf:
                    continue

                usable.append({
                    **ev,
                    "start_frames": sf,
                    "goal_frames": gf,
                })
                counts[split]["usable_events"] += 1

            if usable:
                result[split][eid] = {
                    **ep,
                    "events": usable,
                    "video_path": str(video),
                }
                counts[split]["eligible_episodes"] += 1

    return result, {
        k: dict(v)
        for k, v in counts.items()
    }


def scene_split_train(episodes, seed):
    scenes = sorted(set(ep["scene_id"] for ep in episodes.values()))
    rng = random.Random(seed)
    rng.shuffle(scenes)

    mid = len(scenes) // 2
    calibration = set(scenes[:mid])
    tuning = set(scenes[mid:])
    return calibration, tuning


def cap_scenes(scene_set, cap, seed, tag):
    xs = sorted(scene_set)
    if cap is None or cap <= 0 or cap >= len(xs):
        return set(xs)
    rng = random.Random(stable_int(seed, tag))
    rng.shuffle(xs)
    return set(xs[:cap])


def round_robin_events(episodes, allowed_scenes=None, cap=None, seed=0, tag=""):
    per_scene = defaultdict(list)

    for ep in episodes.values():
        if allowed_scenes is not None and ep["scene_id"] not in allowed_scenes:
            continue
        for ev in ep["events"]:
            per_scene[ep["scene_id"]].append({
                "episode_id": ep["episode_id"],
                "scene_id": ep["scene_id"],
                "video_path": ep["video_path"],
                **ev,
            })

    for scene, rows in per_scene.items():
        rng = random.Random(stable_int(seed, tag, scene))
        rng.shuffle(rows)

    scenes = sorted(per_scene)
    out = []
    idx = 0

    while scenes:
        next_scenes = []
        for s in scenes:
            rows = per_scene[s]
            if idx < len(rows):
                out.append(rows[idx])
                if cap and len(out) >= cap:
                    return out
            if idx + 1 < len(rows):
                next_scenes.append(s)
        scenes = next_scenes
        idx += 1

    return out


# ---------------------------------------------------------------------
# RGB bundles
# ---------------------------------------------------------------------

def representative_frames(indices, n):
    xs = sorted(set(indices))
    if len(xs) <= n:
        return xs
    qs = [(i + 1) / (n + 1) for i in range(n)]
    return sorted(set(xs[round(q * (len(xs) - 1))] for q in qs))


def make_bundles(indices, bundles_per_state=2, images_per_bundle=2):
    max_needed = bundles_per_state * images_per_bundle
    chosen = representative_frames(indices, max_needed)

    if not chosen:
        return []

    if bundles_per_state <= 1:
        return [chosen[:images_per_bundle]]

    # Split chosen frames as evenly as possible into ordered bundles.
    bundles = []
    n = len(chosen)
    for b in range(bundles_per_state):
        lo = round(b * n / bundles_per_state)
        hi = round((b + 1) * n / bundles_per_state)
        part = chosen[lo:hi]
        if part:
            bundles.append(part[:images_per_bundle])

    return bundles


# ---------------------------------------------------------------------
# VLM inference cache
# ---------------------------------------------------------------------

def load_vlm(model_id):
    from mlx_vlm import load
    from mlx_vlm.utils import load_config

    print()
    print(f"Loading local VLM: {model_id}")
    model, processor = load(model_id)
    config = load_config(model_id)

    try:
        if hasattr(processor, "image_processor") and hasattr(
            processor.image_processor, "max_pixels"
        ):
            processor.image_processor.max_pixels = 1280 * 28 * 28
    except Exception:
        pass

    return model, processor, config


def read_frame(path: Path, idx: int):
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not decode frame {idx} from {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def save_rgb(path: Path, rgb):
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(path),
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
    )


def generation_text(x):
    if isinstance(x, str):
        return x
    t = getattr(x, "text", None)
    return t if isinstance(t, str) else str(x)


def parse_ab(text):
    t = str(text).strip().upper()
    if "NOT_VISIBLE" in t or "NOT VISIBLE" in t:
        return "NOT_VISIBLE"
    if "UNCERTAIN" in t or "UNSURE" in t:
        return "UNCERTAIN"
    m = re.search(r"\b(A|B)\b", t)
    if m:
        return m.group(1)
    return "PARSE_ERROR"


def vlm_classify(model, processor, config, image_paths, option_a, option_b):
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    prompt = (
        f"You are given {len(image_paths)} egocentric robot image(s) from one "
        "short observation window. "
        f"Option A = {normalize_category(option_a)}. "
        f"Option B = {normalize_category(option_b)}. "
        "Which receptacle/location category is better supported by the images? "
        "Reply with exactly one token: A, B, NOT_VISIBLE, or UNCERTAIN."
    )

    formatted = apply_chat_template(
        processor,
        config,
        prompt,
        num_images=len(image_paths),
    )
    res = generate(
        model,
        processor,
        formatted,
        [str(p) for p in image_paths],
        max_tokens=8,
        temp=0.0,
        verbose=False,
    )
    raw = generation_text(res).strip()
    return parse_ab(raw), raw


def cache_key(row):
    return json.dumps({
        "model_id": row["model_id"],
        "split": row["split"],
        "episode_id": row["episode_id"],
        "interaction_index": row["interaction_index"],
        "phase": row["phase"],
        "bundle_index": row["bundle_index"],
        "frame_indices": row["frame_indices"],
        "option_a": row["option_a"],
        "option_b": row["option_b"],
    }, sort_keys=True)


def load_cache(path: Path):
    out = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            out[cache_key(r)] = r
    return out


def generate_observations(
    events,
    split,
    out_dir,
    cache_path,
    model,
    processor,
    config,
    model_id,
    seed,
    bundles_per_state,
    images_per_bundle,
):
    cache = load_cache(cache_path)
    frames_dir = out_dir / "rgb_frames"
    observations = []

    total_expected = 0
    for ev in events:
        total_expected += len(make_bundles(
            ev["start_frames"], bundles_per_state, images_per_bundle
        ))
        total_expected += len(make_bundles(
            ev["goal_frames"], bundles_per_state, images_per_bundle
        ))

    done_counter = 0

    with cache_path.open("a", encoding="utf-8") as cache_f:
        for event_n, ev in enumerate(events, 1):
            for phase, frame_key, gt_state in (
                ("START", "start_frames", "START"),
                ("GOAL", "goal_frames", "GOAL"),
            ):
                bundles = make_bundles(
                    ev[frame_key],
                    bundles_per_state=bundles_per_state,
                    images_per_bundle=images_per_bundle,
                )

                for bundle_idx, frame_indices in enumerate(bundles):
                    start_is_a = stable_int(
                        seed,
                        split,
                        ev["episode_id"],
                        ev["interaction_index"],
                        phase,
                        bundle_idx,
                        modulo=2,
                    ) == 0

                    if start_is_a:
                        option_a = ev["start_receptacle_category"]
                        option_b = ev["goal_receptacle_category"]
                        a_state, b_state = "START", "GOAL"
                    else:
                        option_a = ev["goal_receptacle_category"]
                        option_b = ev["start_receptacle_category"]
                        a_state, b_state = "GOAL", "START"

                    base = {
                        "model_id": model_id,
                        "split": split,
                        "episode_id": ev["episode_id"],
                        "scene_id": ev["scene_id"],
                        "interaction_index": ev["interaction_index"],
                        "object_handle": ev["object_handle"],
                        "phase": phase,
                        "gt_state": gt_state,
                        "bundle_index": bundle_idx,
                        "frame_indices": frame_indices,
                        "option_a": option_a,
                        "option_b": option_b,
                        "a_state": a_state,
                        "b_state": b_state,
                        "transition_gap_actions": ev["local_desnap_action_index"],
                    }

                    key = cache_key(base)
                    if key in cache:
                        r = cache[key]
                        observations.append(r)
                        done_counter += 1
                        continue

                    image_paths = []
                    for fi in frame_indices:
                        frame_path = frames_dir / (
                            f"{split}_ep_{ev['episode_id']}_"
                            f"obj_{ev['interaction_index']:02d}_"
                            f"{phase.lower()}_b{bundle_idx}_f{fi:02d}.jpg"
                        )
                        if not frame_path.exists():
                            rgb = read_frame(Path(ev["video_path"]), fi)
                            save_rgb(frame_path, rgb)
                        image_paths.append(frame_path)

                    raw_label, raw_text = vlm_classify(
                        model,
                        processor,
                        config,
                        image_paths,
                        option_a,
                        option_b,
                    )

                    if raw_label == "A":
                        pred_state = a_state
                    elif raw_label == "B":
                        pred_state = b_state
                    else:
                        pred_state = None

                    r = {
                        **base,
                        "raw_label": raw_label,
                        "pred_state": pred_state,
                        "raw_output": raw_text,
                        "correct": pred_state == gt_state if pred_state else False,
                        "num_images": len(image_paths),
                    }

                    cache_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    cache_f.flush()
                    cache[key] = r
                    observations.append(r)
                    done_counter += 1

                    if done_counter % 20 == 0 or done_counter == total_expected:
                        print(
                            f"[{split}] VLM observations "
                            f"{done_counter}/{total_expected}"
                        )

    return observations


# ---------------------------------------------------------------------
# Reliability calibration
# ---------------------------------------------------------------------

def calibration_feature(obs):
    if obs["pred_state"] not in {"START", "GOAL"}:
        return None
    return (
        obs["raw_label"],
        obs["pred_state"],
        int(obs["num_images"]),
    )


def fit_calibration(observations, min_group_n=20, prior_strength=12.0):
    judged = [o for o in observations if o["pred_state"] in {"START", "GOAL"}]
    if not judged:
        raise RuntimeError("No judged calibration observations.")

    global_correct = sum(o["correct"] for o in judged)
    global_r = global_correct / len(judged)

    groups = defaultdict(lambda: [0, 0])

    for o in judged:
        k = calibration_feature(o)
        groups[k][0] += 1
        groups[k][1] += int(o["correct"])

    table = {}
    report_groups = []

    for k, (n, correct) in sorted(groups.items(), key=lambda x: str(x[0])):
        empirical = correct / n
        if n < min_group_n:
            calibrated = global_r
            used_fallback = True
        else:
            calibrated = (
                correct + prior_strength * global_r
            ) / (
                n + prior_strength
            )
            used_fallback = False

        # Allow anti-evidence if a repeatable feature is genuinely < .5, but
        # avoid numerical extremes.
        calibrated = max(0.05, min(0.95, calibrated))
        table[k] = calibrated

        report_groups.append({
            "raw_label": k[0],
            "pred_state": k[1],
            "num_images": k[2],
            "n": n,
            "correct": correct,
            "empirical_accuracy": empirical,
            "calibrated_reliability": calibrated,
            "fallback_to_global": used_fallback,
        })

    return {
        "global_reliability": global_r,
        "judged_observations": len(judged),
        "abstentions": len(observations) - len(judged),
        "groups": report_groups,
    }, table


def reliability_for(obs, table, global_r):
    k = calibration_feature(obs)
    if k is None:
        return None
    return table.get(k, global_r)


# ---------------------------------------------------------------------
# Models / baselines
# ---------------------------------------------------------------------

def effective_stay(base_stay, delta):
    base_stay = max(0.5, min(0.999999, base_stay))
    delta = max(0.0, float(delta))
    return 0.5 + 0.5 * ((2 * base_stay - 1) ** delta)


def transition_predict(p_goal, stay, delta):
    eff = effective_stay(stay, delta)
    return eff * p_goal + (1 - eff) * (1 - p_goal)


def bayes_update(p_goal, obs_state, reliability):
    if obs_state not in {"START", "GOAL"} or reliability is None:
        return p_goal

    r = clamp(reliability, 0.001, 0.999)

    if obs_state == "GOAL":
        like_goal = r
        like_start = 1 - r
    else:
        like_goal = 1 - r
        like_start = r

    num = p_goal * like_goal
    den = num + (1 - p_goal) * like_start
    if den <= 0:
        return p_goal
    return num / den


def hard_state_from_p(p_goal):
    return "GOAL" if p_goal >= 0.5 else "START"


def calibrated_recency_prediction(history, decay):
    """
    history: list of (pred_state, reliability)
    Exponential recency over observation index. Reliability enters as signed
    log-odds, so train-derived anti-evidence (r < .5) can be inverted.
    """
    if not history:
        return None, 0.5

    score = 0.0
    n = len(history)

    for i, (state, r) in enumerate(history):
        age = n - 1 - i
        w = math.exp(-decay * age)
        signed = logit(clamp(r, 0.01, 0.99))
        if state == "GOAL":
            score += w * signed
        else:
            score -= w * signed

    p_goal = sigmoid(score)
    return hard_state_from_p(p_goal), p_goal


def majority_prediction(labels):
    judged = [x for x in labels if x in {"START", "GOAL"}]
    if not judged:
        return None
    c = Counter(judged)
    if c["GOAL"] == c["START"]:
        return judged[-1]
    return "GOAL" if c["GOAL"] > c["START"] else "START"


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

METHODS = [
    "Latest",
    "Majority",
    "Recency-Calibrated",
    "Bayes-NoDynamics",
    "StateMem-Uniform",
    "StateMem-Calibrated",
]


def group_observations_by_event(observations):
    d = defaultdict(list)
    for o in observations:
        d[
            (
                o["episode_id"],
                o["interaction_index"],
                o["object_handle"],
            )
        ].append(o)

    result = []
    for key, rows in d.items():
        start = sorted(
            [r for r in rows if r["phase"] == "START"],
            key=lambda r: (r["bundle_index"], r["frame_indices"]),
        )
        goal = sorted(
            [r for r in rows if r["phase"] == "GOAL"],
            key=lambda r: (r["bundle_index"], r["frame_indices"]),
        )
        if start and goal:
            result.append((key, start, goal))

    return result


def evaluate(
    observations,
    calibration_table,
    global_r,
    stay,
    recency_decay,
    time_scale,
    split_name,
    write_steps=False,
):
    grouped = group_observations_by_event(observations)

    aggregate = {
        m: {
            "correct": 0,
            "n": 0,
            "pre_correct": 0,
            "pre_n": 0,
            "post_correct": 0,
            "post_n": 0,
            "immediate_correct": 0,
            "immediate_n": 0,
            "final_correct": 0,
            "final_n": 0,
            "brier_sum": 0.0,
            "brier_n": 0,
            "lag_sum": 0.0,
            "lag_n": 0,
            "misses": 0,
            "transitions": 0,
        }
        for m in METHODS
    }

    transition_rows = []
    step_rows = []

    for key, start_obs, goal_obs in grouped:
        eid, interaction_idx, object_handle = key

        latest = None
        majority_labels = []
        recency_history = []

        p_nodyn = 0.5
        p_uniform = 0.5
        p_cal = 0.5

        method_goal_first_correct_index = {m: None for m in METHODS}

        all_steps = [
            ("START", i, o)
            for i, o in enumerate(start_obs)
        ] + [
            ("GOAL", i, o)
            for i, o in enumerate(goal_obs)
        ]

        transition_gap = goal_obs[0]["transition_gap_actions"]
        delta = max(0.1, transition_gap / max(1e-9, time_scale))
        transitioned = False

        last_predictions = {}

        for phase, phase_idx, o in all_steps:
            if phase == "GOAL" and not transitioned:
                p_uniform = transition_predict(p_uniform, stay, delta)
                p_cal = transition_predict(p_cal, stay, delta)
                # Bayes-NoDynamics intentionally receives no transition prior.
                transitioned = True

            pred_obs = o["pred_state"]
            r_cal = reliability_for(o, calibration_table, global_r)

            if pred_obs in {"START", "GOAL"}:
                latest = pred_obs
                majority_labels.append(pred_obs)
                recency_history.append((pred_obs, r_cal))
                p_nodyn = bayes_update(p_nodyn, pred_obs, r_cal)
                p_uniform = bayes_update(p_uniform, pred_obs, global_r)
                p_cal = bayes_update(p_cal, pred_obs, r_cal)

            rec_state, rec_p = calibrated_recency_prediction(
                recency_history,
                recency_decay,
            )

            preds = {
                "Latest": (latest, 1.0 if latest == "GOAL" else 0.0 if latest == "START" else 0.5),
                "Majority": (
                    majority_prediction(majority_labels),
                    None,
                ),
                "Recency-Calibrated": (rec_state, rec_p),
                "Bayes-NoDynamics": (hard_state_from_p(p_nodyn), p_nodyn),
                "StateMem-Uniform": (hard_state_from_p(p_uniform), p_uniform),
                "StateMem-Calibrated": (hard_state_from_p(p_cal), p_cal),
            }

            gt = phase

            for m, (pred, prob_goal) in preds.items():
                a = aggregate[m]
                a["n"] += 1
                correct = pred == gt
                a["correct"] += int(correct)

                if phase == "START":
                    a["pre_n"] += 1
                    a["pre_correct"] += int(correct)
                else:
                    a["post_n"] += 1
                    a["post_correct"] += int(correct)

                    if phase_idx == 0:
                        a["immediate_n"] += 1
                        a["immediate_correct"] += int(correct)

                    if method_goal_first_correct_index[m] is None and pred == "GOAL":
                        method_goal_first_correct_index[m] = phase_idx

                if prob_goal is not None:
                    y = 1.0 if gt == "GOAL" else 0.0
                    # Binary Brier.
                    a["brier_sum"] += (prob_goal - y) ** 2
                    a["brier_n"] += 1

                last_predictions[m] = pred

                if write_steps:
                    step_rows.append({
                        "split": split_name,
                        "episode_id": eid,
                        "interaction_index": interaction_idx,
                        "object_handle": object_handle,
                        "phase": phase,
                        "phase_observation_index": phase_idx,
                        "method": m,
                        "gt_state": gt,
                        "prediction": pred,
                        "correct": int(correct),
                        "prob_goal": prob_goal,
                        "raw_vlm_state": pred_obs,
                        "raw_vlm_option": o["raw_label"],
                        "calibrated_reliability": r_cal,
                    })

        for m in METHODS:
            a = aggregate[m]
            a["transitions"] += 1

            final_pred = last_predictions.get(m)
            final_correct = final_pred == "GOAL"
            a["final_n"] += 1
            a["final_correct"] += int(final_correct)

            lag = method_goal_first_correct_index[m]
            if lag is None:
                a["misses"] += 1
            else:
                a["lag_sum"] += lag
                a["lag_n"] += 1

            transition_rows.append({
                "split": split_name,
                "episode_id": eid,
                "interaction_index": interaction_idx,
                "object_handle": object_handle,
                "method": m,
                "num_start_observations": len(start_obs),
                "num_goal_observations": len(goal_obs),
                "transition_gap_actions": transition_gap,
                "transition_delta_units": delta,
                "final_prediction": final_pred,
                "final_correct": int(final_correct),
                "goal_lag_observations": lag,
                "goal_missed": int(lag is None),
            })

    summary = {}
    for m, a in aggregate.items():
        summary[m] = {
            "accuracy": a["correct"] / a["n"] if a["n"] else None,
            "pre_accuracy": a["pre_correct"] / a["pre_n"] if a["pre_n"] else None,
            "post_accuracy": a["post_correct"] / a["post_n"] if a["post_n"] else None,
            "immediate_transition_accuracy": (
                a["immediate_correct"] / a["immediate_n"]
                if a["immediate_n"] else None
            ),
            "final_goal_accuracy": (
                a["final_correct"] / a["final_n"]
                if a["final_n"] else None
            ),
            "brier": (
                a["brier_sum"] / a["brier_n"]
                if a["brier_n"] else None
            ),
            "transition_lag_observations": (
                a["lag_sum"] / a["lag_n"]
                if a["lag_n"] else None
            ),
            "transition_miss_rate": (
                a["misses"] / a["transitions"]
                if a["transitions"] else None
            ),
            "num_steps": a["n"],
            "num_transitions": a["transitions"],
        }

    return summary, transition_rows, step_rows


# ---------------------------------------------------------------------
# CSV / JSON
# ---------------------------------------------------------------------

def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--full", action="store_true")

    ap.add_argument(
        "--metadata-root",
        default="findingdory_probe/metadata/findingdory",
    )
    ap.add_argument(
        "--transitions",
        default="findingdory_transitions/real_object_transitions.jsonl",
    )
    ap.add_argument("--cache-dir", default="findingdory_rgb_data")
    ap.add_argument("--out-dir", default="findingdory_rgb_statemem")
    ap.add_argument("--model", default=MODEL_ID)

    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--bundles-per-state", type=int, default=2)
    ap.add_argument("--images-per-bundle", type=int, default=2)

    ap.add_argument("--calibration-scene-cap", type=int, default=None)
    ap.add_argument("--tuning-scene-cap", type=int, default=None)
    ap.add_argument("--max-calibration-events", type=int, default=None)
    ap.add_argument("--max-tuning-events", type=int, default=None)
    ap.add_argument("--max-test-events", type=int, default=None)
    ap.add_argument("--test-episode-cap", type=int, default=None)

    args = ap.parse_args()

    if args.pilot and args.full:
        raise SystemExit("Choose either --pilot or --full, not both.")
    if not args.pilot and not args.full:
        args.pilot = True

    # Conservative pipeline pilot. Final run is larger but still caps train
    # events so calibration/tuning are computationally manageable on a laptop.
    if args.pilot:
        args.calibration_scene_cap = args.calibration_scene_cap or 8
        args.tuning_scene_cap = args.tuning_scene_cap or 8
        args.max_calibration_events = args.max_calibration_events or 60
        args.max_tuning_events = args.max_tuning_events or 60
        args.max_test_events = args.max_test_events or 60
        args.test_episode_cap = args.test_episode_cap or 15
    else:
        args.max_calibration_events = args.max_calibration_events or 300
        args.max_tuning_events = args.max_tuning_events or 300
        # None = all usable official validation events.
        if args.max_test_events is not None and args.max_test_events <= 0:
            args.max_test_events = None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "vlm_cache.jsonl"

    print("=" * 100)
    print("FINDINGDORY REAL-RGB -> STATEMEM INTEGRATION")
    print("=" * 100)
    print("Mode:", "PILOT" if args.pilot else "FULL")
    print("Model:", args.model)

    native, native_verification = build_native_events(
        Path(args.metadata_root),
        Path(args.transitions),
    )

    supervision, supervision_stats = load_hf_supervision()

    eligible, eligibility_counts = attach_rgb_supervision(
        native,
        supervision,
        Path(args.cache_dir),
    )

    cal_scenes, tuning_scenes = scene_split_train(
        eligible["train"],
        args.seed,
    )
    cal_scenes = cap_scenes(
        cal_scenes,
        args.calibration_scene_cap,
        args.seed,
        "calibration_scene_cap",
    )
    tuning_scenes = cap_scenes(
        tuning_scenes,
        args.tuning_scene_cap,
        args.seed,
        "tuning_scene_cap",
    )

    cal_events = round_robin_events(
        eligible["train"],
        allowed_scenes=cal_scenes,
        cap=args.max_calibration_events,
        seed=args.seed,
        tag="calibration",
    )
    tune_events = round_robin_events(
        eligible["train"],
        allowed_scenes=tuning_scenes,
        cap=args.max_tuning_events,
        seed=args.seed,
        tag="tuning",
    )

    # Official validation remains test-only. Optionally cap episodes/events for
    # the pipeline pilot, never for hyperparameter selection.
    test_eps = eligible["val"]
    if args.test_episode_cap:
        ids = sorted(test_eps, key=lambda x: int(x))
        ids = ids[:args.test_episode_cap]
        test_eps = {k: test_eps[k] for k in ids}

    test_events = round_robin_events(
        test_eps,
        allowed_scenes=None,
        cap=args.max_test_events,
        seed=args.seed,
        tag="test",
    )

    train_timing = [
        ev["local_desnap_action_index"]
        for ev in cal_events + tune_events
        if ev["local_desnap_action_index"] > 0
    ]
    time_scale = statistics.median(train_timing) if train_timing else 1.0

    config = {
        "mode": "pilot" if args.pilot else "full",
        "model_id": args.model,
        "seed": args.seed,
        "bundles_per_state": args.bundles_per_state,
        "images_per_bundle": args.images_per_bundle,
        "calibration_scene_count": len(cal_scenes),
        "tuning_scene_count": len(tuning_scenes),
        "calibration_event_count": len(cal_events),
        "tuning_event_count": len(tune_events),
        "test_event_count": len(test_events),
        "test_episode_count": len(set(e["episode_id"] for e in test_events)),
        "native_time_scale_median_pick_to_place_actions": time_scale,
        "stay_candidates": STAY_CANDIDATES,
        "recency_decay_candidates": RECENCY_DECAYS,
        "oracle_frame_selection": True,
        "frame_supervision": "FindingDory task_58/task_59 ordered answer-frame groups",
        "validation_used_for_tuning": False,
    }
    (out_dir / "experiment_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    protocol = {
        "native_verification": native_verification,
        "supervision_stats": supervision_stats,
        "eligibility_counts": eligibility_counts,
        "train_calibration_tuning_scene_overlap": len(cal_scenes & tuning_scenes),
        "official_train_val_scene_overlap": len(
            set(ep["scene_id"] for ep in eligible["train"].values())
            & set(ep["scene_id"] for ep in eligible["val"].values())
        ),
        "config": config,
    }
    (out_dir / "protocol_report.json").write_text(
        json.dumps(protocol, indent=2),
        encoding="utf-8",
    )

    print()
    print("Selected events:")
    print("  calibration:", len(cal_events))
    print("  tuning:     ", len(tune_events))
    print("  test:       ", len(test_events))
    print("Scene overlap calibration/tuning:", len(cal_scenes & tuning_scenes))
    print(
        "Official train/val scene overlap:",
        protocol["official_train_val_scene_overlap"],
    )
    print("Native timing scale:", time_scale)

    if not cal_events or not tune_events or not test_events:
        raise SystemExit(
            "A required split has zero usable events. "
            "Inspect protocol_report.json."
        )

    model, processor, vlm_config = load_vlm(args.model)

    print()
    print("Generating / loading calibration RGB observations...")
    cal_obs = generate_observations(
        cal_events,
        "train_calibration",
        out_dir,
        cache_path,
        model,
        processor,
        vlm_config,
        args.model,
        args.seed,
        args.bundles_per_state,
        args.images_per_bundle,
    )

    print()
    print("Generating / loading tuning RGB observations...")
    tune_obs = generate_observations(
        tune_events,
        "train_tuning",
        out_dir,
        cache_path,
        model,
        processor,
        vlm_config,
        args.model,
        args.seed,
        args.bundles_per_state,
        args.images_per_bundle,
    )

    print()
    print("Generating / loading official validation RGB observations...")
    test_obs = generate_observations(
        test_events,
        "official_val",
        out_dir,
        cache_path,
        model,
        processor,
        vlm_config,
        args.model,
        args.seed,
        args.bundles_per_state,
        args.images_per_bundle,
    )

    calibration_report, calibration_table = fit_calibration(cal_obs)
    (out_dir / "calibration_table.json").write_text(
        json.dumps(calibration_report, indent=2),
        encoding="utf-8",
    )

    global_r = calibration_report["global_reliability"]

    # Tune recency on train tuning scenes only.
    recency_rows = []
    best_recency = None
    best_recency_key = None

    for decay in RECENCY_DECAYS:
        summary, _, _ = evaluate(
            tune_obs,
            calibration_table,
            global_r,
            stay=0.80,  # irrelevant for Recency metric selection
            recency_decay=decay,
            time_scale=time_scale,
            split_name="train_tuning",
        )
        m = summary["Recency-Calibrated"]
        row = {
            "decay": decay,
            **m,
        }
        recency_rows.append(row)

        key = (
            m["post_accuracy"] if m["post_accuracy"] is not None else -1,
            m["immediate_transition_accuracy"] if m["immediate_transition_accuracy"] is not None else -1,
            m["final_goal_accuracy"] if m["final_goal_accuracy"] is not None else -1,
        )
        if best_recency_key is None or key > best_recency_key:
            best_recency_key = key
            best_recency = decay

    write_csv(out_dir / "tuning_recency_sweep.csv", recency_rows)

    # Tune StateMem stay on train tuning scenes only.
    stay_rows = []
    best_stay = None
    best_stay_key = None

    for stay in STAY_CANDIDATES:
        summary, _, _ = evaluate(
            tune_obs,
            calibration_table,
            global_r,
            stay=stay,
            recency_decay=best_recency,
            time_scale=time_scale,
            split_name="train_tuning",
        )
        m = summary["StateMem-Calibrated"]
        row = {
            "stay": stay,
            **m,
        }
        stay_rows.append(row)

        key = (
            m["post_accuracy"] if m["post_accuracy"] is not None else -1,
            m["immediate_transition_accuracy"] if m["immediate_transition_accuracy"] is not None else -1,
            m["final_goal_accuracy"] if m["final_goal_accuracy"] is not None else -1,
            -(m["brier"] if m["brier"] is not None else 1e9),
        )
        if best_stay_key is None or key > best_stay_key:
            best_stay_key = key
            best_stay = stay

    write_csv(out_dir / "tuning_stay_sweep.csv", stay_rows)

    print()
    print("Frozen tuning choices:")
    print("  recency decay:", best_recency)
    print("  StateMem stay:", best_stay)
    print("  global RGB reliability:", round(global_r, 4))

    test_summary, transition_rows, step_rows = evaluate(
        test_obs,
        calibration_table,
        global_r,
        stay=best_stay,
        recency_decay=best_recency,
        time_scale=time_scale,
        split_name="official_val",
        write_steps=True,
    )

    test_overall = {
        "selected_stay": best_stay,
        "selected_recency_decay": best_recency,
        "global_calibration_reliability": global_r,
        "official_val_events": len(test_events),
        "methods": test_summary,
        "important_limitation": (
            "Real RGB/VLM evidence and train-only calibration are used, but "
            "FindingDory answer-frame annotations select informative frames."
        ),
    }

    (out_dir / "test_overall.json").write_text(
        json.dumps(test_overall, indent=2),
        encoding="utf-8",
    )
    write_csv(
        out_dir / "test_transition_results.csv",
        transition_rows,
    )
    write_csv(
        out_dir / "test_step_results.csv",
        step_rows,
    )

    print()
    print("=" * 100)
    print("REAL-RGB -> STATEMEM RESULT")
    print("=" * 100)

    for m in METHODS:
        x = test_summary[m]
        print(
            f"{m:<22} "
            f"overall={100*x['accuracy']:.2f}% "
            f"post={100*x['post_accuracy']:.2f}% "
            f"immediate={100*x['immediate_transition_accuracy']:.2f}% "
            f"final={100*x['final_goal_accuracy']:.2f}% "
            f"miss={100*x['transition_miss_rate']:.2f}%"
        )

    print()
    print("Outputs:")
    for name in (
        "experiment_config.json",
        "protocol_report.json",
        "calibration_table.json",
        "tuning_recency_sweep.csv",
        "tuning_stay_sweep.csv",
        "test_overall.json",
        "test_transition_results.csv",
        "test_step_results.csv",
        "vlm_cache.jsonl",
    ):
        print(" ", out_dir / name)

    print()
    if args.pilot:
        print(
            "This was the PIPELINE PILOT. If the protocol looks clean and "
            "StateMem behaves sensibly, rerun with:\n"
            "  python3 findingdory_rgb_statemem_integration.py --full\n"
            "The existing VLM cache is reused automatically."
        )
    else:
        print(
            "Upload test_overall.json, protocol_report.json, "
            "calibration_table.json, tuning_stay_sweep.csv, "
            "tuning_recency_sweep.csv, and test_transition_results.csv."
        )


if __name__ == "__main__":
    main()
