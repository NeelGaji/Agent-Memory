#!/usr/bin/env python3
"""
FindingDory RGB/VLM pilot v3 — official ground-truth frame supervision
=====================================================================

Why this version exists
-----------------------
The previous pilot proved the 4-bit MLX VLM can run, but it mapped native
Habitat action indices linearly onto the 96-frame video. The resulting PRE
frames were often not actually pre-transition views.

This version removes that approximation.

FindingDory's official subsampled dataset stores answer frame indices for its
tasks. We use:
  task_32 = nth receptacle the robot PICKED an object from  -> START evidence
  task_33 = nth receptacle the robot PLACED an object on    -> GOAL evidence

Therefore frame selection is now native FindingDory supervision.

This is still a perception-component pilot, not the final StateMem run.
It tests whether a local VLM can recognize START-vs-GOAL receptacle evidence
from real egocentric RGB frames.

Confidence is NOT self-reported by the VLM. It is agreement across several
officially-correct RGB frames for the same state.

Expected existing assets
------------------------
findingdory_probe/metadata/findingdory/val/episodes.json.gz
findingdory_transitions/real_object_transitions.jsonl
findingdory_rgb_data/   # videos from the earlier pilot

Run
---
cd ~/Downloads
python3 findingdory_rgb_vlm_pilot_v3.py --pilot-episodes 3

Outputs
-------
findingdory_rgb_pilot_v3/
  setup_report.json
  frames/
  frame_predictions.jsonl
  event_predictions.jsonl
  rgb_vlm_report.json
"""

from __future__ import annotations

import argparse
import ast
import gzip
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

DATASET_REPO = "yali30/findingdory-subsampled-96"
MODEL_ID = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"

ORDINALS = {
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


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------

def require_imports():
    missing = []

    try:
        import cv2  # noqa
    except Exception:
        missing.append("opencv-python")

    try:
        import datasets  # noqa
    except Exception:
        missing.append("datasets")

    try:
        import mlx_vlm  # noqa
    except Exception:
        missing.append("mlx-vlm")

    if missing:
        raise SystemExit(
            "Missing packages: " + ", ".join(missing) + "\n\n"
            "Install with:\n"
            "  python3 -m pip install -U datasets opencv-python mlx-vlm\n"
        )


def load_json_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def parse_answer(x):
    if isinstance(x, list):
        return x

    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return None

    return None


def flatten_answer_frames(answer):
    """
    For task_32/task_33, FindingDory normally stores one answer group:
        [[53, 54, 55, ...]]
    """
    ans = parse_answer(answer)

    if not isinstance(ans, list) or not ans:
        return []

    if isinstance(ans[0], list):
        vals = ans[0]
    else:
        vals = ans

    out = []
    for x in vals:
        if isinstance(x, (int, float)):
            x = int(x)
            if 0 <= x <= 95:
                out.append(x)

    return sorted(set(out))


def ordinal_from_question(question: str):
    q = question.lower()

    for word, idx in ORDINALS.items():
        if re.search(rf"\b{re.escape(word)}\b", q):
            return idx

    return None


def sample_representative_frames(indices, k=3):
    xs = sorted(set(indices))

    if len(xs) <= k:
        return xs

    # Interior quantiles are less likely to land on transient edge frames.
    qs = [(i + 1) / (k + 1) for i in range(k)]
    chosen = []

    for q in qs:
        j = round(q * (len(xs) - 1))
        chosen.append(xs[j])

    return sorted(set(chosen))


def normalize_category(x):
    return str(x or "").replace("_", " ").strip().lower()


# ---------------------------------------------------------------------
# Native FindingDory metadata
# ---------------------------------------------------------------------

def load_transitions(path: Path):
    out = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            r = json.loads(line)

            if r.get("split") != "val":
                continue

            out[
                (
                    str(r["episode_id"]),
                    r["object_handle"],
                )
            ] = r

    return out


def candidate_category_map(ep):
    """
    Map exact object config filename to FindingDory's human-readable category.
    """
    out = {}

    for key in (
        "candidate_objects",
        "candidate_objects_noninteracted",
    ):
        for x in ep.get(key) or []:
            name = x.get("object_name")
            category = x.get("object_category")

            if name and category:
                out[name] = category

    return out


def build_val_events(metadata_root: Path, transitions_path: Path):
    data = load_json_gz(
        metadata_root / "val" / "episodes.json.gz"
    )
    transitions = load_transitions(transitions_path)

    episodes = {}
    warnings = []

    for ep in data["episodes"]:
        eid = str(ep["episode_id"])
        target_handles = list((ep.get("targets") or {}).keys())
        category_map = candidate_category_map(ep)

        events = []

        for i, handle in enumerate(target_handles):
            tr = transitions.get((eid, handle))

            if tr is None:
                warnings.append(
                    {
                        "episode_id": eid,
                        "object_handle": handle,
                        "type": "missing_transition",
                    }
                )
                continue

            object_category = category_map.get(
                tr.get("object_config")
            )

            if not object_category:
                object_category = (
                    str(tr.get("object_config", "object"))
                    .replace(".object_config.json", "")
                    .replace("_", " ")
                )

            events.append(
                {
                    **tr,
                    "interaction_index": i,
                    "object_category": normalize_category(
                        object_category
                    ),
                }
            )

        if len(events) == len(target_handles) and events:
            episodes[eid] = {
                "episode_id": eid,
                "scene_id": ep["scene_id"],
                "events": events,
            }

    return episodes, warnings


# ---------------------------------------------------------------------
# Official FindingDory frame supervision
# ---------------------------------------------------------------------

def load_official_frame_labels():
    from datasets import load_dataset

    print("Loading official FindingDory validation metadata...")

    ds = load_dataset(
        DATASET_REPO,
        split="validation",
    )

    # episode -> interaction index -> official answer frame indices
    result = defaultdict(
        lambda: {
            "start": defaultdict(set),
            "goal": defaultdict(set),
            "video": None,
        }
    )

    stats = Counter()

    for row in ds:
        raw_ep = str(row["ep_id"])
        eid = raw_ep[3:] if raw_ep.startswith("ep_") else raw_ep

        result[eid]["video"] = row["video"]

        task_id = str(row["task_id"])

        if task_id not in {"task_32", "task_33"}:
            continue

        interaction_idx = ordinal_from_question(
            str(row["question"])
        )

        if interaction_idx is None:
            stats["ordinal_parse_failed"] += 1
            continue

        frames = flatten_answer_frames(row["answer"])

        if not frames:
            stats["empty_or_invalid_answer"] += 1
            continue

        side = "start" if task_id == "task_32" else "goal"
        result[eid][side][interaction_idx].update(frames)

        stats[f"{side}_rows"] += 1

    # Convert sets for JSON-friendly deterministic behavior.
    final = {}

    for eid, x in result.items():
        final[eid] = {
            "video": x["video"],
            "start": {
                int(k): sorted(v)
                for k, v in x["start"].items()
            },
            "goal": {
                int(k): sorted(v)
                for k, v in x["goal"].items()
            },
        }

    return final, dict(stats)


# ---------------------------------------------------------------------
# Video IO
# ---------------------------------------------------------------------

def build_video_index(cache_dir: Path):
    out = {}

    for p in cache_dir.rglob("*.mp4"):
        out[p.stem] = p

    return out


def read_frame(video_path: Path, frame_idx: int):
    import cv2

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(
            f"Could not decode frame {frame_idx} "
            f"from {video_path}"
        )

    return cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )


def save_rgb(path: Path, rgb):
    import cv2

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(path),
        cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2BGR,
        ),
    )


# ---------------------------------------------------------------------
# MLX VLM
# ---------------------------------------------------------------------

def load_vlm(model_id):
    from mlx_vlm import load
    from mlx_vlm.utils import load_config

    print()
    print(f"Loading local VLM: {model_id}")

    model, processor = load(model_id)
    config = load_config(model_id)

    try:
        ip = processor.image_processor

        if hasattr(ip, "max_pixels"):
            ip.max_pixels = 1280 * 28 * 28

    except Exception:
        pass

    return model, processor, config


def generation_text(result):
    if isinstance(result, str):
        return result

    text = getattr(result, "text", None)

    if isinstance(text, str):
        return text

    return str(result)


def category_aliases(category):
    c = normalize_category(category)

    aliases = {c}

    aliases.add(c.replace(" ", "_"))
    aliases.add(c.replace(" ", "-"))

    return {
        a for a in aliases
        if a
    }


def parse_state_output(
    text,
    start_category,
    goal_category,
):
    """
    Robustly accept:
      START
      GOAL
      "cabinet"
      "chest_of_drawers"
      NOT_VISIBLE
      UNCERTAIN
    """
    t = text.strip()
    tl = t.lower()

    if re.search(
        r"\bnot[\s_-]*visible\b",
        tl,
    ):
        return "NOT_VISIBLE"

    if any(
        x in tl
        for x in (
            "uncertain",
            "unsure",
            "cannot tell",
            "can't tell",
        )
    ):
        return "UNCERTAIN"

    # Explicit state labels have priority.
    if re.search(r"\bstart\b", tl):
        return "START"

    if re.search(r"\bgoal\b", tl):
        return "GOAL"

    start_hits = any(
        alias in tl
        for alias in category_aliases(start_category)
    )

    goal_hits = any(
        alias in tl
        for alias in category_aliases(goal_category)
    )

    if start_hits and not goal_hits:
        return "START"

    if goal_hits and not start_hits:
        return "GOAL"

    if start_hits and goal_hits:
        return "UNCERTAIN"

    return "PARSE_ERROR"


def classify_frame(
    model,
    processor,
    config,
    image_path,
    start_category,
    goal_category,
):
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    start_h = normalize_category(start_category)
    goal_h = normalize_category(goal_category)

    prompt = (
        "Look only at this egocentric robot image. "
        f"Option START corresponds to a {start_h}. "
        f"Option GOAL corresponds to a {goal_h}. "
        "Which option is visually better supported by the main "
        "receptacle/location shown in the image? "
        "Reply with exactly one of: START, GOAL, "
        "NOT_VISIBLE, UNCERTAIN."
    )

    formatted = apply_chat_template(
        processor,
        config,
        prompt,
        num_images=1,
    )

    result = generate(
        model,
        processor,
        formatted,
        [str(image_path)],
        max_tokens=12,
        temp=0.0,
        verbose=False,
    )

    raw = generation_text(result).strip()

    return (
        parse_state_output(
            raw,
            start_category,
            goal_category,
        ),
        raw,
    )


def sanity_check(
    model,
    processor,
    config,
    image_path,
):
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    prompt = "Describe this image in one short sentence."

    formatted = apply_chat_template(
        processor,
        config,
        prompt,
        num_images=1,
    )

    result = generate(
        model,
        processor,
        formatted,
        [str(image_path)],
        max_tokens=32,
        temp=0.0,
        verbose=False,
    )

    text = generation_text(result).strip()

    print()
    print("VLM sanity-check output:")
    print(text[:500])

    alnum = sum(c.isalnum() for c in text)

    if alnum < 3:
        raise RuntimeError(
            "VLM sanity check failed: degenerate generation."
        )

    print("VLM sanity check: PASS")


# ---------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------

def aggregate_state_frames(rows, expected_state):
    state_rows = [
        r
        for r in rows
        if r["predicted_label"] in {"START", "GOAL"}
    ]

    if not state_rows:
        prediction = "NO_STATE_JUDGMENT"
        agreement = 0.0
    else:
        c = Counter(
            r["predicted_label"]
            for r in state_rows
        )
        prediction, votes = c.most_common(1)[0]

        # Agreement over ALL sampled frames. An uncertain frame lowers
        # confidence rather than being silently discarded.
        agreement = votes / len(rows)

    return {
        "expected_state": expected_state,
        "predicted_state": prediction,
        "correct": prediction == expected_state,
        "agreement_confidence": agreement,
        "state_judgment_coverage": (
            len(state_rows) / len(rows)
            if rows
            else 0.0
        ),
        "num_frames": len(rows),
        "num_state_judgments": len(state_rows),
        "label_counts": dict(
            Counter(
                r["predicted_label"]
                for r in rows
            )
        ),
    }


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
        "--cache-dir",
        default="findingdory_rgb_data",
    )
    ap.add_argument(
        "--out-dir",
        default="findingdory_rgb_pilot_v3",
    )
    ap.add_argument(
        "--pilot-episodes",
        type=int,
        default=3,
    )
    ap.add_argument(
        "--frames-per-state",
        type=int,
        default=3,
    )
    ap.add_argument(
        "--model",
        default=MODEL_ID,
    )
    ap.add_argument(
        "--include-same-category",
        action="store_true",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=20260916,
    )

    args = ap.parse_args()

    require_imports()

    metadata_root = Path(args.metadata_root)
    transitions_path = Path(args.transitions)
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    frames_dir = out_dir / "frames"

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    frames_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 96)
    print(
        "FINDINGDORY RGB/VLM PILOT V3 "
        "— OFFICIAL GROUND-TRUTH FRAMES"
    )
    print("=" * 96)

    events_by_ep, metadata_warnings = build_val_events(
        metadata_root,
        transitions_path,
    )

    official_frames, official_stats = (
        load_official_frame_labels()
    )

    video_index = build_video_index(cache_dir)

    eligible = []

    event_counts = Counter()

    for eid, ep in events_by_ep.items():
        official = official_frames.get(eid)

        if official is None:
            event_counts["missing_official_episode"] += 1
            continue

        video_path = video_index.get(f"ep_{eid}")

        if video_path is None:
            event_counts["missing_video"] += 1
            continue

        usable_events = []

        for ev in ep["events"]:
            i = ev["interaction_index"]

            same_category = (
                normalize_category(
                    ev.get("start_receptacle_category")
                )
                == normalize_category(
                    ev.get("goal_receptacle_category")
                )
            )

            if (
                same_category
                and not args.include_same_category
            ):
                event_counts["same_category_skipped"] += 1
                continue

            start_frames = official["start"].get(i, [])
            goal_frames = official["goal"].get(i, [])

            if not start_frames:
                event_counts["missing_start_frames"] += 1

            if not goal_frames:
                event_counts["missing_goal_frames"] += 1

            if not start_frames or not goal_frames:
                continue

            usable_events.append(
                {
                    **ev,
                    "official_start_frames": start_frames,
                    "official_goal_frames": goal_frames,
                }
            )
            event_counts["usable"] += 1

        if usable_events:
            eligible.append(
                {
                    **ep,
                    "events": usable_events,
                    "video_path": str(video_path),
                }
            )

    if not eligible:
        setup_report = {
            "dataset_repo": DATASET_REPO,
            "official_stats": official_stats,
            "event_counts": dict(event_counts),
            "metadata_warnings": metadata_warnings[:20],
        }

        (out_dir / "setup_report.json").write_text(
            json.dumps(
                setup_report,
                indent=2,
            ),
            encoding="utf-8",
        )

        raise SystemExit(
            "No eligible episodes. Upload setup_report.json."
        )

    eligible.sort(
        key=lambda ep: (
            -len(ep["events"]),
            int(ep["episode_id"]),
        )
    )

    rng = random.Random(args.seed)

    pool = eligible[
        : max(
            args.pilot_episodes * 3,
            args.pilot_episodes,
        )
    ]
    rng.shuffle(pool)

    selected = pool[: args.pilot_episodes]

    setup_report = {
        "dataset_repo": DATASET_REPO,
        "model_id": args.model,
        "native_validation_episode_count": len(
            events_by_ep
        ),
        "official_task_frame_stats": official_stats,
        "eligible_episode_count": len(eligible),
        "selected_episode_ids": [
            ep["episode_id"]
            for ep in selected
        ],
        "metadata_warning_count": len(
            metadata_warnings
        ),
        "event_counts": dict(event_counts),
        "frames_per_state": args.frames_per_state,
        "same_category_included": (
            args.include_same_category
        ),
        "frame_source": (
            "Official FindingDory 96-frame answer indices: "
            "task_32 for nth picked-from receptacle (START) "
            "and task_33 for nth placed-on receptacle (GOAL)."
        ),
        "confidence_source": (
            "Agreement across multiple real RGB frames; "
            "no VLM self-reported confidence."
        ),
        "important_limitation": (
            "This pilot measures RGB recognition of the native "
            "start-vs-goal receptacle evidence. It is not yet the "
            "full StateMem temporal belief-reconciliation experiment."
        ),
    }

    (out_dir / "setup_report.json").write_text(
        json.dumps(
            setup_report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Eligible validation episodes: {len(eligible)}"
    )
    print(
        "Selected pilot episodes: "
        + ", ".join(
            ep["episode_id"]
            for ep in selected
        )
    )
    print(
        f"Usable event mappings: {event_counts['usable']}"
    )

    model, processor, config = load_vlm(
        args.model
    )

    # Sanity-check one official frame.
    first_ep = selected[0]
    first_ev = first_ep["events"][0]

    sanity_frames = sample_representative_frames(
        first_ev["official_start_frames"],
        args.frames_per_state,
    )

    sanity_idx = sanity_frames[0]
    sanity_rgb = read_frame(
        Path(first_ep["video_path"]),
        sanity_idx,
    )
    sanity_path = (
        frames_dir / "SANITY_CHECK.jpg"
    )

    save_rgb(
        sanity_path,
        sanity_rgb,
    )

    sanity_check(
        model,
        processor,
        config,
        sanity_path,
    )

    frame_rows = []
    event_rows = []

    for ep_n, ep in enumerate(selected, 1):
        eid = ep["episode_id"]
        video_path = Path(ep["video_path"])

        print()
        print(
            f"[episode {ep_n}/{len(selected)}] "
            f"id={eid} "
            f"usable_events={len(ep['events'])}"
        )

        for ev in ep["events"]:
            for expected, frame_key in (
                ("START", "official_start_frames"),
                ("GOAL", "official_goal_frames"),
            ):
                sampled = sample_representative_frames(
                    ev[frame_key],
                    args.frames_per_state,
                )

                local_rows = []

                for frame_idx in sampled:
                    rgb = read_frame(
                        video_path,
                        frame_idx,
                    )

                    img_name = (
                        f"ep_{eid}_"
                        f"obj_{ev['interaction_index']:02d}_"
                        f"{expected.lower()}_"
                        f"f{frame_idx:02d}.jpg"
                    )

                    img_path = (
                        frames_dir / img_name
                    )

                    save_rgb(
                        img_path,
                        rgb,
                    )

                    pred, raw = classify_frame(
                        model=model,
                        processor=processor,
                        config=config,
                        image_path=img_path,
                        start_category=ev[
                            "start_receptacle_category"
                        ],
                        goal_category=ev[
                            "goal_receptacle_category"
                        ],
                    )

                    row = {
                        "episode_id": eid,
                        "interaction_index": ev[
                            "interaction_index"
                        ],
                        "object_handle": ev[
                            "object_handle"
                        ],
                        "object_category": ev[
                            "object_category"
                        ],
                        "start_receptacle_category": ev[
                            "start_receptacle_category"
                        ],
                        "goal_receptacle_category": ev[
                            "goal_receptacle_category"
                        ],
                        "expected_state": expected,
                        "frame_idx": frame_idx,
                        "predicted_label": pred,
                        "raw_output": raw,
                        "frame_path": str(
                            img_path
                        ),
                    }

                    frame_rows.append(row)
                    local_rows.append(row)

                    mark = (
                        "✓"
                        if pred == expected
                        else "·"
                    )

                    print(
                        f"  {mark} "
                        f"obj={ev['interaction_index']:>2} "
                        f"{expected:<5} "
                        f"f={frame_idx:>2} "
                        f"{normalize_category(ev['start_receptacle_category'])[:12]}"
                        f" -> "
                        f"{normalize_category(ev['goal_receptacle_category'])[:12]} "
                        f"pred={pred}"
                    )

                agg = aggregate_state_frames(
                    local_rows,
                    expected,
                )

                event_rows.append(
                    {
                        "episode_id": eid,
                        "interaction_index": ev[
                            "interaction_index"
                        ],
                        "object_handle": ev[
                            "object_handle"
                        ],
                        "object_category": ev[
                            "object_category"
                        ],
                        "start_receptacle_category": ev[
                            "start_receptacle_category"
                        ],
                        "goal_receptacle_category": ev[
                            "goal_receptacle_category"
                        ],
                        **agg,
                    }
                )

    frame_path = (
        out_dir / "frame_predictions.jsonl"
    )

    with frame_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in frame_rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    event_path = (
        out_dir / "event_predictions.jsonl"
    )

    with event_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in event_rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------
    frame_labels = Counter(
        r["predicted_label"]
        for r in frame_rows
    )

    state_frame_rows = [
        r
        for r in frame_rows
        if r["predicted_label"]
        in {"START", "GOAL"}
    ]

    judged_events = [
        r
        for r in event_rows
        if r["predicted_state"]
        in {"START", "GOAL"}
    ]

    correct_events = [
        r
        for r in judged_events
        if r["correct"]
    ]

    by_state = {}

    for state in ("START", "GOAL"):
        rr = [
            r
            for r in event_rows
            if r["expected_state"] == state
        ]

        jj = [
            r
            for r in rr
            if r["predicted_state"]
            in {"START", "GOAL"}
        ]

        by_state[state] = {
            "n_state_events": len(rr),
            "event_state_judgment_coverage": (
                len(jj) / len(rr)
                if rr
                else None
            ),
            "event_accuracy_when_judged": (
                sum(r["correct"] for r in jj)
                / len(jj)
                if jj
                else None
            ),
            "mean_agreement_confidence": (
                sum(
                    r["agreement_confidence"]
                    for r in rr
                )
                / len(rr)
                if rr
                else None
            ),
        }

    report = {
        "num_frame_predictions": len(frame_rows),
        "num_state_events": len(event_rows),
        "frame_label_counts": dict(
            frame_labels
        ),
        "frame_state_judgment_coverage": (
            len(state_frame_rows)
            / len(frame_rows)
            if frame_rows
            else None
        ),
        "event_state_judgment_coverage": (
            len(judged_events)
            / len(event_rows)
            if event_rows
            else None
        ),
        "event_accuracy_when_judged": (
            len(correct_events)
            / len(judged_events)
            if judged_events
            else None
        ),
        "all_event_accuracy": (
            len(correct_events)
            / len(event_rows)
            if event_rows
            else None
        ),
        "by_expected_state": by_state,
        "selected_episode_ids": [
            ep["episode_id"]
            for ep in selected
        ],
        "model_id": args.model,
        "frame_supervision": (
            "Official FindingDory task_32/task_33 "
            "answer-frame indices."
        ),
        "confidence_definition": (
            "Fraction of sampled official RGB frames "
            "voting for the aggregated state."
        ),
        "interpretation_note": (
            "This is a real-RGB perception pilot. "
            "If START and GOAL performance are both healthy, "
            "the next experiment feeds these observations into "
            "StateMem with train-scene calibration and official "
            "validation-scene evaluation."
        ),
    }

    report_path = (
        out_dir / "rgb_vlm_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 96)
    print("RGB/VLM PILOT V3 COMPLETE")
    print("=" * 96)

    print(
        "Frame state-judgment coverage: "
        f"{100 * report['frame_state_judgment_coverage']:.2f}%"
        if report["frame_state_judgment_coverage"]
        is not None
        else "Frame state-judgment coverage: n/a"
    )

    print(
        "Event state-judgment coverage: "
        f"{100 * report['event_state_judgment_coverage']:.2f}%"
        if report["event_state_judgment_coverage"]
        is not None
        else "Event state-judgment coverage: n/a"
    )

    print(
        "Event accuracy when judged: "
        f"{100 * report['event_accuracy_when_judged']:.2f}%"
        if report["event_accuracy_when_judged"]
        is not None
        else "Event accuracy when judged: n/a"
    )

    print(
        "All-event accuracy: "
        f"{100 * report['all_event_accuracy']:.2f}%"
        if report["all_event_accuracy"]
        is not None
        else "All-event accuracy: n/a"
    )

    for state in ("START", "GOAL"):
        x = report["by_expected_state"][state]

        cov = x[
            "event_state_judgment_coverage"
        ]
        acc = x[
            "event_accuracy_when_judged"
        ]

        cov_s = (
            f"{100 * cov:.2f}%"
            if cov is not None
            else "n/a"
        )
        acc_s = (
            f"{100 * acc:.2f}%"
            if acc is not None
            else "n/a"
        )

        print(
            f"{state}: coverage={cov_s} "
            f"accuracy={acc_s}"
        )

    print()
    print(
        f"Setup:       {out_dir / 'setup_report.json'}"
    )
    print(
        f"Report:      {report_path}"
    )
    print(
        f"Frames:      {frame_path}"
    )
    print(
        f"Events:      {event_path}"
    )
    print()
    print(
        "Upload setup_report.json, rgb_vlm_report.json, "
        "and event_predictions.jsonl."
    )


if __name__ == "__main__":
    main()
