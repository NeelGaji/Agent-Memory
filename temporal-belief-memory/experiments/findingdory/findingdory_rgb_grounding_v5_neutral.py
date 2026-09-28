#!/usr/bin/env python3
"""
FindingDory RGB grounding v5 — neutral-label multi-frame ablation
================================================================

Purpose
-------
Test whether the remaining START/GOAL asymmetry comes from the *label/prompt
formulation* rather than model capacity.

Compared with v4:
- same native FindingDory task_58/task_59 supervision
- same changed-category transitions
- Qwen2.5-VL-7B-4bit by default
- TWO official frames are shown jointly when available
- the VLM never sees the words START or GOAL
- candidate receptacle categories are randomly assigned to neutral labels A/B
- A/B assignment is deterministic per event, so results are reproducible
- one VLM call per state-event rather than one call per frame

If this makes START and GOAL much more balanced, the old START/GOAL wording was
part of the problem.

Run
---
cd ~/Downloads
python3 findingdory_rgb_grounding_v5_neutral.py --pilot-episodes 5

Outputs
-------
findingdory_rgb_grounding_v5_neutral/
  setup_report.json
  event_predictions.jsonl
  rgb_grounding_report.json
"""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import random
import re
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


def stable_bit(*parts):
    s = "|".join(map(str, parts)).encode("utf-8")
    return hashlib.sha256(s).digest()[0] & 1


def normalize_category(x):
    return str(x or "").replace("_", " ").strip().lower()


def load_json_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def load_transitions(path):
    out = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("split") == "val":
                out[(str(r["episode_id"]), r["object_handle"])] = r
    return out


def candidate_category_map(ep):
    out = {}
    for key in ("candidate_objects", "candidate_objects_noninteracted"):
        for x in ep.get(key) or []:
            if x.get("object_name") and x.get("object_category"):
                out[x["object_name"]] = x["object_category"]
    return out


def build_val_events(metadata_root, transitions_path):
    eps = load_json_gz(
        Path(metadata_root) / "val" / "episodes.json.gz"
    )["episodes"]
    transitions = load_transitions(transitions_path)

    result = {}

    for ep in eps:
        eid = str(ep["episode_id"])
        handles = list((ep.get("targets") or {}).keys())
        cmap = candidate_category_map(ep)
        events = []

        for i, handle in enumerate(handles):
            tr = transitions.get((eid, handle))
            if tr is None:
                continue

            obj_cat = cmap.get(
                tr.get("object_config"),
                str(tr.get("object_config", "object"))
                .replace(".object_config.json", "")
                .replace("_", " "),
            )

            events.append({
                **tr,
                "interaction_index": i,
                "object_category": normalize_category(obj_cat),
            })

        if len(events) == len(handles) and events:
            result[eid] = {
                "episode_id": eid,
                "scene_id": ep["scene_id"],
                "events": events,
            }

    return result


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
        question.lower(),
    )
    if not m:
        return None

    words = re.findall(
        r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh)\b",
        m.group(1),
    )
    return [ORDINAL_TO_INDEX[w] for w in words] if words else None


def clean_group(group):
    if not isinstance(group, list):
        return []
    return sorted(set(
        int(x)
        for x in group
        if isinstance(x, (int, float)) and 0 <= int(x) <= 95
    ))


def load_supervision():
    from datasets import load_dataset

    ds = load_dataset(DATASET_REPO, split="validation")
    out = defaultdict(lambda: {"start": {}, "goal": {}})

    for row in ds:
        task = str(row["task_id"])
        if task not in {"task_58", "task_59"}:
            continue

        eid = str(row["ep_id"])
        if eid.startswith("ep_"):
            eid = eid[3:]

        order = parse_order(str(row["question"]))
        ans = parse_answer(row["answer"])

        if order is None or not isinstance(ans, list):
            continue
        if len(order) != len(ans):
            continue

        side = "start" if task == "task_58" else "goal"

        for idx, group in zip(order, ans):
            frames = clean_group(group)
            if frames:
                cur = set(out[eid][side].get(idx, []))
                cur.update(frames)
                out[eid][side][idx] = sorted(cur)

    return dict(out)


def build_video_index(cache_dir):
    return {p.stem: p for p in Path(cache_dir).rglob("*.mp4")}


def pick_frames(indices, k=2):
    xs = sorted(set(indices))
    if len(xs) <= k:
        return xs
    qs = [(i + 1) / (k + 1) for i in range(k)]
    return sorted(set(xs[round(q * (len(xs) - 1))] for q in qs))


def read_frame(path, idx):
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(f"Could not decode frame {idx} from {path}")

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def save_rgb(path, rgb):
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(path),
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
    )


def load_vlm(model_id):
    from mlx_vlm import load
    from mlx_vlm.utils import load_config

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


def generation_text(x):
    if isinstance(x, str):
        return x
    t = getattr(x, "text", None)
    return t if isinstance(t, str) else str(x)


def parse_ab(text):
    t = text.strip().upper()

    # Prefer a single-token answer when possible.
    m = re.search(r"\b(A|B)\b", t)
    if m:
        return m.group(1)

    if "NOT_VISIBLE" in t or "NOT VISIBLE" in t:
        return "NOT_VISIBLE"
    if "UNCERTAIN" in t or "UNSURE" in t:
        return "UNCERTAIN"

    return "PARSE_ERROR"


def classify_multiframe(
    model,
    processor,
    config,
    image_paths,
    option_a,
    option_b,
):
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    prompt = (
        f"You are given {len(image_paths)} egocentric robot image(s) from the "
        "same interaction state and different nearby views. "
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
    ap.add_argument("--cache-dir", default="findingdory_rgb_data")
    ap.add_argument(
        "--out-dir",
        default="findingdory_rgb_grounding_v5_neutral",
    )
    ap.add_argument("--pilot-episodes", type=int, default=5)
    ap.add_argument("--frames-per-state", type=int, default=2)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--seed", type=int, default=20260916)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    frames_dir = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    events_by_ep = build_val_events(
        args.metadata_root,
        args.transitions,
    )
    supervision = load_supervision()
    videos = build_video_index(args.cache_dir)

    eligible = []

    for eid, ep in events_by_ep.items():
        sup = supervision.get(eid)
        video = videos.get(f"ep_{eid}")

        if sup is None or video is None:
            continue

        usable = []

        for ev in ep["events"]:
            if normalize_category(
                ev["start_receptacle_category"]
            ) == normalize_category(
                ev["goal_receptacle_category"]
            ):
                continue

            i = ev["interaction_index"]
            sf = sup["start"].get(i, [])
            gf = sup["goal"].get(i, [])

            if sf and gf:
                usable.append({
                    **ev,
                    "start_frames": sf,
                    "goal_frames": gf,
                })

        if usable:
            eligible.append({
                **ep,
                "events": usable,
                "video_path": str(video),
            })

    eligible.sort(
        key=lambda e: (-len(e["events"]), int(e["episode_id"]))
    )

    rng = random.Random(args.seed)
    pool = eligible[:max(args.pilot_episodes * 3, args.pilot_episodes)]
    rng.shuffle(pool)
    selected = pool[:args.pilot_episodes]

    setup = {
        "selected_episode_ids": [e["episode_id"] for e in selected],
        "selected_transition_count": sum(len(e["events"]) for e in selected),
        "frames_per_state": args.frames_per_state,
        "model_id": args.model,
        "neutral_labels": True,
        "joint_multiframe_prompt": True,
        "candidate_order_randomized": True,
    }

    (out_dir / "setup_report.json").write_text(
        json.dumps(setup, indent=2),
        encoding="utf-8",
    )

    print("=" * 94)
    print("FINDINGDORY RGB GROUNDING V5 — NEUTRAL A/B MULTI-FRAME")
    print("=" * 94)
    print("Selected episodes:", ", ".join(setup["selected_episode_ids"]))
    print("Selected transitions:", setup["selected_transition_count"])

    model, processor, config = load_vlm(args.model)

    rows = []

    for epn, ep in enumerate(selected, 1):
        eid = ep["episode_id"]
        video = Path(ep["video_path"])

        print()
        print(
            f"[episode {epn}/{len(selected)}] "
            f"id={eid} events={len(ep['events'])}"
        )

        for ev in ep["events"]:
            # One neutral A/B assignment per native transition.
            start_is_a = stable_bit(
                args.seed,
                eid,
                ev["interaction_index"],
            ) == 0

            if start_is_a:
                option_a = ev["start_receptacle_category"]
                option_b = ev["goal_receptacle_category"]
                a_state, b_state = "START", "GOAL"
            else:
                option_a = ev["goal_receptacle_category"]
                option_b = ev["start_receptacle_category"]
                a_state, b_state = "GOAL", "START"

            for expected, key in (
                ("START", "start_frames"),
                ("GOAL", "goal_frames"),
            ):
                chosen = pick_frames(
                    ev[key],
                    args.frames_per_state,
                )

                image_paths = []
                for fi in chosen:
                    rgb = read_frame(video, fi)
                    p = frames_dir / (
                        f"ep_{eid}_obj_{ev['interaction_index']:02d}_"
                        f"{expected.lower()}_f{fi:02d}.jpg"
                    )
                    save_rgb(p, rgb)
                    image_paths.append(p)

                raw_label, raw_text = classify_multiframe(
                    model,
                    processor,
                    config,
                    image_paths,
                    option_a,
                    option_b,
                )

                if raw_label == "A":
                    pred = a_state
                elif raw_label == "B":
                    pred = b_state
                else:
                    pred = raw_label

                row = {
                    "episode_id": eid,
                    "interaction_index": ev["interaction_index"],
                    "object_handle": ev["object_handle"],
                    "start_receptacle_category": ev["start_receptacle_category"],
                    "goal_receptacle_category": ev["goal_receptacle_category"],
                    "option_a": option_a,
                    "option_b": option_b,
                    "a_state": a_state,
                    "b_state": b_state,
                    "expected_state": expected,
                    "predicted_state": pred,
                    "raw_neutral_label": raw_label,
                    "correct": pred == expected,
                    "num_images": len(image_paths),
                    "frame_indices": chosen,
                    "raw_output": raw_text,
                }
                rows.append(row)

                mark = "✓" if row["correct"] else "·"
                print(
                    f"  {mark} obj={ev['interaction_index']:>2} "
                    f"{expected:<5} "
                    f"A={normalize_category(option_a)[:11]:<11} "
                    f"B={normalize_category(option_b)[:11]:<11} "
                    f"raw={raw_label:<11} pred={pred}"
                )

    out_pred = out_dir / "event_predictions.jsonl"
    with out_pred.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    judged = [
        r for r in rows
        if r["predicted_state"] in {"START", "GOAL"}
    ]

    by_state = {}
    for state in ("START", "GOAL"):
        rr = [r for r in rows if r["expected_state"] == state]
        jj = [
            r for r in rr
            if r["predicted_state"] in {"START", "GOAL"}
        ]
        by_state[state] = {
            "n": len(rr),
            "coverage": len(jj) / len(rr) if rr else None,
            "accuracy_when_judged": (
                sum(x["correct"] for x in jj) / len(jj)
                if jj else None
            ),
            "all_event_accuracy": (
                sum(x["correct"] for x in rr) / len(rr)
                if rr else None
            ),
        }

    # A/B preference helps diagnose residual prompt-position bias.
    neutral_counts = Counter(r["raw_neutral_label"] for r in rows)

    report = {
        "num_state_events": len(rows),
        "neutral_label_counts": dict(neutral_counts),
        "state_judgment_coverage": (
            len(judged) / len(rows) if rows else None
        ),
        "accuracy_when_judged": (
            sum(r["correct"] for r in judged) / len(judged)
            if judged else None
        ),
        "all_event_accuracy": (
            sum(r["correct"] for r in rows) / len(rows)
            if rows else None
        ),
        "by_expected_state": by_state,
        "model_id": args.model,
        "neutral_labels": True,
        "joint_multiframe_prompt": True,
        "candidate_order_randomized": True,
        "interpretation_note": (
            "This is a prompt-formulation ablation on the same oracle-frame "
            "RGB grounding task. It tests whether START/GOAL wording and "
            "single-frame independent classification caused the earlier bias."
        ),
    }

    out_report = out_dir / "rgb_grounding_report.json"
    out_report.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 94)
    print("V5 COMPLETE")
    print("=" * 94)
    print(
        f"Coverage: {100*report['state_judgment_coverage']:.2f}%"
    )
    print(
        f"Accuracy when judged: "
        f"{100*report['accuracy_when_judged']:.2f}%"
    )
    print(
        f"All-event accuracy: "
        f"{100*report['all_event_accuracy']:.2f}%"
    )
    for state, x in by_state.items():
        print(
            f"{state}: coverage={100*x['coverage']:.2f}% "
            f"acc={100*x['accuracy_when_judged']:.2f}%"
        )
    print("Neutral label counts:", dict(neutral_counts))
    print()
    print(f"Upload {out_report}")
    print(f"and    {out_pred}")


if __name__ == "__main__":
    main()
