#!/usr/bin/env python3
"""
FindingDory RGB grounding v4 — ordered multi-goal supervision
=============================================================

Purpose
-------
Scale the v3 RGB grounding diagnostic from a handful of task_32/task_33
matches to (almost) every native rearrangement.

FindingDory provides ordered multi-goal tasks:
  task_58: revisit all PICKED-FROM receptacles in a stated order
  task_59: revisit all PLACED-ON receptacles in a stated order

The question explicitly states an ordinal permutation, e.g.
    "The order to revisit them is: second, first."
and the answer contains one frame-index group per item in that same order.

We invert that permutation to recover, for every interaction index:
  START evidence frames <- task_58
  GOAL evidence frames  <- task_59

This is an ORACLE-FRAME RGB GROUNDING diagnostic:
the ground-truth frame groups are used to select the images shown to the VLM.
It is NOT yet the final end-to-end perception experiment.

Run
---
cd ~/Downloads
python3 findingdory_rgb_grounding_v4.py --pilot-episodes 5

Cheap mapping-only check:
python3 findingdory_rgb_grounding_v4.py --prepare-only

Outputs
-------
findingdory_rgb_grounding_v4/
  coverage_report.json
  setup_report.json
  frames/
  frame_predictions.jsonl
  event_predictions.jsonl
  rgb_grounding_report.json
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


def require_imports(run_vlm=True):
    missing = []
    try:
        import datasets  # noqa
    except Exception:
        missing.append("datasets")
    try:
        import cv2  # noqa
    except Exception:
        missing.append("opencv-python")
    if run_vlm:
        try:
            import mlx_vlm  # noqa
        except Exception:
            missing.append("mlx-vlm")

    if missing:
        raise SystemExit(
            "Missing packages: " + ", ".join(missing) + "\n"
            "Install with:\n"
            "  python3 -m pip install -U datasets opencv-python mlx-vlm\n"
        )


def load_json_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def load_transitions(path: Path):
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("split") != "val":
                continue
            rows[(str(r["episode_id"]), r["object_handle"])] = r
    return rows


def normalize_category(x):
    return str(x or "").replace("_", " ").strip().lower()


def candidate_category_map(ep):
    out = {}
    for key in ("candidate_objects", "candidate_objects_noninteracted"):
        for x in ep.get(key) or []:
            name = x.get("object_name")
            cat = x.get("object_category")
            if name and cat:
                out[name] = cat
    return out


def build_val_events(metadata_root: Path, transition_path: Path):
    eps = load_json_gz(metadata_root / "val" / "episodes.json.gz")["episodes"]
    transitions = load_transitions(transition_path)

    result = {}
    warnings = []

    for ep in eps:
        eid = str(ep["episode_id"])
        handles = list((ep.get("targets") or {}).keys())
        cats = candidate_category_map(ep)
        events = []

        for i, handle in enumerate(handles):
            tr = transitions.get((eid, handle))
            if tr is None:
                warnings.append({
                    "episode_id": eid,
                    "object_handle": handle,
                    "type": "missing_transition",
                })
                continue

            obj_cat = cats.get(tr.get("object_config"))
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
            })

        if len(events) == len(handles) and events:
            result[eid] = {
                "episode_id": eid,
                "scene_id": ep["scene_id"],
                "events": events,
            }

    return result, warnings


def parse_answer(answer):
    if isinstance(answer, list):
        return answer
    if isinstance(answer, str):
        try:
            return ast.literal_eval(answer)
        except Exception:
            return None
    return None


def parse_order(question: str):
    """
    Extract:
      "The order to revisit them is: second, first."
    -> [1, 0]
    """
    q = question.lower()
    m = re.search(
        r"order\s+to\s+revisit\s+them\s+is\s*:\s*([^\.]+)",
        q,
    )
    if not m:
        return None

    tail = m.group(1)
    words = re.findall(
        r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh)\b",
        tail,
    )
    if not words:
        return None

    return [ORDINAL_TO_INDEX[w] for w in words]


def clean_frame_group(group):
    if not isinstance(group, list):
        return []
    out = []
    for x in group:
        if isinstance(x, (int, float)):
            i = int(x)
            if 0 <= i <= 95:
                out.append(i)
    return sorted(set(out))


def load_ordered_supervision():
    from datasets import load_dataset

    print("Loading official FindingDory validation task metadata...")
    ds = load_dataset(DATASET_REPO, split="validation")

    by_ep = defaultdict(lambda: {"start": {}, "goal": {}, "video": None})
    stats = Counter()
    failures = []

    for row in ds:
        task = str(row["task_id"])
        if task not in {"task_58", "task_59"}:
            continue

        raw_ep = str(row["ep_id"])
        eid = raw_ep[3:] if raw_ep.startswith("ep_") else raw_ep
        by_ep[eid]["video"] = row["video"]

        order = parse_order(str(row["question"]))
        ans = parse_answer(row["answer"])

        if order is None:
            stats["order_parse_failed"] += 1
            failures.append({
                "episode_id": eid,
                "task_id": task,
                "reason": "order_parse_failed",
                "question": row["question"],
            })
            continue

        if not isinstance(ans, list):
            stats["answer_parse_failed"] += 1
            continue

        if len(order) != len(ans):
            stats["order_answer_length_mismatch"] += 1
            failures.append({
                "episode_id": eid,
                "task_id": task,
                "reason": "order_answer_length_mismatch",
                "order_len": len(order),
                "answer_len": len(ans),
            })
            continue

        side = "start" if task == "task_58" else "goal"

        for interaction_idx, group in zip(order, ans):
            frames = clean_frame_group(group)
            if not frames:
                stats["empty_group"] += 1
                continue

            existing = set(by_ep[eid][side].get(interaction_idx, []))
            existing.update(frames)
            by_ep[eid][side][interaction_idx] = sorted(existing)

        stats[f"{side}_rows"] += 1
        stats[f"{side}_mapped_groups"] += len(order)

    return dict(by_ep), dict(stats), failures


def build_video_index(cache_dir: Path):
    return {p.stem: p for p in cache_dir.rglob("*.mp4")}


def sample_frames(indices, k=2):
    xs = sorted(set(indices))
    if len(xs) <= k:
        return xs
    qs = [(i + 1) / (k + 1) for i in range(k)]
    return sorted(set(xs[round(q * (len(xs) - 1))] for q in qs))


def read_frame(path: Path, frame_idx: int):
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not decode frame {frame_idx} from {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def save_rgb(path: Path, rgb):
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


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


def aliases(category):
    c = normalize_category(category)
    return {c, c.replace(" ", "_"), c.replace(" ", "-")}


def parse_prediction(text, start_cat, goal_cat):
    tl = text.strip().lower()

    if re.search(r"\bnot[\s_-]*visible\b", tl):
        return "NOT_VISIBLE"
    if any(x in tl for x in ("uncertain", "unsure", "cannot tell", "can't tell")):
        return "UNCERTAIN"
    if re.search(r"\bstart\b", tl):
        return "START"
    if re.search(r"\bgoal\b", tl):
        return "GOAL"

    sh = any(a and a in tl for a in aliases(start_cat))
    gh = any(a and a in tl for a in aliases(goal_cat))

    if sh and not gh:
        return "START"
    if gh and not sh:
        return "GOAL"
    if sh and gh:
        return "UNCERTAIN"
    return "PARSE_ERROR"


def classify_frame(model, processor, config, image_path, start_cat, goal_cat):
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    s = normalize_category(start_cat)
    g = normalize_category(goal_cat)

    prompt = (
        "Inspect this egocentric robot image. "
        f"START means the relevant receptacle/location is a {s}. "
        f"GOAL means it is a {g}. "
        "Which option is visually supported by the image? "
        "Reply with exactly one of: START, GOAL, NOT_VISIBLE, UNCERTAIN."
    )

    formatted = apply_chat_template(
        processor,
        config,
        prompt,
        num_images=1,
    )
    res = generate(
        model,
        processor,
        formatted,
        [str(image_path)],
        max_tokens=12,
        temp=0.0,
        verbose=False,
    )
    raw = generation_text(res).strip()
    return parse_prediction(raw, start_cat, goal_cat), raw


def aggregate(rows, expected):
    usable = [r for r in rows if r["predicted_label"] in {"START", "GOAL"}]
    if not usable:
        pred = "NO_STATE_JUDGMENT"
        agreement = 0.0
    else:
        c = Counter(r["predicted_label"] for r in usable)
        pred, votes = c.most_common(1)[0]
        agreement = votes / len(rows)

    return {
        "expected_state": expected,
        "predicted_state": pred,
        "correct": pred == expected,
        "agreement_confidence": agreement,
        "state_judgment_coverage": len(usable) / len(rows) if rows else 0.0,
        "num_frames": len(rows),
        "label_counts": dict(Counter(r["predicted_label"] for r in rows)),
    }


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
    ap.add_argument("--out-dir", default="findingdory_rgb_grounding_v4")
    ap.add_argument("--pilot-episodes", type=int, default=5)
    ap.add_argument("--frames-per-state", type=int, default=2)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--include-same-category", action="store_true")
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--seed", type=int, default=20260916)
    args = ap.parse_args()

    require_imports(run_vlm=not args.prepare_only)

    out_dir = Path(args.out_dir)
    frames_dir = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    events_by_ep, meta_warnings = build_val_events(
        Path(args.metadata_root),
        Path(args.transitions),
    )
    supervision, sup_stats, sup_failures = load_ordered_supervision()
    videos = build_video_index(Path(args.cache_dir))

    coverage = Counter()
    eligible = []

    for eid, ep in events_by_ep.items():
        sup = supervision.get(eid)
        if not sup:
            coverage["missing_supervision_episode"] += 1
            continue
        video = videos.get(f"ep_{eid}")
        if video is None:
            coverage["missing_video"] += 1
            continue

        usable_events = []
        for ev in ep["events"]:
            i = ev["interaction_index"]

            same_cat = (
                normalize_category(ev.get("start_receptacle_category"))
                == normalize_category(ev.get("goal_receptacle_category"))
            )
            if same_cat and not args.include_same_category:
                coverage["same_category_skipped"] += 1
                continue

            sf = sup["start"].get(i, [])
            gf = sup["goal"].get(i, [])

            if not sf:
                coverage["missing_start_group"] += 1
            if not gf:
                coverage["missing_goal_group"] += 1
            if not sf or not gf:
                continue

            usable_events.append({
                **ev,
                "official_start_frames": sf,
                "official_goal_frames": gf,
            })
            coverage["usable_events"] += 1

        if usable_events:
            eligible.append({
                **ep,
                "events": usable_events,
                "video_path": str(video),
            })

    coverage_report = {
        "official_val_episodes": len(events_by_ep),
        "native_val_events": sum(len(x["events"]) for x in events_by_ep.values()),
        "supervision_stats": sup_stats,
        "supervision_failure_count": len(sup_failures),
        "metadata_warning_count": len(meta_warnings),
        "coverage_counts": dict(coverage),
        "eligible_episode_count": len(eligible),
        "important_note": (
            "task_58/task_59 ground-truth answer groups are used to select RGB frames. "
            "This is oracle-frame grounding, not end-to-end perception."
        ),
    }
    (out_dir / "coverage_report.json").write_text(
        json.dumps(coverage_report, indent=2),
        encoding="utf-8",
    )

    print("=" * 94)
    print("FINDINGDORY RGB GROUNDING V4")
    print("=" * 94)
    print(f"Native val events:     {coverage_report['native_val_events']}")
    print(f"Usable RGB transitions:{coverage['usable_events']}")
    print(f"Eligible episodes:     {len(eligible)}")
    print(f"Same-category skipped: {coverage['same_category_skipped']}")
    print(f"Mapping failures:      {len(sup_failures)}")

    if args.prepare_only:
        print()
        print(f"Coverage report: {out_dir / 'coverage_report.json'}")
        return

    if not eligible:
        raise SystemExit("No eligible episodes. Inspect coverage_report.json.")

    # Favor episodes with many usable changed-category events, then randomize
    # among a top pool so the pilot is not just the exact same few episodes.
    eligible.sort(key=lambda e: (-len(e["events"]), int(e["episode_id"])))
    rng = random.Random(args.seed)
    pool = eligible[:max(args.pilot_episodes * 3, args.pilot_episodes)]
    rng.shuffle(pool)
    selected = pool[:args.pilot_episodes]

    setup = {
        "selected_episode_ids": [e["episode_id"] for e in selected],
        "selected_event_count": sum(len(e["events"]) for e in selected),
        "frames_per_state": args.frames_per_state,
        "model_id": args.model,
        "same_category_included": args.include_same_category,
        "frame_source": "FindingDory task_58/task_59 ordered multi-goal answer groups",
        "oracle_frame_selection": True,
    }
    (out_dir / "setup_report.json").write_text(
        json.dumps(setup, indent=2),
        encoding="utf-8",
    )

    print()
    print("Selected episodes:", ", ".join(setup["selected_episode_ids"]))
    print("Selected transitions:", setup["selected_event_count"])

    model, processor, config = load_vlm(args.model)

    frame_rows = []
    event_rows = []

    for epn, ep in enumerate(selected, 1):
        eid = ep["episode_id"]
        video = Path(ep["video_path"])
        print()
        print(f"[episode {epn}/{len(selected)}] id={eid} events={len(ep['events'])}")

        for ev in ep["events"]:
            for expected, key in (
                ("START", "official_start_frames"),
                ("GOAL", "official_goal_frames"),
            ):
                chosen = sample_frames(ev[key], args.frames_per_state)
                local = []

                for fi in chosen:
                    rgb = read_frame(video, fi)
                    img_path = frames_dir / (
                        f"ep_{eid}_obj_{ev['interaction_index']:02d}_"
                        f"{expected.lower()}_f{fi:02d}.jpg"
                    )
                    save_rgb(img_path, rgb)

                    pred, raw = classify_frame(
                        model, processor, config, img_path,
                        ev["start_receptacle_category"],
                        ev["goal_receptacle_category"],
                    )

                    row = {
                        "episode_id": eid,
                        "interaction_index": ev["interaction_index"],
                        "object_handle": ev["object_handle"],
                        "object_category": ev["object_category"],
                        "start_receptacle_category": ev["start_receptacle_category"],
                        "goal_receptacle_category": ev["goal_receptacle_category"],
                        "expected_state": expected,
                        "frame_idx": fi,
                        "predicted_label": pred,
                        "raw_output": raw,
                        "frame_path": str(img_path),
                    }
                    frame_rows.append(row)
                    local.append(row)

                    mark = "✓" if pred == expected else "·"
                    print(
                        f"  {mark} obj={ev['interaction_index']:>2} "
                        f"{expected:<5} f={fi:>2} "
                        f"{normalize_category(ev['start_receptacle_category'])[:11]}"
                        f"->{normalize_category(ev['goal_receptacle_category'])[:11]} "
                        f"pred={pred}"
                    )

                event_rows.append({
                    "episode_id": eid,
                    "interaction_index": ev["interaction_index"],
                    "object_handle": ev["object_handle"],
                    "object_category": ev["object_category"],
                    "start_receptacle_category": ev["start_receptacle_category"],
                    "goal_receptacle_category": ev["goal_receptacle_category"],
                    **aggregate(local, expected),
                })

    with (out_dir / "frame_predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in frame_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with (out_dir / "event_predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in event_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    judged = [r for r in event_rows if r["predicted_state"] in {"START", "GOAL"}]
    correct = [r for r in judged if r["correct"]]
    frame_judged = [r for r in frame_rows if r["predicted_label"] in {"START", "GOAL"}]

    by_state = {}
    for state in ("START", "GOAL"):
        rr = [r for r in event_rows if r["expected_state"] == state]
        jj = [r for r in rr if r["predicted_state"] in {"START", "GOAL"}]
        by_state[state] = {
            "n": len(rr),
            "coverage": len(jj) / len(rr) if rr else None,
            "accuracy_when_judged": (
                sum(x["correct"] for x in jj) / len(jj) if jj else None
            ),
            "all_event_accuracy": (
                sum(x["correct"] for x in rr) / len(rr) if rr else None
            ),
        }

    report = {
        "num_frame_predictions": len(frame_rows),
        "num_state_events": len(event_rows),
        "frame_label_counts": dict(Counter(r["predicted_label"] for r in frame_rows)),
        "frame_state_judgment_coverage": (
            len(frame_judged) / len(frame_rows) if frame_rows else None
        ),
        "event_state_judgment_coverage": (
            len(judged) / len(event_rows) if event_rows else None
        ),
        "event_accuracy_when_judged": (
            len(correct) / len(judged) if judged else None
        ),
        "all_event_accuracy": (
            len(correct) / len(event_rows) if event_rows else None
        ),
        "by_expected_state": by_state,
        "oracle_frame_selection": True,
        "interpretation_note": (
            "This scales real-RGB START/GOAL grounding using official ordered "
            "revisitation frame annotations. It is an oracle-frame perception "
            "diagnostic, not yet the final end-to-end StateMem experiment."
        ),
    }

    (out_dir / "rgb_grounding_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 94)
    print("RGB GROUNDING V4 COMPLETE")
    print("=" * 94)
    print(
        f"Frame judgment coverage: "
        f"{100*report['frame_state_judgment_coverage']:.2f}%"
    )
    print(
        f"Event judgment coverage: "
        f"{100*report['event_state_judgment_coverage']:.2f}%"
    )
    print(
        f"Event accuracy when judged: "
        f"{100*report['event_accuracy_when_judged']:.2f}%"
    )
    print(f"All-event accuracy: {100*report['all_event_accuracy']:.2f}%")
    for state, x in by_state.items():
        print(
            f"{state}: n={x['n']} "
            f"coverage={100*x['coverage']:.2f}% "
            f"acc={100*x['accuracy_when_judged']:.2f}%"
        )
    print()
    print("Upload:")
    print(f"  {out_dir / 'coverage_report.json'}")
    print(f"  {out_dir / 'rgb_grounding_report.json'}")
    print(f"  {out_dir / 'event_predictions.jsonl'}")


if __name__ == "__main__":
    main()
