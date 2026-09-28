#!/usr/bin/env python3
"""
LMEE -> StateMem trajectory skeleton extractor
==============================================

This flattens the REAL released LMEE task trajectories into one JSONL stream.

Important:
- It does NOT invent dynamic object states.
- LMEE's released `object` field is a list of semantic terms, not a persistent
  object-state label.
- This extractor preserves the real LMEE/Habitat action timeline, position,
  target object ID/class, semantic terms, RGB path triplet, and QA checkpoints.
- We will use this flattened stream as the substrate for the next dynamic-state
  experiment.

Expected input (created by the earlier probe):
    lmee_probe/downloaded/.../success/task.json
    lmee_probe/downloaded/.../success/task_QA.json

Run:
    cd ~/Downloads
    python3 lmee_to_statemem_skeleton.py

Outputs:
    lmee_statemem/
      lmee_trajectory_skeleton.jsonl
      extraction_summary.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


STEP_RE = re.compile(r"(?:^|/)(-?\d+)_([^/]+)/(?:left|front|right)\.png$")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_img_triplet(img_paths: Any):
    """
    LMEE stores each observation as [left.png, front.png, right.png].
    We parse the simulator/action step from any one path in the triplet.
    """
    if not isinstance(img_paths, list) or not img_paths:
        return None, None, None

    candidate = next((x for x in img_paths if isinstance(x, str)), None)
    if candidate is None:
        return None, None, None

    m = STEP_RE.search(candidate)
    if not m:
        return None, None, None

    sim_step = int(m.group(1))
    descriptor = m.group(2)

    # e.g. "93_stop_for_microwave"
    action_from_path = descriptor.split("_for_", 1)[0] if "_for_" in descriptor else descriptor
    target_from_path = descriptor.split("_for_", 1)[1] if "_for_" in descriptor else None

    return sim_step, action_from_path, target_from_path


def object_class_from_id(object_id: str | None) -> str | None:
    if not object_id:
        return None
    # "exercise bike_400" -> "exercise bike"
    return re.sub(r"_\d+$", "", object_id)


def trial_number(name: str) -> int:
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else 10**9


def load_qas(task_json_path: Path):
    qa_path = task_json_path.with_name("task_QA.json")
    if not qa_path.exists():
        return []

    data = load_json(qa_path)
    if not isinstance(data, dict):
        return []

    out = []
    for qa in data.get("QAs", []):
        if not isinstance(qa, dict):
            continue

        image_path = qa.get("image_path")
        stop_step = None
        trial_name = None

        if isinstance(image_path, str):
            m = re.search(r"/(trial_\d+)/(-?\d+)_stop_for_", image_path)
            if m:
                trial_name = m.group(1)
                stop_step = int(m.group(2))

        out.append({
            "category": qa.get("category"),
            "question": qa.get("question"),
            "choices": qa.get("choices"),
            "answer": qa.get("answer"),
            "open_answer": qa.get("open_answer"),
            "object_id": qa.get("object_id"),
            "image_path": image_path,
            "trial_name": trial_name,
            "stop_step": stop_step,
        })

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="lmee_probe/downloaded",
        help="Root containing downloaded LMEE task folders.",
    )
    parser.add_argument(
        "--out-dir",
        default="lmee_statemem",
    )
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    task_files = sorted(root.rglob("success/task.json"))
    if not task_files:
        raise SystemExit(
            f"No success/task.json found under {root}\n"
            "Run from ~/Downloads or pass --root explicitly."
        )

    jsonl_path = out_dir / "lmee_trajectory_skeleton.jsonl"
    summary_path = out_dir / "extraction_summary.json"

    summary = {
        "root": str(root),
        "num_tasks": len(task_files),
        "tasks": [],
        "notes": [
            "Rows preserve real LMEE trajectory events.",
            "semantic_terms comes from LMEE's `object` array.",
            "No dynamic true_state/observed_state is invented here.",
            "sim_step is parsed from LMEE image filenames.",
            "Repeated sim_step values can occur for move/stop at the same checkpoint.",
        ],
    }

    total_rows = 0

    with jsonl_path.open("w", encoding="utf-8") as out_f:
        for task_i, task_path in enumerate(task_files):
            data = load_json(task_path)
            if not isinstance(data, dict):
                continue

            scene = data.get("Scene")
            difficulty = data.get("Difficulty")
            instruction = data.get("Task instruction")
            object_ids = data.get("Object_id", [])
            trials = data.get("trial", {})
            qas = load_qas(task_path)

            try:
                rel_task = task_path.relative_to(root)
                task_id = str(rel_task.parent.parent)
            except Exception:
                task_id = str(task_path.parent)

            task_summary = {
                "task_id": task_id,
                "scene": scene,
                "difficulty": difficulty,
                "object_ids": object_ids,
                "num_trials": 0,
                "num_events": 0,
                "trial_ranges": [],
                "continuity_warnings": [],
            }

            if not isinstance(trials, dict):
                summary["tasks"].append(task_summary)
                continue

            ordered_trials = sorted(trials.items(), key=lambda kv: trial_number(kv[0]))
            task_summary["num_trials"] = len(ordered_trials)

            global_event_index = 0
            previous_last_step = None

            for subtask_idx, (trial_name, trial) in enumerate(ordered_trials):
                if not isinstance(trial, dict):
                    continue

                pos = trial.get("pos", [])
                action = trial.get("action", [])
                semantic = trial.get("object", [])
                img = trial.get("img_path", [])

                lengths = [len(x) for x in [pos, action, semantic, img] if isinstance(x, list)]
                if len(lengths) != 4 or len(set(lengths)) != 1:
                    task_summary["continuity_warnings"].append(
                        f"{trial_name}: parallel arrays are missing or unequal"
                    )
                    continue

                target_object_id = (
                    object_ids[subtask_idx]
                    if isinstance(object_ids, list) and subtask_idx < len(object_ids)
                    else None
                )
                target_class = object_class_from_id(target_object_id)

                trial_steps = []

                for j in range(lengths[0]):
                    sim_step, path_action, path_target = parse_img_triplet(img[j])

                    matching_qas = [
                        q for q in qas
                        if q["trial_name"] == trial_name
                        and q["stop_step"] == sim_step
                        and action[j] == "stop"
                    ]

                    row = {
                        "task_id": task_id,
                        "scene": scene,
                        "difficulty": difficulty,
                        "instruction": instruction,
                        "trial_name": trial_name,
                        "subtask_index": subtask_idx,
                        "target_object_id": target_object_id,
                        "target_object_class": target_class,
                        "event_index": global_event_index,
                        "trial_event_index": j,
                        "sim_step": sim_step,
                        "action": action[j],
                        "action_from_path": path_action,
                        "path_target_label": path_target,
                        "position": pos[j],
                        "semantic_terms": semantic[j],
                        "img_paths": img[j],
                        "is_stop": action[j] == "stop",
                        "qa_checkpoint": bool(matching_qas),
                        "qas": matching_qas,
                    }

                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

                    if sim_step is not None:
                        trial_steps.append(sim_step)

                    global_event_index += 1
                    total_rows += 1

                first_step = trial_steps[0] if trial_steps else None
                last_step = trial_steps[-1] if trial_steps else None

                task_summary["trial_ranges"].append({
                    "trial_name": trial_name,
                    "target_object_id": target_object_id,
                    "target_object_class": target_class,
                    "num_events": lengths[0],
                    "first_sim_step": first_step,
                    "last_sim_step": last_step,
                })

                # LMEE often has:
                # trial_0: -1_stop, 0..., 24_move, 24_stop
                # trial_1: 25..., 93_move, 93_stop
                # So next trial usually begins at last_step + 1.
                if (
                    previous_last_step is not None
                    and first_step is not None
                    and first_step != previous_last_step + 1
                ):
                    task_summary["continuity_warnings"].append(
                        f"{trial_name}: starts at {first_step}, "
                        f"previous trial ended at {previous_last_step}"
                    )

                previous_last_step = last_step

            task_summary["num_events"] = global_event_index
            summary["tasks"].append(task_summary)

    summary["total_events"] = total_rows

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 90)
    print("LMEE -> StateMem trajectory skeleton extraction complete")
    print("=" * 90)
    print(f"Tasks:  {len(summary['tasks'])}")
    print(f"Events: {total_rows}")
    print(f"JSONL:  {jsonl_path}")
    print(f"Summary:{summary_path}")
    print()

    for t in summary["tasks"][:5]:
        print(f"{t['scene']} | events={t['num_events']} | warnings={len(t['continuity_warnings'])}")
        for r in t["trial_ranges"]:
            print(
                f"  {r['trial_name']}: {r['first_sim_step']}..{r['last_sim_step']} "
                f"target={r['target_object_id']} events={r['num_events']}"
            )
        for w in t["continuity_warnings"]:
            print(f"  WARNING: {w}")
        print()

    print(
        "Next step: use this real LMEE timeline to construct the controlled "
        "dynamic-state diagnostic stream for StateMem."
    )


if __name__ == "__main__":
    main()
