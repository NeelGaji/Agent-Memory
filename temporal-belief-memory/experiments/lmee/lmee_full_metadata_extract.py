#!/usr/bin/env python3
"""
Download the FULL LMEE-Bench test metadata (JSON only) and flatten it into
a StateMem-ready trajectory skeleton.

This intentionally does NOT download the PNG observations.

Requirements:
    python3 -m pip install huggingface_hub

Run:
    cd ~/Downloads
    python3 lmee_full_metadata_extract.py

Optional smoke test:
    python3 lmee_full_metadata_extract.py --max-tasks 10

Outputs:
    lmee_full_statemem/
      metadata/...
      lmee_trajectory_skeleton_full.jsonl
      extraction_summary_full.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
    from huggingface_hub import HfApi, hf_hub_download
except ImportError:
    raise SystemExit(
        "Missing huggingface_hub.\n"
        "Install with:\n"
        "  python3 -m pip install huggingface_hub"
    )

REPO_ID = "wangsen99/LMEE-Bench"
STEP_RE = re.compile(r"(?:^|/)(-?\d+)_([^/]+)/(?:left|front|right)\.png$")


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_img_triplet(img_paths):
    if not isinstance(img_paths, list):
        return None, None, None

    candidate = next((x for x in img_paths if isinstance(x, str)), None)
    if candidate is None:
        return None, None, None

    m = STEP_RE.search(candidate)
    if not m:
        return None, None, None

    sim_step = int(m.group(1))
    descriptor = m.group(2)

    if "_for_" in descriptor:
        action_from_path, target_from_path = descriptor.split("_for_", 1)
    else:
        action_from_path, target_from_path = descriptor, None

    return sim_step, action_from_path, target_from_path


def object_class_from_id(object_id):
    if not isinstance(object_id, str):
        return None
    return re.sub(r"_\d+$", "", object_id)


def trial_number(name):
    m = re.search(r"(\d+)$", str(name))
    return int(m.group(1)) if m else 10**9


def parse_qas(qa_path):
    if qa_path is None or not Path(qa_path).exists():
        return []

    data = load_json(qa_path)
    if not isinstance(data, dict):
        return []

    out = []
    for qa in data.get("QAs", []):
        if not isinstance(qa, dict):
            continue

        image_path = qa.get("image_path")
        trial_name = None
        stop_step = None

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


def download_one(relpath, out_dir):
    return relpath, hf_hub_download(
        repo_id=REPO_ID,
        filename=relpath,
        repo_type="dataset",
        local_dir=str(out_dir),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="lmee_full_statemem")
    ap.add_argument("--max-tasks", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    metadata_dir = out_dir / "metadata"
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    print(f"Listing files in {REPO_ID} ...")
    api = HfApi()
    repo_files = api.list_repo_files(REPO_ID, repo_type="dataset")

    task_jsons = sorted(
        p for p in repo_files
        if p.startswith("task_test/")
        and p.endswith("/success/task.json")
    )

    if args.max_tasks is not None:
        task_jsons = task_jsons[: args.max_tasks]

    if not task_jsons:
        raise SystemExit("No task_test/*/success/task.json files found.")

    selected = set(task_jsons)
    qa_jsons = []

    for task_rel in task_jsons:
        qa_rel = task_rel[:-len("task.json")] + "task_QA.json"
        if qa_rel in repo_files:
            qa_jsons.append(qa_rel)

    files_to_download = sorted(selected | set(qa_jsons))

    print(f"Tasks selected: {len(task_jsons)}")
    print(f"JSON files to download: {len(files_to_download)}")
    print("PNG images selected: 0")
    print()

    downloaded = {}

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = {
            ex.submit(download_one, rel, metadata_dir): rel
            for rel in files_to_download
        }

        done = 0
        for fut in as_completed(futures):
            rel = futures[fut]
            try:
                rel2, local = fut.result()
                downloaded[rel2] = local
            except Exception as e:
                print(f"[ERROR] {rel}: {e}")

            done += 1
            if done % 25 == 0 or done == len(futures):
                print(f"Downloaded {done}/{len(futures)} JSON files")

    jsonl_path = out_dir / "lmee_trajectory_skeleton_full.jsonl"
    summary_path = out_dir / "extraction_summary_full.json"

    task_summaries = []
    total_events = 0
    difficulty_counts = Counter()
    scene_counts = Counter()
    qa_category_counts = Counter()
    failed_tasks = []

    with jsonl_path.open("w", encoding="utf-8") as out_f:
        for task_rel in task_jsons:
            local_task = downloaded.get(task_rel)
            if not local_task:
                failed_tasks.append({
                    "task_json": task_rel,
                    "reason": "task.json download missing"
                })
                continue

            try:
                data = load_json(local_task)
            except Exception as e:
                failed_tasks.append({
                    "task_json": task_rel,
                    "reason": f"JSON read error: {e}"
                })
                continue

            if not isinstance(data, dict):
                failed_tasks.append({
                    "task_json": task_rel,
                    "reason": "task.json is not a dict"
                })
                continue

            qa_rel = task_rel[:-len("task.json")] + "task_QA.json"
            qas = parse_qas(downloaded.get(qa_rel))

            for q in qas:
                if q.get("category"):
                    qa_category_counts[q["category"]] += 1

            difficulty = data.get("Difficulty")
            scene = data.get("Scene")
            difficulty_counts[str(difficulty)] += 1
            scene_counts[str(scene)] += 1

            task_id = task_rel[:-len("/success/task.json")]
            object_ids = data.get("Object_id", [])
            trials = data.get("trial", {})

            ts = {
                "task_id": task_id,
                "scene": scene,
                "difficulty": difficulty,
                "object_ids": object_ids,
                "num_trials": 0,
                "num_events": 0,
                "num_qas": len(qas),
                "trial_ranges": [],
                "continuity_warnings": [],
            }

            if not isinstance(trials, dict):
                failed_tasks.append({
                    "task_json": task_rel,
                    "reason": "trial field is not a dict"
                })
                task_summaries.append(ts)
                continue

            ordered_trials = sorted(
                trials.items(),
                key=lambda kv: trial_number(kv[0])
            )
            ts["num_trials"] = len(ordered_trials)

            previous_last_step = None
            global_event_index = 0

            for subtask_idx, (trial_name, trial) in enumerate(ordered_trials):
                if not isinstance(trial, dict):
                    ts["continuity_warnings"].append(
                        f"{trial_name}: trial is not a dict"
                    )
                    continue

                pos = trial.get("pos")
                action = trial.get("action")
                semantic = trial.get("object")
                img = trial.get("img_path")

                arrays = [pos, action, semantic, img]

                if not all(isinstance(x, list) for x in arrays):
                    ts["continuity_warnings"].append(
                        f"{trial_name}: missing parallel list"
                    )
                    continue

                lengths = [len(x) for x in arrays]

                if len(set(lengths)) != 1:
                    ts["continuity_warnings"].append(
                        f"{trial_name}: unequal parallel lengths {lengths}"
                    )
                    continue

                target_id = (
                    object_ids[subtask_idx]
                    if isinstance(object_ids, list)
                    and subtask_idx < len(object_ids)
                    else None
                )
                target_class = object_class_from_id(target_id)

                parsed_steps = []

                for j in range(lengths[0]):
                    sim_step, path_action, path_target = parse_img_triplet(img[j])

                    if sim_step is not None:
                        parsed_steps.append(sim_step)

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
                        "instruction": data.get("Task instruction"),
                        "trial_name": trial_name,
                        "subtask_index": subtask_idx,
                        "target_object_id": target_id,
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
                    global_event_index += 1
                    total_events += 1

                first_step = parsed_steps[0] if parsed_steps else None
                last_step = parsed_steps[-1] if parsed_steps else None

                ts["trial_ranges"].append({
                    "trial_name": trial_name,
                    "target_object_id": target_id,
                    "target_object_class": target_class,
                    "num_events": lengths[0],
                    "first_sim_step": first_step,
                    "last_sim_step": last_step,
                })

                if (
                    previous_last_step is not None
                    and first_step is not None
                    and first_step != previous_last_step + 1
                ):
                    ts["continuity_warnings"].append(
                        f"{trial_name}: starts at {first_step}, "
                        f"previous trial ended at {previous_last_step}"
                    )

                previous_last_step = last_step

            ts["num_events"] = global_event_index
            task_summaries.append(ts)

    summary = {
        "repo_id": REPO_ID,
        "scope": "task_test",
        "num_tasks_discovered_or_selected": len(task_jsons),
        "num_tasks_extracted": len(task_summaries),
        "failed_tasks": failed_tasks,
        "total_events": total_events,
        "difficulty_counts": dict(difficulty_counts),
        "unique_scenes": len(scene_counts),
        "scene_task_counts": dict(scene_counts),
        "qa_category_counts": dict(qa_category_counts),
        "tasks": task_summaries,
        "notes": [
            "Only JSON metadata was downloaded; no PNG observations were downloaded.",
            "Rows preserve released LMEE trajectory events.",
            "sim_step is parsed from LMEE image filenames.",
            "The LMEE `object` array is preserved as semantic_terms.",
            "No dynamic ground-truth object state is invented.",
            "Terminal move/stop observations can share the same sim_step.",
        ],
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    warning_count = sum(
        len(x["continuity_warnings"]) for x in task_summaries
    )

    print()
    print("=" * 90)
    print("FULL LMEE METADATA EXTRACTION COMPLETE")
    print("=" * 90)
    print(f"Tasks extracted: {len(task_summaries)}")
    print(f"Total events:    {total_events}")
    print(f"Unique scenes:   {len(scene_counts)}")
    print(f"Difficulties:    {dict(difficulty_counts)}")
    print(f"QA categories:   {dict(qa_category_counts)}")
    print(f"Warnings:        {warning_count}")
    print(f"Failed tasks:    {len(failed_tasks)}")
    print()
    print(f"JSONL:   {jsonl_path}")
    print(f"Summary: {summary_path}")
    print()
    print(
        "Upload extraction_summary_full.json when this finishes. "
        "If the task count is large enough and warnings are clean, "
        "the next script will run the LMEE-conditioned dynamic-state diagnostic."
    )


if __name__ == "__main__":
    main()
