#!/usr/bin/env python3
"""
LMEE detailed trial probe
=========================

Reads the task.json files already downloaded by lmee_metadata_probe.py and
inspects the actual contents of each trial's parallel arrays:

    pos
    action
    object
    img_path

It also cross-checks task_QA.json stop indices against the saved image paths.

Run from Downloads:
    python3 lmee_trial_probe.py

Expected existing folder:
    ~/Downloads/lmee_probe/downloaded/

Output:
    lmee_trial_detail_report.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compact(x: Any, max_chars: int = 500) -> Any:
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    text = json.dumps(x, ensure_ascii=False)
    if len(text) <= max_chars:
        return x
    return text[:max_chars] + "...<truncated>"


def parse_step_from_path(path_value: Any):
    if not isinstance(path_value, str):
        return None
    # Match things like 93_stop_for_microwave/front.png or 42/front.png
    m = re.search(r"(?:^|/)(\d+)(?:_|/)", path_value)
    return int(m.group(1)) if m else None


def summarize_array(values, n=5):
    if not isinstance(values, list):
        return {
            "type": type(values).__name__,
            "value": compact(values),
        }

    return {
        "type": "list",
        "length": len(values),
        "element_types": sorted({type(v).__name__ for v in values[: min(50, len(values))]}),
        "first": [compact(v) for v in values[:n]],
        "last": [compact(v) for v in values[-n:]] if values else [],
    }


def inspect_trial(trial_name: str, trial: dict, n: int):
    keys = ["pos", "action", "object", "img_path"]

    result = {
        "trial_name": trial_name,
        "keys": list(trial.keys()),
        "arrays": {},
        "aligned_rows_first": [],
        "aligned_rows_last": [],
    }

    for key in keys:
        result["arrays"][key] = summarize_array(trial.get(key), n=n)

    arrays = [trial.get(k) for k in keys]
    if all(isinstance(a, list) for a in arrays):
        lengths = [len(a) for a in arrays]
        result["parallel_lengths"] = dict(zip(keys, lengths))
        result["all_same_length"] = len(set(lengths)) == 1

        L = min(lengths) if lengths else 0

        def row(i):
            img = trial["img_path"][i]
            return {
                "index_in_array": i,
                "step_from_img_path": parse_step_from_path(img),
                "pos": compact(trial["pos"][i]),
                "action": compact(trial["action"][i]),
                "object": compact(trial["object"][i]),
                "img_path": compact(img),
            }

        result["aligned_rows_first"] = [row(i) for i in range(min(n, L))]
        result["aligned_rows_last"] = [
            row(i) for i in range(max(0, L - n), L)
        ]

        # Inspect image-step numbering to determine whether rows correspond
        # to every simulator timestep or only sampled/checkpoint observations.
        steps = [
            parse_step_from_path(x)
            for x in trial["img_path"]
            if isinstance(x, str)
        ]
        steps = [x for x in steps if x is not None]

        if steps:
            deltas = [b - a for a, b in zip(steps, steps[1:])]
            result["img_step_stats"] = {
                "count_parseable": len(steps),
                "min_step": min(steps),
                "max_step": max(steps),
                "first_steps": steps[:20],
                "last_steps": steps[-20:],
                "unique_deltas": sorted(set(deltas))[:30],
                "all_consecutive": all(d == 1 for d in deltas),
            }

    return result


def find_matching_qa(task_json_path: Path):
    # task.json is under .../<task>/success/task.json
    qa_path = task_json_path.with_name("task_QA.json")
    if not qa_path.exists():
        return None, []

    data = load_json(qa_path)
    qas = data.get("QAs", []) if isinstance(data, dict) else []

    out = []
    for qa in qas:
        if not isinstance(qa, dict):
            continue
        image_path = qa.get("image_path")
        stop = None
        if isinstance(image_path, str):
            m = re.search(r"/(\d+)_stop_for_", image_path)
            stop = int(m.group(1)) if m else None

        out.append({
            "category": qa.get("category"),
            "question": qa.get("question"),
            "object_id": qa.get("object_id"),
            "image_path": image_path,
            "stop_index": stop,
        })

    return str(qa_path), out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="lmee_probe/downloaded",
        help="Root containing previously downloaded LMEE task files.",
    )
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument(
        "--output",
        default="lmee_trial_detail_report.json",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(
            f"Could not find {root}\n"
            "Run this from ~/Downloads, or pass --root to the correct folder."
        )

    task_files = sorted(root.rglob("success/task.json"))

    if not task_files:
        raise SystemExit(f"No success/task.json files found under {root}")

    print(f"Found {len(task_files)} task.json files.\n")

    report = {
        "root": str(root),
        "num_task_files": len(task_files),
        "tasks": [],
    }

    for idx, path in enumerate(task_files, 1):
        data = load_json(path)

        print("=" * 100)
        print(f"TASK {idx}: {path}")
        print("=" * 100)

        task_report = {
            "task_json": str(path),
            "scene": data.get("Scene") if isinstance(data, dict) else None,
            "difficulty": data.get("Difficulty") if isinstance(data, dict) else None,
            "object_ids": data.get("Object_id") if isinstance(data, dict) else None,
            "trials": [],
        }

        qa_path, qas = find_matching_qa(path)
        task_report["task_QA_json"] = qa_path
        task_report["qas"] = qas

        trials = data.get("trial", {}) if isinstance(data, dict) else {}
        print(f"Scene: {task_report['scene']}")
        print(f"Trial keys: {list(trials.keys()) if isinstance(trials, dict) else type(trials).__name__}")

        if isinstance(trials, dict):
            for trial_name, trial in trials.items():
                if not isinstance(trial, dict):
                    continue

                tr = inspect_trial(trial_name, trial, args.samples)
                task_report["trials"].append(tr)

                print(f"\n{trial_name}")
                print(f"  keys: {tr['keys']}")
                print(f"  lengths: {tr.get('parallel_lengths')}")
                print(f"  same length: {tr.get('all_same_length')}")

                stats = tr.get("img_step_stats")
                if stats:
                    print(
                        f"  image steps: {stats['min_step']}..{stats['max_step']} "
                        f"(n={stats['count_parseable']}, consecutive={stats['all_consecutive']})"
                    )
                    print(f"  first image steps: {stats['first_steps'][:10]}")
                    print(f"  step deltas: {stats['unique_deltas'][:15]}")

                print("  first aligned rows:")
                for row in tr["aligned_rows_first"]:
                    print(
                        f"    idx={row['index_in_array']}, "
                        f"img_step={row['step_from_img_path']}, "
                        f"action={row['action']!r}, "
                        f"object={row['object']!r}, "
                        f"pos={row['pos']!r}, "
                        f"img={row['img_path']!r}"
                    )

        if qas:
            print("\n  QA stop indices:")
            for qa in qas:
                print(
                    f"    stop={qa['stop_index']}, "
                    f"object_id={qa['object_id']!r}, "
                    f"category={qa['category']!r}"
                )

        report["tasks"].append(task_report)
        print()

    out_path = Path(args.output)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=" * 100)
    print("DONE")
    print("=" * 100)
    print(f"Saved: {out_path}")
    print(
        "\nUpload lmee_trial_detail_report.json here. "
        "That will give us the actual element-level LMEE trajectory schema."
    )


if __name__ == "__main__":
    main()
