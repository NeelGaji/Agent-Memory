#!/usr/bin/env python3
"""
LMEE metadata / trajectory probe
================================

Downloads ONLY a few JSON metadata files from the official LMEE-Bench
Hugging Face dataset, inspects their schema, and writes a compact report.

This deliberately avoids downloading the full image dataset.

Install once if needed:
    python3 -m pip install huggingface_hub

Example:
    python3 lmee_metadata_probe.py --difficulty easy --num-tasks 3

Outputs:
    lmee_probe/
      probe_report.json
      downloaded/
        ...

After this runs, send me the terminal output or probe_report.json and we can
write the exact LMEE -> StateMem extractor against the real released schema.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

try:
    from huggingface_hub import HfApi, hf_hub_download
except ImportError:
    raise SystemExit(
        "\nMissing dependency: huggingface_hub\n"
        "Install it with:\n"
        "  python3 -m pip install huggingface_hub\n"
        "Then rerun this script.\n"
    )

REPO_ID = "wangsen99/LMEE-Bench"


def json_summary(value: Any, depth: int = 0, max_depth: int = 3) -> Any:
    """Compact structural summary without dumping huge JSON blobs."""
    if depth >= max_depth:
        if isinstance(value, dict):
            return {"type": "dict", "keys": list(value.keys())[:30]}
        if isinstance(value, list):
            return {"type": "list", "length": len(value)}
        return {"type": type(value).__name__, "example": value}

    if isinstance(value, dict):
        out = {"type": "dict", "keys": list(value.keys())[:50]}
        children = {}
        for k, v in list(value.items())[:15]:
            children[k] = json_summary(v, depth + 1, max_depth)
        out["children"] = children
        return out

    if isinstance(value, list):
        out = {"type": "list", "length": len(value)}
        if value:
            out["first"] = json_summary(value[0], depth + 1, max_depth)
        return out

    return {"type": type(value).__name__, "example": value}


def recursive_candidate_sequences(
    value: Any,
    path: str = "$",
    results: list | None = None,
) -> list:
    """
    Find list-of-dict structures that might represent a trajectory / timestep
    sequence. We only report candidates; we do not assume the schema.
    """
    if results is None:
        results = []

    trajectory_words = {
        "t", "time", "timestamp", "step", "frame", "action",
        "position", "rotation", "pose", "image", "rgb",
        "observation", "agent_state", "state"
    }

    if isinstance(value, list):
        if len(value) >= 5 and all(isinstance(x, dict) for x in value[: min(5, len(value))]):
            keys = set()
            for x in value[: min(5, len(value))]:
                keys.update(str(k).lower() for k in x.keys())

            hits = sorted(k for k in keys if any(w in k for w in trajectory_words))
            results.append({
                "path": path,
                "length": len(value),
                "sample_keys": sorted(keys)[:60],
                "trajectory_like_key_hits": hits,
            })

        for i, item in enumerate(value[:20]):
            recursive_candidate_sequences(item, f"{path}[{i}]", results)

    elif isinstance(value, dict):
        for k, v in list(value.items())[:80]:
            recursive_candidate_sequences(v, f"{path}.{k}", results)

    return results


def extract_stop_index(image_path: str | None) -> int | None:
    if not image_path:
        return None
    m = re.search(r"/(\d+)_stop_for_", image_path)
    return int(m.group(1)) if m else None


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard"],
        default="easy",
    )
    parser.add_argument("--num-tasks", type=int, default=3)
    parser.add_argument("--out-dir", default="lmee_probe")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    dl_dir = out_dir / "downloaded"
    out_dir.mkdir(parents=True, exist_ok=True)
    dl_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()

    print(f"Listing files in {REPO_ID} ...")
    files = api.list_repo_files(REPO_ID, repo_type="dataset")

    prefix = f"task_test/{args.difficulty}/"
    config_files = sorted(
        f for f in files
        if f.startswith(prefix) and f.endswith("/config.json")
    )

    if not config_files:
        raise SystemExit(
            f"No task config files found under {prefix}. "
            "The repository layout may have changed."
        )

    selected = config_files[: args.num_tasks]

    print(f"Found {len(config_files)} {args.difficulty} tasks.")
    print(f"Probing first {len(selected)} tasks only.\n")

    report = {
        "repo_id": REPO_ID,
        "difficulty": args.difficulty,
        "num_available_tasks": len(config_files),
        "num_probed_tasks": len(selected),
        "tasks": [],
    }

    for task_num, config_rel in enumerate(selected, 1):
        task_root = config_rel[: -len("/config.json")]
        qa_rel = f"{task_root}/success/task_QA.json"
        task_json_rel = f"{task_root}/success/task.json"

        print("=" * 90)
        print(f"TASK {task_num}: {task_root.split('/')[-1][:75]}")
        print("=" * 90)

        item = {
            "task_root": task_root,
            "files": {},
        }

        for label, relpath in [
            ("config", config_rel),
            ("task_QA", qa_rel),
            ("task", task_json_rel),
        ]:
            if relpath not in files:
                print(f"[missing] {relpath}")
                item["files"][label] = {"exists": False}
                continue

            print(f"[download] {relpath}")
            local = hf_hub_download(
                repo_id=REPO_ID,
                filename=relpath,
                repo_type="dataset",
                local_dir=str(dl_dir),
            )
            data = load_json(local)

            file_report = {
                "exists": True,
                "relative_path": relpath,
                "local_path": local,
                "summary": json_summary(data),
                "candidate_sequences": recursive_candidate_sequences(data),
            }

            # Pull out the fields we already know are useful.
            if label == "config" and isinstance(data, dict):
                file_report["known_fields"] = {
                    "Task instruction": data.get("Task instruction"),
                    "Scene": data.get("Scene"),
                    "Difficulty": data.get("Difficulty"),
                    "Subtask list": data.get("Subtask list"),
                    "Object": data.get("Object"),
                }

            if label == "task_QA" and isinstance(data, dict):
                qas = data.get("QAs", [])
                file_report["known_fields"] = {
                    "Task instruction": data.get("Task instruction"),
                    "Scene": data.get("Scene"),
                    "Object_id": data.get("Object_id"),
                    "Difficulty": data.get("Difficulty"),
                    "qa_count": len(qas) if isinstance(qas, list) else None,
                    "qas": [],
                }
                if isinstance(qas, list):
                    for qa in qas[:10]:
                        if not isinstance(qa, dict):
                            continue
                        file_report["known_fields"]["qas"].append({
                            "category": qa.get("category"),
                            "question": qa.get("question"),
                            "answer": qa.get("answer"),
                            "open_answer": qa.get("open_answer"),
                            "object_id": qa.get("object_id"),
                            "image_path": qa.get("image_path"),
                            "stop_index": extract_stop_index(qa.get("image_path")),
                        })

            item["files"][label] = file_report

            if label == "task":
                candidates = file_report["candidate_sequences"]
                print(f"  task.json top-level type: {type(data).__name__}")
                if isinstance(data, dict):
                    print(f"  top-level keys: {list(data.keys())[:30]}")
                elif isinstance(data, list):
                    print(f"  top-level list length: {len(data)}")

                print(f"  candidate trajectory-like sequences found: {len(candidates)}")
                for c in candidates[:8]:
                    print(
                        f"    {c['path']}  len={c['length']}  "
                        f"hits={c['trajectory_like_key_hits'][:12]}"
                    )

        report["tasks"].append(item)
        print()

    report_path = out_dir / "probe_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 90)
    print("DONE")
    print("=" * 90)
    print(f"Report: {report_path}")
    print(
        "\nNext: send me the terminal output, or upload probe_report.json. "
        "Then I can write the exact LMEE-to-StateMem extractor without guessing "
        "the trajectory schema."
    )


if __name__ == "__main__":
    main()
