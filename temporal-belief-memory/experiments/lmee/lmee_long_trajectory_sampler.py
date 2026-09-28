#!/usr/bin/env python3
"""CPU-only LMEE multi-object RGB trajectory sampler."""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path, PurePosixPath

REPO = "wangsen99/LMEE-Bench"


def evenly_sample(items, n):
    n = min(n, len(items))
    if n == 0:
        return []
    if n == 1:
        return [items[len(items) // 2]]
    return [
        items[round(i * (len(items) - 1) / (n - 1))]
        for i in range(n)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--public-tasks", type=Path,
        default=Path.home() / "Downloads/lmee_adapter_out/tasks_public.jsonl"
    )
    parser.add_argument("--task-id", default="00800-TEEsavR23oF:0000")
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--min-per-trial", type=int, default=2)
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path.home() / "Downloads/lmee_rgb_long"
    )
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    from huggingface_hub import HfApi, hf_hub_download

    tasks = [
        json.loads(line)
        for line in args.public_tasks.read_text().splitlines()
        if line.strip()
    ]
    matches = [t for t in tasks if t["task_id"] == args.task_id]
    if len(matches) != 1:
        parser.error("Task ID not uniquely found in public metadata")

    task = matches[0]

    def normalize(s):
        return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

    api = HfApi()
    root = f"task_test/{task['difficulty']}"
    dirs = [
        entry.path
        for entry in api.list_repo_tree(
            repo_id=REPO,
            repo_type="dataset",
            path_in_repo=root,
            recursive=False
        )
    ]

    instruction = normalize(task["instruction"])
    matching = [
        d for d in dirs
        if instruction.startswith(normalize(PurePosixPath(d).name))
        and len(normalize(PurePosixPath(d).name)) > 24
    ]

    if len(matching) != 1:
        matching = [
            d for d in dirs
            if SequenceMatcher(
                None,
                instruction[:65],
                normalize(PurePosixPath(d).name)[:65]
            ).ratio() > 0.80
        ]

    if len(matching) != 1:
        parser.error(f"Could not uniquely match task folder: {matching}")

    folder = matching[0]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    metadata = hf_hub_download(
        REPO,
        f"{folder}/success/task.json",
        repo_type="dataset",
        local_dir=str(args.out_dir / "hf_files")
    )
    log = json.loads(Path(metadata).read_text())

    if (
        log.get("Scene") != task["scene"]
        or log.get("Object_id") != task["target_object_ids"]
    ):
        parser.error("Task metadata mismatch")

    events = {}
    for trial in sorted(
        log["trial"],
        key=lambda s: int(s.split("_")[-1])
    ):
        block = log["trial"][trial]

        for paths, action in zip(
            block["img_path"], block["action"]
        ):
            if isinstance(paths, str):
                paths = [paths]

            front = next(
                (
                    p for p in paths
                    if isinstance(p, str) and p.endswith("/front.png")
                ),
                None
            )
            if front is None:
                continue

            path = PurePosixPath(front)
            if path.is_absolute() or ".." in path.parts:
                parser.error(f"Unsafe image path: {front}")

            match = re.match(r"^(\d+)_", path.parent.name)
            if not match:
                continue

            t = int(match.group(1))
            event = {
                "timestep": t,
                "trial": trial,
                "action": str(action),
                "remote_path": f"{folder}/success/{trial}/{front}"
            }

            if (
                t not in events
                or (
                    events[t]["action"] != "stop"
                    and action == "stop"
                )
            ):
                events[t] = event

    timeline = [events[t] for t in sorted(events)]
    if not timeline:
        parser.error("No usable front-camera RGB frames")

    n = min(args.frames, len(timeline))
    groups = defaultdict(list)
    for event in timeline:
        groups[event["trial"]].append(event)

    if args.min_per_trial * len(groups) > n:
        parser.error("Not enough frames for the requested trial coverage")

    if args.min_per_trial == 0 or n == len(timeline):
        selected = evenly_sample(timeline, n)
    else:
        selected = {
            e["timestep"]: e
            for group in groups.values()
            for e in evenly_sample(group, args.min_per_trial)
        }

        while len(selected) < n:
            remaining = [
                e for e in timeline
                if e["timestep"] not in selected
            ]
            e = max(
                remaining,
                key=lambda e: (
                    min(
                        abs(e["timestep"] - t)
                        for t in selected
                    ),
                    -e["timestep"]
                )
            )
            selected[e["timestep"]] = e

        selected = sorted(
            selected.values(),
            key=lambda e: e["timestep"]
        )

    print("Task:", task["task_id"])
    print("Available frames:", len(timeline))
    print("Selected frames:", len(selected))
    print("Trial coverage:", dict(
        Counter(e["trial"] for e in selected)
    ))
    print("Timesteps:", [
        e["timestep"] for e in selected
    ])

    if not args.download:
        print("DRY RUN complete. Add --download to fetch RGB frames.")
        return

    def fetch(index_event):
        i, event = index_event
        path = Path(hf_hub_download(
            REPO,
            event["remote_path"],
            repo_type="dataset",
            local_dir=str(args.out_dir / "hf_files")
        )).resolve()

        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return i, str(path), digest

    fetched = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(fetch, item)
            for item in enumerate(selected)
        ]
        for future in as_completed(futures):
            i, path, digest = future.result()
            fetched[i] = (path, digest)
            print(
                f"Downloaded {len(fetched)}/{len(selected)}",
                flush=True
            )

    rows = [
        {
            "sample_index": i,
            "task_id": task["task_id"],
            **event,
            "local_path": fetched[i][0],
            "sha256": fetched[i][1]
        }
        for i, event in enumerate(selected)
    ]

    manifest = {
        "pilot_only": True,
        "official_benchmark_score": None,
        "task_id": task["task_id"],
        "scene": task["scene"],
        "instruction": task["instruction"],
        "total_available_front_frames": len(timeline),
        "selection_uses_gold_QA_labels_or_gold_image_paths": False,
        "selected_frames": rows
    }

    output = args.out_dir / "lmee_rgb_pilot_manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print("Manifest:", output)


if __name__ == "__main__":
    main()
