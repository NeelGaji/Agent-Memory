#!/usr/bin/env python3
"""Select chronological LMEE RGB sample frames without looking at QA gold labels.

Run from Mac/VS Code with huggingface_hub installed. Default is a DRY RUN.
For a tiny smoke test, use --download to fetch uniformly sampled front.png frames.
This is NOT an official LMEE evaluation and does not generate object-state evidence.
"""
import argparse
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import sys


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def task_folder_match(instruction, folder_name):
    """Instruction-derived LMEE folder names are truncated; require strong prefix match."""
    def normalize(x):
        return re.sub(r"[^a-z0-9]+", " ", x.lower()).strip()
    a, b = normalize(instruction), normalize(folder_name)
    # 'Starts with' also avoids matching unrelated tasks that share many words.
    if len(b) >= 25 and (a.startswith(b) or a[:min(60, len(a))] == b[:min(60, len(b))]):
        return True
    return len(b) >= 40 and a[:40] == b[:40] and SequenceMatcher(None, a[:len(b)], b).ratio() >= 0.83


def timestep(path):
    folder = Path(path).parent.name
    match = re.match(r"^(\d+)", folder)
    if match:
        return int(match.group(1))
    return -1


def choose_evenly(frames, wanted):
    if not frames:
        return []
    n = min(len(frames), wanted)
    if n == 1:
        return [frames[0]]
    return [frames[round(i * (len(frames) - 1) / (n - 1))] for i in range(n)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--public-tasks", type=Path, default=Path.home() / "Downloads/lmee_adapter_out/tasks_public.jsonl")
    p.add_argument("--local-dir", type=Path, default=Path.home() / "Downloads/lmee_rgb_pilot")
    p.add_argument("--frames", type=int, default=24, help="uniformly spaced RGB frames to download")
    p.add_argument("--task-id", help="optional public task ID; otherwise select first easy task with an available trajectory")
    p.add_argument("--download", action="store_true", help="download the selected frames; default prints the plan only")
    args = p.parse_args()
    if args.frames < 1:
        p.error("--frames must be >= 1")
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        p.error('Install with: python3 -m pip install "huggingface_hub<2"')

    public_tasks = read_jsonl(args.public_tasks.expanduser())
    if args.task_id:
        candidates = [t for t in public_tasks if t["task_id"] == args.task_id]
        if not candidates:
            p.error(f"Task not found in public manifest: {args.task_id}")
    else:
        candidates = [t for t in public_tasks if t.get("difficulty") == "easy"]
    if not candidates:
        p.error("No matching tasks found")

    api = HfApi()
    repo = "wangsen99/LMEE-Bench"
    # List one shallow difficulty folder at a time, rather than traversing 22 GB of file metadata.
    cached_folders = {}
    match = None
    for task in candidates:
        diff = str(task["difficulty"])
        if diff not in cached_folders:
            root = f"task_test/{diff}"
            cached_folders[diff] = [e.path for e in api.list_repo_tree(
                repo_id=repo, repo_type="dataset", path_in_repo=root, recursive=False
            ) if e.path != root]
        matches = [folder for folder in cached_folders[diff]
                   if task_folder_match(task["instruction"], Path(folder).name)]
        if len(matches) == 1:
            match = task, matches[0]
            break
        if args.task_id:
            print("No unique trajectory directory for this task. Candidate remote directories:")
            print("\n".join(cached_folders[diff][:12]))
            sys.exit(2)

    if match is None:
        print("No unambiguous public task → remote trajectory mapping found.")
        for diff, folders in cached_folders.items():
            print(diff, "examples:", folders[:5])
        sys.exit(2)

    task, folder = match
    trial = f"{folder}/success/trial_1"
    try:
        entries = list(api.list_repo_tree(repo_id=repo, repo_type="dataset", path_in_repo=trial, recursive=True))
    except Exception as exc:
        print(f"Could not list official trajectory at {trial}: {exc}")
        sys.exit(2)

    # No gold-reference_image_path, QA answer, or evaluation file is opened here.
    all_frames = sorted(
        (e for e in entries if e.path.endswith("/front.png") and timestep(e.path) >= 0),
        key=lambda e: (timestep(e.path), e.path),
    )
    if not all_frames:
        print(f"No numbered front.png frames found under {trial}.")
        print("First remote entries:", [e.path for e in entries[:12]])
        sys.exit(2)
    # Avoid multiple frames from the same timestep, if present.
    unique = {}
    for e in all_frames:
        unique.setdefault(timestep(e.path), e)
    chronological = [unique[k] for k in sorted(unique)]
    selected = choose_evenly(chronological, args.frames)
    sizes = [getattr(e, "size", None) for e in selected]
    known_bytes = sum(s for s in sizes if isinstance(s, int))

    print("LMEE RGB PILOT (NOT AN OFFICIAL BENCHMARK SCORE)")
    print("Public task ID:", task["task_id"])
    print("Instruction:", task["instruction"])
    print("Remote trial:", trial)
    print("Available chronological front frames:", len(chronological))
    print("Selected uniformly by timestep:", len(selected))
    if all(isinstance(s, int) for s in sizes):
        print(f"Selected download size: {known_bytes / (1024 * 1024):.1f} MiB")
    else:
        print("Selected download size: unknown (repository did not supply all sizes)")
    print("Selected timesteps:", [timestep(e.path) for e in selected])
    if not args.download:
        print("\nDRY RUN complete. Add --download to fetch the selected RGB frames.")
        return

    dest = args.local_dir.expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    manifest = {
        "pilot_only": True, "official_benchmark_score": None,
        "selection": "uniform chronological timestep sampling; no private QA labels accessed",
        "task_id": task["task_id"], "instruction": task["instruction"],
        "scene": task["scene"], "trial_remote_dir": trial,
        "total_available_front_frames": len(chronological),
        "selected_frames": [],
    }
    for i, e in enumerate(selected):
        path = hf_hub_download(repo_id=repo, repo_type="dataset", filename=e.path, local_dir=str(dest))
        manifest["selected_frames"].append({
            "sample_index": i, "timestep": timestep(e.path),
            "remote_path": e.path, "local_path": str(Path(path).resolve()),
        })
        print(f"[{i+1}/{len(selected)}] timestep={timestep(e.path)} {path}")
    output = dest / "lmee_rgb_pilot_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\nSaved:", output)
    print("Use images for ingestion smoke test only; do not tune on the official subset QA labels.")


if __name__ == "__main__":
    main()
