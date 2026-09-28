#!/usr/bin/env python3
"""
Extract REAL object rearrangement transitions from FindingDory metadata.

Uses:
  findingdory_probe/metadata/findingdory/{train,val}/episodes.json.gz
  findingdory_probe/metadata/findingdory/{train,val}/transformations.npy

For each target object:
- initial pose comes from rigid_objs -> transformations.npy
- goal pose comes from episode["targets"]
- start receptacle comes from name_to_receptacle
- goal receptacle is aligned from goal_receptacles when lengths match

No Habitat install or scene assets required.

Run:
    cd ~/Downloads
    python3 findingdory_real_transition_extract.py

Outputs:
    findingdory_transitions/
      real_object_transitions.jsonl
      extraction_report.json
"""

from __future__ import annotations
import argparse, gzip, json, math
from collections import Counter
from pathlib import Path
import numpy as np


def load_episodes(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)["episodes"]


def xyz_from_3x4(m):
    a = np.asarray(m, dtype=float)
    if a.shape != (3,4):
        raise ValueError(f"expected 3x4, got {a.shape}")
    return [float(a[0,3]), float(a[1,3]), float(a[2,3])]


def xyz_from_4x4(m):
    a = np.asarray(m, dtype=float)
    if a.shape != (4,4):
        raise ValueError(f"expected 4x4, got {a.shape}")
    return [float(a[0,3]), float(a[1,3]), float(a[2,3])]


def dist(a,b):
    return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))


def recep_category_map(ep):
    out = {}
    fields = [
        "candidate_start_receps",
        "candidate_start_receps_noninteracted",
        "candidate_goal_receps",
        "candidate_goal_receps_noninteracted",
    ]
    for field in fields:
        for r in ep.get(field, []) or []:
            if isinstance(r, dict) and r.get("object_name"):
                out[r["object_name"]] = r.get("object_category")
    return out


def normalize_recep_entry(x):
    if isinstance(x, list) and x:
        return x[0]
    if isinstance(x, str):
        return x
    return None


def quantiles(vals):
    if not vals:
        return {}
    a = np.asarray(vals, dtype=float)
    return {
        "min": float(np.min(a)),
        "p25": float(np.percentile(a,25)),
        "median": float(np.median(a)),
        "p75": float(np.percentile(a,75)),
        "p90": float(np.percentile(a,90)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="findingdory_probe/metadata/findingdory")
    ap.add_argument("--out-dir", default="findingdory_transitions")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_rows = []
    report = {
        "root": str(root),
        "splits": {},
        "warnings": [],
        "interpretation": (
            "These are native FindingDory/Habitat rearrangement endpoint transitions: "
            "initial rigid-object pose -> target/goal pose. They are not synthetic state overlays."
        )
    }

    for split in ("train","val"):
        ep_path = root / split / "episodes.json.gz"
        tr_path = root / split / "transformations.npy"
        if not ep_path.exists() or not tr_path.exists():
            raise SystemExit(f"Missing {ep_path} or {tr_path}")

        episodes = load_episodes(ep_path)
        transforms = np.load(tr_path, mmap_mode="r")

        stats = Counter()
        displacements = []
        task_types = Counter()
        start_goal_pairs = Counter()
        sample_rows = []

        for ep in episodes:
            stats["episodes"] += 1
            scene_id = ep.get("scene_id")
            episode_id = ep.get("episode_id")

            for spec in (ep.get("instructions") or {}).values():
                if isinstance(spec, dict) and spec.get("task_type"):
                    task_types[spec["task_type"]] += 1

            rigid = {}
            for item in ep.get("rigid_objs", []) or []:
                if isinstance(item, list) and len(item) >= 3:
                    cfg, idx, handle = item[0], item[1], item[2]
                    rigid[handle] = (cfg, idx)

            targets = ep.get("targets") or {}
            target_names = list(targets.keys())
            goal_receps = ep.get("goal_receptacles") or []
            target_receps = ep.get("target_receptacles") or []
            name_to_recep = ep.get("name_to_receptacle") or {}
            rcat = recep_category_map(ep)

            if len(goal_receps) != len(target_names):
                stats["goal_recep_length_mismatch_episodes"] += 1
            if len(target_receps) != len(target_names):
                stats["target_recep_length_mismatch_episodes"] += 1

            for i, handle in enumerate(target_names):
                stats["targets"] += 1

                if handle not in rigid:
                    stats["target_missing_rigid_obj"] += 1
                    continue

                cfg, transform_idx = rigid[handle]

                if not isinstance(transform_idx, int) or not (0 <= transform_idx < len(transforms)):
                    stats["invalid_transform_index"] += 1
                    continue

                try:
                    start_xyz = xyz_from_3x4(transforms[transform_idx])
                    goal_xyz = xyz_from_4x4(targets[handle])
                except Exception:
                    stats["bad_transform_shape"] += 1
                    continue

                d = dist(start_xyz, goal_xyz)
                displacements.append(d)

                start_recep = name_to_recep.get(handle)
                indexed_start_recep = (
                    normalize_recep_entry(target_receps[i])
                    if i < len(target_receps) else None
                )
                goal_recep = (
                    normalize_recep_entry(goal_receps[i])
                    if i < len(goal_receps) else None
                )

                if start_recep is not None:
                    stats["has_start_receptacle"] += 1
                if goal_recep is not None:
                    stats["has_goal_receptacle"] += 1

                if start_recep == indexed_start_recep:
                    stats["start_receptacle_alignment_match"] += 1
                elif start_recep is not None and indexed_start_recep is not None:
                    stats["start_receptacle_alignment_mismatch"] += 1

                start_cat = rcat.get(start_recep)
                goal_cat = rcat.get(goal_recep)

                if start_cat and goal_cat:
                    start_goal_pairs[f"{start_cat}->{goal_cat}"] += 1

                row = {
                    "split": split,
                    "episode_id": episode_id,
                    "scene_id": scene_id,
                    "object_handle": handle,
                    "object_config": cfg,
                    "transform_index": transform_idx,
                    "start_xyz": start_xyz,
                    "goal_xyz": goal_xyz,
                    "displacement_m": d,
                    "start_receptacle": start_recep,
                    "start_receptacle_category": start_cat,
                    "goal_receptacle": goal_recep,
                    "goal_receptacle_category": goal_cat,
                    "indexed_target_receptacle": indexed_start_recep,
                    "start_receptacle_alignment_match": start_recep == indexed_start_recep,
                }
                all_rows.append(row)
                if len(sample_rows) < 8:
                    sample_rows.append(row)

        report["splits"][split] = {
            "episode_count": len(episodes),
            "transformations_shape": list(transforms.shape),
            "counts": dict(stats),
            "displacement_m": quantiles(displacements),
            "task_type_counts": dict(task_types),
            "most_common_start_goal_receptacle_pairs": start_goal_pairs.most_common(20),
            "sample_transitions": sample_rows,
        }

    jsonl = out / "real_object_transitions.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report["total_extracted_transitions"] = len(all_rows)
    rep = out / "extraction_report.json"
    rep.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("="*92)
    print("FINDINGDORY REAL TRANSITION EXTRACTION")
    print("="*92)
    for split in ("train","val"):
        s = report["splits"][split]
        c = s["counts"]
        print(
            f"{split:<5} episodes={s['episode_count']} "
            f"targets={c.get('targets',0)} "
            f"extracted={c.get('targets',0)-c.get('target_missing_rigid_obj',0)-c.get('invalid_transform_index',0)-c.get('bad_transform_shape',0)}"
        )
        print("  displacement(m):", json.dumps(s["displacement_m"]))
        print(
            "  start-receptacle alignment: "
            f"match={c.get('start_receptacle_alignment_match',0)} "
            f"mismatch={c.get('start_receptacle_alignment_mismatch',0)}"
        )
        print("  top task types:", s["task_type_counts"])
        print()

    print(f"Total extracted transitions: {len(all_rows)}")
    print(f"JSONL:  {jsonl}")
    print(f"Report: {rep}")
    print()
    print("Upload extraction_report.json next.")

if __name__ == "__main__":
    main()
