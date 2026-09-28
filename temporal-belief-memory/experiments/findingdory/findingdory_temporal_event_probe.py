#!/usr/bin/env python3
"""
FindingDory temporal-event probe
================================

Purpose:
Inspect the *native* temporal/order metadata already present in FindingDory,
especially:
  - place_oracle_action_seq
  - place_valid_receps
  - timestep / appearance-order / relative-order task definitions
  - whether event order can be mapped cleanly to target objects/receptacles

This does NOT download anything and does NOT require Habitat.

Expected existing files:
  findingdory_probe/metadata/findingdory/train/episodes.json.gz
  findingdory_probe/metadata/findingdory/val/episodes.json.gz

Run:
    cd ~/Downloads
    python3 findingdory_temporal_event_probe.py

Output:
    findingdory_temporal_probe/
      temporal_event_report.json
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_episodes(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    return data["episodes"]


def preview(x: Any, limit: int = 2500) -> str:
    try:
        s = json.dumps(x, ensure_ascii=False)
    except Exception:
        s = repr(x)
    return s if len(s) <= limit else s[:limit] + "...<truncated>"


def type_signature(x: Any) -> str:
    if isinstance(x, list):
        if not x:
            return "list[empty]"
        child_types = sorted({type_signature(v) for v in x[:20]})
        return "list[" + ",".join(child_types[:8]) + "]"
    if isinstance(x, dict):
        return "dict{" + ",".join(list(x.keys())[:12]) + "}"
    return type(x).__name__


def recurse_scalars(x: Any, path="$", out=None, max_items=200):
    if out is None:
        out = []
    if len(out) >= max_items:
        return out

    if isinstance(x, dict):
        for k, v in x.items():
            recurse_scalars(v, f"{path}.{k}", out, max_items)
            if len(out) >= max_items:
                break
    elif isinstance(x, list):
        for i, v in enumerate(x[:50]):
            recurse_scalars(v, f"{path}[{i}]", out, max_items)
            if len(out) >= max_items:
                break
    else:
        out.append({
            "path": path,
            "type": type(x).__name__,
            "value": x,
        })
    return out


def temporal_tasks(ep):
    out = []
    for key, spec in (ep.get("instructions") or {}).items():
        if not isinstance(spec, dict):
            continue

        task_type = str(spec.get("task_type", ""))
        if any(tok in task_type for tok in (
            "timestep",
            "appearance_order",
            "right_after",
            "right_before",
            "after_N",
            "before_N",
            "between",
            "longest_interaction",
            "shortest_interaction",
        )):
            out.append({
                "instruction_key": key,
                "task_id": spec.get("task_id"),
                "task_type": task_type,
                "lang": spec.get("lang"),
                "goal_expr": spec.get("goal_expr"),
                "sampled_objects": spec.get("sampled_objects"),
                "sampled_receps": spec.get("sampled_receps"),
                "sequential_goals": spec.get("sequential_goals"),
            })
    return out


def summarize_split(episodes, split):
    action_type_counts = Counter()
    action_length_counts = Counter()
    valid_recep_type_counts = Counter()

    num_targets_vs_action_len = Counter()
    num_targets_vs_valid_len = Counter()

    examples = []
    temporal_examples = []
    scalar_examples = []

    for ep in episodes:
        seq = ep.get("place_oracle_action_seq")
        valid = ep.get("place_valid_receps")
        targets = ep.get("targets") or {}

        action_type_counts[type_signature(seq)] += 1
        valid_recep_type_counts[type_signature(valid)] += 1

        seq_len = len(seq) if isinstance(seq, (list, dict)) else None
        valid_len = len(valid) if isinstance(valid, (list, dict)) else None
        n_targets = len(targets)

        if seq_len is not None:
            action_length_counts[str(seq_len)] += 1
            num_targets_vs_action_len[f"{n_targets}->{seq_len}"] += 1

        if valid_len is not None:
            num_targets_vs_valid_len[f"{n_targets}->{valid_len}"] += 1

        if len(examples) < 10:
            examples.append({
                "episode_id": ep.get("episode_id"),
                "scene_id": ep.get("scene_id"),
                "num_targets": n_targets,
                "object_labels": ep.get("info", {}).get("object_labels"),
                "target_handles": list(targets.keys()),
                "target_receptacles": ep.get("target_receptacles"),
                "goal_receptacles": ep.get("goal_receptacles"),
                "place_valid_receps": valid,
                "place_oracle_action_seq": seq,
                "place_oracle_action_seq_type": type_signature(seq),
                "place_oracle_action_seq_len": seq_len,
                "nav_goal_pos_preview": preview(ep.get("nav_goal_pos"), 1800),
                "nav_goal_rot_preview": preview(ep.get("nav_goal_rot"), 1800),
            })

        tt = temporal_tasks(ep)
        if tt and len(temporal_examples) < 8:
            temporal_examples.append({
                "episode_id": ep.get("episode_id"),
                "num_targets": n_targets,
                "temporal_tasks": tt[:20],
            })

        if seq is not None and len(scalar_examples) < 10:
            scalar_examples.append({
                "episode_id": ep.get("episode_id"),
                "sequence_scalar_paths": recurse_scalars(seq)[:120],
            })

    return {
        "split": split,
        "episode_count": len(episodes),
        "place_oracle_action_seq_type_counts": dict(action_type_counts),
        "place_oracle_action_seq_length_counts": dict(action_length_counts),
        "place_valid_receps_type_counts": dict(valid_recep_type_counts),
        "num_targets_to_action_seq_len_counts": dict(num_targets_vs_action_len),
        "num_targets_to_place_valid_len_counts": dict(num_targets_vs_valid_len),
        "episode_examples": examples,
        "temporal_task_examples": temporal_examples,
        "action_sequence_scalar_examples": scalar_examples,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default="findingdory_probe/metadata/findingdory",
    )
    ap.add_argument(
        "--out-dir",
        default="findingdory_temporal_probe",
    )
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    report = {
        "purpose": (
            "Determine whether FindingDory metadata alone exposes native "
            "interaction ordering/timing strongly enough to replace the "
            "controlled transition-time assumption in StateMem experiments."
        ),
        "splits": {},
    }

    for split in ("train", "val"):
        p = root / split / "episodes.json.gz"
        if not p.exists():
            raise SystemExit(f"Missing: {p}")

        print(f"Loading {split}: {p}")
        eps = load_episodes(p)
        report["splits"][split] = summarize_split(eps, split)

    out_path = out / "temporal_event_report.json"
    out_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("=" * 92)
    print("FINDINGDORY TEMPORAL-EVENT PROBE COMPLETE")
    print("=" * 92)

    for split, sr in report["splits"].items():
        print(f"{split}: {sr['episode_count']} episodes")
        print("  action-seq types:")
        for k, v in sr["place_oracle_action_seq_type_counts"].items():
            print(f"    {k}: {v}")

        print("  most common target-count -> action-seq-length:")
        pairs = Counter(sr["num_targets_to_action_seq_len_counts"])
        for k, v in pairs.most_common(10):
            print(f"    {k}: {v}")
        print()

    print(f"Report: {out_path}")
    print()
    print(
        "Upload temporal_event_report.json next. "
        "If the action sequence maps cleanly to object/receptacle interactions, "
        "we can replace controlled transition timing with native FindingDory "
        "event order before installing full Habitat."
    )


if __name__ == "__main__":
    main()
