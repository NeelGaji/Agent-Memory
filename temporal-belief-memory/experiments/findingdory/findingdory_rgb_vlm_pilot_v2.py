#!/usr/bin/env python3
"""
FindingDory RGB/VLM perception pilot for StateMem
=================================================

Goal
----
Replace synthetic observations with REAL egocentric RGB frames from the
official 96-frame FindingDory video dataset and turn those frames into
structured StateMem evidence using a local VLM on Apple Silicon.

This is intentionally a PILOT before the full 100-episode run.

Native signals:
- official FindingDory validation episodes
- native start/goal receptacles
- native interaction order
- native DESNAP_OBJECT placement event
- official 96-frame egocentric videos

Model:
- mlx-community/Qwen2.5-VL-3B-Instruct-3bit
- runs locally with mlx-vlm on Apple Silicon

Outputs:
findingdory_rgb_pilot_v2/
  setup_report.json
  frames/
  rgb_vlm_predictions.jsonl
  rgb_vlm_report.json

Install:
  python3 -m pip install -U huggingface_hub datasets opencv-python mlx-vlm

Run setup + 5-episode pilot:
  python3 findingdory_rgb_vlm_pilot_v2.py --pilot-episodes 5

Prepare only (downloads/extracts data but does not run VLM):
  python3 findingdory_rgb_vlm_pilot_v2.py --prepare-only

Then scale:
  python3 findingdory_rgb_vlm_pilot_v2.py --pilot-episodes 20
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import random
import re
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

DATASET_REPO = "yali30/findingdory-subsampled-96"
MODEL_ID = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"


# ---------------------------------------------------------------------
# Imports with clear install instructions
# ---------------------------------------------------------------------

def require_imports():
    missing = []

    try:
        import cv2  # noqa
    except Exception:
        missing.append("opencv-python")

    try:
        import datasets  # noqa
    except Exception:
        missing.append("datasets")

    try:
        import huggingface_hub  # noqa
    except Exception:
        missing.append("huggingface_hub")

    if missing:
        raise SystemExit(
            "Missing packages: " + ", ".join(missing) + "\n\n"
            "Install with:\n"
            "  python3 -m pip install -U huggingface_hub datasets opencv-python mlx-vlm\n"
        )


# ---------------------------------------------------------------------
# FindingDory metadata
# ---------------------------------------------------------------------

def load_json_gz(path: Path):
    import gzip
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def load_transition_jsonl(path: Path):
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            rows[(r["split"], str(r["episode_id"]), r["object_handle"])] = r
    return rows


def action_name(item):
    if isinstance(item, list) and item:
        return str(item[0])
    return ""


def clean_object_name(config_name: str):
    s = config_name.replace(".object_config.json", "")
    s = re.sub(r"_:\d+$", "", s)
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()

    # Preserve identifiers if no human-readable words exist.
    alpha = sum(c.isalpha() for c in s)
    if alpha < 3:
        return config_name
    return s


def normalize_recep(x):
    if isinstance(x, list) and x:
        return x[0]
    if isinstance(x, str):
        return x
    return None


def build_val_events(metadata_root: Path, transition_path: Path):
    data = load_json_gz(metadata_root / "val" / "episodes.json.gz")
    episodes = data["episodes"]
    transitions = load_transition_jsonl(transition_path)

    out = {}
    verification = Counter()

    for ep in episodes:
        eid = str(ep["episode_id"])
        targets = list((ep.get("targets") or {}).keys())
        seqs = ep.get("place_oracle_action_seq") or {}
        target_receps = ep.get("target_receptacles") or []
        goal_receps = ep.get("goal_receptacles") or []

        cumulative = 0
        events = []

        if set(seqs) != {str(i) for i in range(len(targets))}:
            verification["bad_seq_keys"] += 1
            continue

        for i, handle in enumerate(targets):
            tr = transitions.get(("val", eid, handle))
            if tr is None:
                verification["missing_transition"] += 1
                break

            seq = seqs[str(i)]
            names = [action_name(x) for x in seq]
            des = [j for j, n in enumerate(names) if n == "DESNAP_OBJECT"]

            if len(des) != 1:
                verification["bad_desnap"] += 1
                break

            start_meta = normalize_recep(target_receps[i]) if i < len(target_receps) else None
            goal_meta = normalize_recep(goal_receps[i]) if i < len(goal_receps) else None

            if start_meta != tr["start_receptacle"]:
                verification["bad_start_recep"] += 1
                break
            if goal_meta != tr["goal_receptacle"]:
                verification["bad_goal_recep"] += 1
                break

            local_desnap = des[0]
            global_start = cumulative
            global_desnap = cumulative + local_desnap
            global_end = cumulative + len(seq) - 1

            events.append({
                **tr,
                "interaction_index": i,
                "object_display_name": clean_object_name(tr["object_config"]),
                "global_start": global_start,
                "global_desnap": global_desnap,
                "global_end": global_end,
                "local_desnap": local_desnap,
                "action_count": len(seq),
            })

            cumulative += len(seq)

        if len(events) == len(targets) and events:
            out[eid] = {
                "episode_id": eid,
                "scene_id": ep["scene_id"],
                "events": events,
                "total_action_steps": cumulative,
            }
            verification["good_episodes"] += 1
            verification["good_events"] += len(events)

    return out, dict(verification)


# ---------------------------------------------------------------------
# Official video dataset preparation
# ---------------------------------------------------------------------

def download_dataset_assets(cache_dir: Path):
    from huggingface_hub import hf_hub_download

    cache_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading FindingDory 96-frame metadata/mapping...")
    mapping = hf_hub_download(
        repo_id=DATASET_REPO,
        filename="episode_id_mappings.json",
        repo_type="dataset",
        local_dir=str(cache_dir),
    )

    print()
    print("Downloading official 96-frame video archive (~3.2 GB)...")
    print("This is the large one-time download.")
    videos_zip = hf_hub_download(
        repo_id=DATASET_REPO,
        filename="videos.zip",
        repo_type="dataset",
        local_dir=str(cache_dir),
    )

    return Path(mapping), Path(videos_zip)


def extract_videos(videos_zip: Path, cache_dir: Path):
    marker = cache_dir / ".videos_extracted"

    if marker.exists():
        print("Videos already extracted; skipping unzip.")
        return

    print("Extracting videos.zip...")
    with zipfile.ZipFile(videos_zip, "r") as z:
        z.extractall(cache_dir)

    marker.write_text("ok\n", encoding="utf-8")


def build_video_index(cache_dir: Path):
    idx = {}
    for p in cache_dir.rglob("*.mp4"):
        idx[p.stem] = p
    return idx


def load_hf_val_manifest():
    from datasets import load_dataset

    print("Loading lightweight FindingDory validation metadata...")
    ds = load_dataset(DATASET_REPO, split="validation")

    # Deduplicate by episode.
    out = {}
    for row in ds:
        ep = str(row["ep_id"])
        if ep not in out:
            out[ep] = {
                "ep_id": ep,
                "video": row["video"],
                "num_interactions": int(row["num_interactions"]),
            }
    return out


def recursively_collect_pairs(x, out):
    if isinstance(x, dict):
        for k, v in x.items():
            if isinstance(v, (str, int)):
                out.append((str(k), str(v)))
            recursively_collect_pairs(v, out)
    elif isinstance(x, list):
        for v in x:
            recursively_collect_pairs(v, out)


def build_episode_to_video_id(habitat_ids, hf_manifest, mapping_path):
    """
    Try direct ep_<habitat_id> first; then use the provided mapping file
    in both directions without assuming its exact schema.
    """
    hf_ids = set(hf_manifest)

    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    pairs = []
    recursively_collect_pairs(mapping, pairs)

    scalar_map = defaultdict(set)
    for a, b in pairs:
        scalar_map[a].add(b)
        scalar_map[b].add(a)

    resolved = {}
    method = Counter()

    for eid in habitat_ids:
        direct = f"ep_{eid}"
        if direct in hf_ids:
            resolved[eid] = direct
            method["direct_ep_id"] += 1
            continue

        candidates = {eid, direct}
        frontier = list(candidates)

        for x in frontier:
            for y in scalar_map.get(x, []):
                candidates.add(y)
                if not str(y).startswith("ep_"):
                    candidates.add(f"ep_{y}")

        hits = [x for x in candidates if x in hf_ids]

        if len(hits) == 1:
            resolved[eid] = hits[0]
            method["mapping_file"] += 1
        elif len(hits) > 1:
            # Prefer exact-looking numeric id.
            exact = [x for x in hits if x == direct]
            if exact:
                resolved[eid] = exact[0]
                method["mapping_file_ambiguous_direct"] += 1
            else:
                resolved[eid] = sorted(hits)[0]
                method["mapping_file_ambiguous"] += 1
        else:
            method["unresolved"] += 1

    return resolved, dict(method)


# ---------------------------------------------------------------------
# Video frames
# ---------------------------------------------------------------------

def video_frame_count(path: Path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def read_frame(path: Path, idx: int):
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(f"Could not decode frame {idx}: {path}")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def save_rgb(path: Path, arr):
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def action_to_frame(action_idx, total_actions, n_frames):
    if total_actions <= 1 or n_frames <= 1:
        return 0
    x = action_idx / max(1, total_actions - 1)
    return max(0, min(n_frames - 1, round(x * (n_frames - 1))))


def pilot_frame_specs(ep, video_path, changed_category_only=True):
    n_frames = video_frame_count(video_path)
    specs = []

    for ev in ep["events"]:
        same_cat = (
            ev.get("start_receptacle_category")
            == ev.get("goal_receptacle_category")
        )

        if changed_category_only and same_cat:
            continue

        # PRE: early enough in the native interaction sequence to bias toward
        # the start receptacle, but not the first navigation frame.
        pre_action = ev["global_start"] + max(
            1,
            min(
                ev["local_desnap"] - 1,
                int(max(1, ev["local_desnap"]) * 0.20),
            ),
        )

        # POST: just after the native placement event.
        post_action = min(
            ev["global_end"],
            ev["global_desnap"] + 2,
        )

        specs.append({
            "phase": "PRE",
            "expected_label": "START",
            "native_action_idx": pre_action,
            "frame_idx": action_to_frame(
                pre_action,
                ep["total_action_steps"],
                n_frames,
            ),
            "n_video_frames": n_frames,
            "event": ev,
        })

        specs.append({
            "phase": "POST",
            "expected_label": "GOAL",
            "native_action_idx": post_action,
            "frame_idx": action_to_frame(
                post_action,
                ep["total_action_steps"],
                n_frames,
            ),
            "n_video_frames": n_frames,
            "event": ev,
        })

    return specs


# ---------------------------------------------------------------------
# MLX VLM
# ---------------------------------------------------------------------

def load_mlx_vlm(model_id):
    try:
        from mlx_vlm import load
        from mlx_vlm.utils import load_config
    except Exception as e:
        raise SystemExit(
            "mlx-vlm is not available.\n"
            "Install with:\n"
            "  python3 -m pip install -U mlx-vlm\n\n"
            f"Original import error: {e}"
        )

    print()
    print(f"Loading local VLM: {model_id}")
    print("The 4-bit checkpoint is used because the previous 3-bit run produced degenerate text.")

    model, processor = load(model_id)
    config = load_config(model_id)

    # Qwen's recommended practical image-token ceiling. This is also a useful
    # guard against unnecessarily huge visual contexts in current mlx-vlm.
    try:
        if hasattr(processor, "image_processor") and hasattr(
            processor.image_processor, "max_pixels"
        ):
            processor.image_processor.max_pixels = 1280 * 28 * 28
            print(
                "Set processor.image_processor.max_pixels="
                f"{processor.image_processor.max_pixels}"
            )
    except Exception as e:
        print(f"Note: could not set max_pixels override: {e}")

    return model, processor, config


def extract_generation_text(result):
    if isinstance(result, str):
        return result
    text = getattr(result, "text", None)
    if isinstance(text, str):
        return text
    return str(result)


def looks_degenerate(text):
    t = (text or "").strip()
    if len(t) < 2:
        return True

    # Require some ordinary language content.
    alnum = sum(ch.isalnum() for ch in t)
    if alnum < 3:
        return True

    # Catch punctuation/token collapse patterns such as !!!!! or repeated \".
    if len(set(t)) <= 6 and len(t) >= 20:
        return True

    bad_tokens = t.count(r'\"') + t.count("!")
    if len(t) >= 40 and bad_tokens >= 15:
        return True

    return False


def run_vlm_sanity_check(model, processor, config, image_path):
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    prompt = "Describe this image in one short sentence."
    formatted = apply_chat_template(
        processor,
        config,
        prompt,
        num_images=1,
    )

    result = generate(
        model,
        processor,
        formatted,
        [str(image_path)],
        max_tokens=40,
        temp=0.0,
        verbose=False,
    )
    text = extract_generation_text(result).strip()

    print()
    print("VLM sanity-check output:")
    print(text[:500])

    if looks_degenerate(text):
        raise RuntimeError(
            "VLM SANITY CHECK FAILED: generation is still degenerate. "
            "Do not run the full pilot. Save the terminal output and send it here."
        )

    print("VLM sanity check: PASS")
    return text

def parse_jsonish(text):
    # Strip markdown fences.
    t = text.strip()
    t = re.sub(r"^```(?:json)?", "", t, flags=re.I).strip()
    t = re.sub(r"```$", "", t).strip()

    # Find first object.
    m = re.search(r"\{.*\}", t, flags=re.S)
    if m:
        t = m.group(0)

    try:
        return json.loads(t)
    except Exception:
        try:
            return ast.literal_eval(t)
        except Exception:
            return None


def vlm_classify(
    model,
    processor,
    config,
    image_path,
    object_name,
    start_category,
    goal_category,
):
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    # Short prompt on purpose: the previous run showed generation instability,
    # and we want the perception test to depend on the image rather than a long
    # instruction context.
    prompt = (
        f"Target object: {object_name}. "
        f"Old/start receptacle: {start_category}. "
        f"New/goal receptacle: {goal_category}. "
        "From this image only, classify the target object's visible state as "
        "START, GOAL, HELD, NOT_VISIBLE, or UNCERTAIN. "
        "Answer exactly LABEL|CONFIDENCE, for example GOAL|0.87."
    )

    formatted = apply_chat_template(
        processor,
        config,
        prompt,
        num_images=1,
    )

    result = generate(
        model,
        processor,
        formatted,
        [str(image_path)],
        max_tokens=32,
        temp=0.0,
        verbose=False,
    )

    text = extract_generation_text(result).strip()

    if looks_degenerate(text):
        return {
            "label": "DEGENERATE_OUTPUT",
            "confidence": 0.0,
            "raw_output": text,
        }

    allowed = {
        "START", "GOAL", "HELD", "NOT_VISIBLE", "UNCERTAIN"
    }

    # Preferred format: LABEL|0.xx
    m = re.search(
        r"\b(START|GOAL|HELD|NOT_VISIBLE|UNCERTAIN)\b"
        r"\s*(?:\||,|:|\s)\s*"
        r"([01](?:\.\d+)?)",
        text,
        flags=re.I,
    )

    if m:
        label = m.group(1).upper()
        conf = max(0.0, min(1.0, float(m.group(2))))
        return {
            "label": label,
            "confidence": conf,
            "raw_output": text,
        }

    # Fallback: accept a valid label anywhere in the output and a nearby number.
    label_match = re.search(
        r"\b(START|GOAL|HELD|NOT_VISIBLE|UNCERTAIN)\b",
        text,
        flags=re.I,
    )
    num_match = re.search(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", text)

    if label_match:
        label = label_match.group(1).upper()
        conf = float(num_match.group(1)) if num_match else 0.5
        return {
            "label": label,
            "confidence": max(0.0, min(1.0, conf)),
            "raw_output": text,
        }

    # Last fallback: JSON-ish parsing, in case the model ignores the requested
    # compact format but still produces structured output.
    parsed = parse_jsonish(text)
    if isinstance(parsed, dict):
        label = str(parsed.get("label", "")).upper().strip()
        if label in allowed:
            try:
                conf = float(parsed.get("confidence", 0.5))
            except Exception:
                conf = 0.5
            return {
                "label": label,
                "confidence": max(0.0, min(1.0, conf)),
                "raw_output": text,
            }

    return {
        "label": "PARSE_ERROR",
        "confidence": 0.0,
        "raw_output": text,
    }


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

def summarize_predictions(rows):
    labels = Counter(r["predicted_label"] for r in rows)
    total = len(rows)

    state_rows = [
        r for r in rows
        if r["predicted_label"] in {"START", "GOAL"}
    ]

    visible_rows = [
        r for r in rows
        if r["predicted_label"]
        not in {"NOT_VISIBLE", "PARSE_ERROR"}
    ]

    correct_state = sum(
        r["predicted_label"] == r["expected_label"]
        for r in state_rows
    )

    all_correct = sum(
        r["predicted_label"] == r["expected_label"]
        for r in rows
    )

    phase = {}
    for ph in ("PRE", "POST"):
        rr = [r for r in rows if r["phase"] == ph]
        state = [
            r for r in rr
            if r["predicted_label"] in {"START", "GOAL"}
        ]
        phase[ph] = {
            "n": len(rr),
            "state_judgment_coverage": (
                len(state) / len(rr) if rr else 0.0
            ),
            "state_accuracy_when_judged": (
                sum(r["predicted_label"] == r["expected_label"] for r in state)
                / len(state)
                if state else None
            ),
        }

    return {
        "num_predictions": total,
        "label_counts": dict(labels),
        "all_frame_exact_state_accuracy": (
            all_correct / total if total else None
        ),
        "state_judgment_coverage": (
            len(state_rows) / total if total else None
        ),
        "state_accuracy_when_judged": (
            correct_state / len(state_rows) if state_rows else None
        ),
        "non_not_visible_coverage": (
            len(visible_rows) / total if total else None
        ),
        "mean_self_reported_confidence": (
            sum(r["confidence"] for r in rows) / total if total else None
        ),
        "phase_metrics": phase,
        "interpretation_note": (
            "This is a perception pilot, not yet the final StateMem RGB result. "
            "Self-reported VLM confidence is provisional; the next integration "
            "should calibrate it on train/calibration scenes before testing."
        ),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

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
    ap.add_argument(
        "--cache-dir",
        default="findingdory_rgb_data",
    )
    ap.add_argument(
        "--out-dir",
        default="findingdory_rgb_pilot_v2",
    )
    ap.add_argument(
        "--pilot-episodes",
        type=int,
        default=5,
    )
    ap.add_argument(
        "--model",
        default=MODEL_ID,
    )
    ap.add_argument(
        "--prepare-only",
        action="store_true",
    )
    ap.add_argument(
        "--include-same-category",
        action="store_true",
        help=(
            "Include same-category start->goal moves. "
            "Default pilot excludes them because category-only visual prompts "
            "cannot identify two different instances of the same category."
        ),
    )
    ap.add_argument("--seed", type=int, default=20260916)
    args = ap.parse_args()

    require_imports()

    metadata_root = Path(args.metadata_root)
    transition_path = Path(args.transitions)
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    frames_dir = out_dir / "frames"

    if not (metadata_root / "val" / "episodes.json.gz").exists():
        raise SystemExit(
            f"Missing {metadata_root / 'val' / 'episodes.json.gz'}"
        )

    if not transition_path.exists():
        raise SystemExit(f"Missing {transition_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 94)
    print("FINDINGDORY RGB/VLM PERCEPTION PILOT")
    print("=" * 94)

    val_events, native_verify = build_val_events(
        metadata_root,
        transition_path,
    )

    print(
        f"Native validation metadata: "
        f"{len(val_events)} episodes, "
        f"{sum(len(x['events']) for x in val_events.values())} target events"
    )

    mapping_path, videos_zip = download_dataset_assets(cache_dir)
    extract_videos(videos_zip, cache_dir)

    hf_manifest = load_hf_val_manifest()
    video_index = build_video_index(cache_dir)

    resolved, resolution_method = build_episode_to_video_id(
        val_events.keys(),
        hf_manifest,
        mapping_path,
    )

    aligned = {}
    missing_video = []

    for eid, video_ep_id in resolved.items():
        p = video_index.get(video_ep_id)
        if p is None:
            # Fallback: locate by relative manifest path.
            rel = hf_manifest.get(video_ep_id, {}).get("video")
            if rel:
                candidates = list(cache_dir.rglob(Path(rel).name))
                if candidates:
                    p = candidates[0]

        if p is not None and p.exists():
            aligned[eid] = {
                "video_ep_id": video_ep_id,
                "video_path": str(p),
            }
        else:
            missing_video.append({
                "habitat_episode_id": eid,
                "video_ep_id": video_ep_id,
            })

    setup_report = {
        "dataset_repo": DATASET_REPO,
        "model_id": args.model,
        "native_validation_episode_count": len(val_events),
        "native_validation_event_count": sum(
            len(x["events"]) for x in val_events.values()
        ),
        "hf_validation_unique_video_ids": len(hf_manifest),
        "episode_id_resolution": resolution_method,
        "aligned_episode_count": len(aligned),
        "missing_video_count": len(missing_video),
        "missing_video_examples": missing_video[:20],
        "native_metadata_verification": native_verify,
        "important_method_note": (
            "96-frame videos are evenly downsampled by FindingDory. "
            "Native DESNAP action positions are mapped proportionally to the "
            "96-frame video timeline for this pilot; this mapping is approximate."
        ),
    }

    (out_dir / "setup_report.json").write_text(
        json.dumps(setup_report, indent=2),
        encoding="utf-8",
    )

    print()
    print(
        f"Video alignment: {len(aligned)}/{len(val_events)} "
        f"validation episodes"
    )
    print("Episode mapping:", resolution_method)

    if len(aligned) == 0:
        raise SystemExit(
            "No Habitat episodes could be aligned to video files. "
            "Upload findingdory_rgb_pilot_v2/setup_report.json."
        )

    if args.prepare_only:
        print()
        print("Prepare-only mode complete.")
        print(f"Report: {out_dir / 'setup_report.json'}")
        return

    rng = random.Random(args.seed)
    candidate_ids = sorted(aligned)

    # Prefer episodes with changed-category events.
    scored = []
    for eid in candidate_ids:
        n_changed = sum(
            e.get("start_receptacle_category")
            != e.get("goal_receptacle_category")
            for e in val_events[eid]["events"]
        )
        scored.append((n_changed, eid))

    scored.sort(key=lambda x: (-x[0], x[1]))
    top_pool = [eid for _, eid in scored[:max(args.pilot_episodes * 3, args.pilot_episodes)]]
    rng.shuffle(top_pool)
    selected = top_pool[:args.pilot_episodes]

    print()
    print("Selected pilot Habitat episodes:", ", ".join(selected))

    model, processor, config = load_mlx_vlm(args.model)

    # Sanity check one actual FindingDory frame BEFORE spending time on all
    # pilot inferences.
    sanity_eid = selected[0]
    sanity_ep = val_events[sanity_eid]
    sanity_video = Path(aligned[sanity_eid]["video_path"])
    sanity_specs = pilot_frame_specs(
        sanity_ep,
        sanity_video,
        changed_category_only=not args.include_same_category,
    )
    if not sanity_specs:
        raise SystemExit("No eligible frame found for VLM sanity check.")

    sanity_spec = sanity_specs[0]
    sanity_arr = read_frame(sanity_video, sanity_spec["frame_idx"])
    sanity_path = frames_dir / "SANITY_CHECK.jpg"
    save_rgb(sanity_path, sanity_arr)
    sanity_text = run_vlm_sanity_check(
        model,
        processor,
        config,
        sanity_path,
    )

    predictions = []

    for ep_num, eid in enumerate(selected, 1):
        ep = val_events[eid]
        video_path = Path(aligned[eid]["video_path"])

        specs = pilot_frame_specs(
            ep,
            video_path,
            changed_category_only=not args.include_same_category,
        )

        print()
        print(
            f"[episode {ep_num}/{len(selected)}] "
            f"habitat={eid} video={video_path.name} "
            f"VLM frames={len(specs)}"
        )

        for j, spec in enumerate(specs, 1):
            ev = spec["event"]
            arr = read_frame(video_path, spec["frame_idx"])

            frame_name = (
                f"ep_{eid}_obj_{ev['interaction_index']:02d}_"
                f"{spec['phase'].lower()}_f{spec['frame_idx']:03d}.jpg"
            )
            frame_path = frames_dir / frame_name
            save_rgb(frame_path, arr)

            result = vlm_classify(
                model=model,
                processor=processor,
                config=config,
                image_path=frame_path,
                object_name=ev["object_display_name"],
                start_category=ev.get("start_receptacle_category") or "unknown",
                goal_category=ev.get("goal_receptacle_category") or "unknown",
            )

            row = {
                "habitat_episode_id": eid,
                "video_ep_id": aligned[eid]["video_ep_id"],
                "video_path": str(video_path),
                "interaction_index": ev["interaction_index"],
                "object_handle": ev["object_handle"],
                "object_name": ev["object_display_name"],
                "start_receptacle_category": ev.get("start_receptacle_category"),
                "goal_receptacle_category": ev.get("goal_receptacle_category"),
                "same_receptacle_category": (
                    ev.get("start_receptacle_category")
                    == ev.get("goal_receptacle_category")
                ),
                "phase": spec["phase"],
                "expected_label": spec["expected_label"],
                "native_action_idx": spec["native_action_idx"],
                "frame_idx": spec["frame_idx"],
                "n_video_frames": spec["n_video_frames"],
                "predicted_label": result["label"],
                "confidence": result["confidence"],
                "raw_output": result["raw_output"],
                "frame_path": str(frame_path),
            }
            predictions.append(row)

            ok = row["predicted_label"] == row["expected_label"]
            mark = "✓" if ok else "·"
            print(
                f"  {j:>2}/{len(specs)} {mark} "
                f"{spec['phase']:<4} "
                f"frame={spec['frame_idx']:>2} "
                f"{ev['object_display_name'][:28]:<28} "
                f"pred={row['predicted_label']:<11} "
                f"conf={row['confidence']:.2f}"
            )

    pred_path = out_dir / "rgb_vlm_predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for r in predictions:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = summarize_predictions(predictions)
    report.update({
        "selected_habitat_episode_ids": selected,
        "model_id": args.model,
        "changed_category_only": not args.include_same_category,
        "frame_mapping": (
            "native action index / total native action steps mapped "
            "proportionally onto official evenly-downsampled video frames"
        ),
    })

    rep_path = out_dir / "rgb_vlm_report.json"
    rep_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 94)
    print("RGB/VLM PILOT COMPLETE")
    print("=" * 94)
    print(f"Predictions: {report['num_predictions']}")
    print(
        "State judgment coverage: "
        f"{100 * report['state_judgment_coverage']:.2f}%"
        if report["state_judgment_coverage"] is not None
        else "State judgment coverage: n/a"
    )
    print(
        "Accuracy when VLM makes START/GOAL judgment: "
        f"{100 * report['state_accuracy_when_judged']:.2f}%"
        if report["state_accuracy_when_judged"] is not None
        else "Accuracy when judged: n/a"
    )
    print(
        "All-frame exact-state accuracy: "
        f"{100 * report['all_frame_exact_state_accuracy']:.2f}%"
        if report["all_frame_exact_state_accuracy"] is not None
        else "All-frame exact-state accuracy: n/a"
    )
    print()
    print(f"Setup report: {out_dir / 'setup_report.json'}")
    print(f"VLM report:   {rep_path}")
    print(f"Predictions:  {pred_path}")
    print()
    print(
        "Upload setup_report.json and rgb_vlm_report.json first. "
        "If the perception pilot is healthy, we will integrate the real VLM "
        "evidence stream into StateMem and scale to the full validation set."
    )


if __name__ == "__main__":
    main()
