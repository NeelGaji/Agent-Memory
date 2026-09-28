#!/usr/bin/env python3
"""CPU-first LMEE RGB pilot: validate timeline, make contact sheet, optional local MLX captions.

No answer paths, QA labels, oracle frames, or official benchmark scoring are read.
By default, no VLM is loaded. Tested with Python 3.13 and Pillow.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def build_contact_sheet(rows, destination, columns=4, cell_w=320, cell_h=240):
    from PIL import Image, ImageDraw, ImageOps

    nrows = (len(rows) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_w, nrows * cell_h), "#ffffff")
    draw = ImageDraw.Draw(sheet)
    for idx, row in enumerate(rows):
        with Image.open(row["local_path"]) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            thumb = ImageOps.contain(img, (cell_w - 16, cell_h - 42))
        x = (idx % columns) * cell_w
        y = (idx // columns) * cell_h
        sheet.paste(thumb, (x + (cell_w - thumb.width) // 2, y + 30))
        draw.text((x + 9, y + 8), f"Frame {idx}   timestep {row['timestep']}", fill="#000000")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, quality=90)


def get_text(result):
    if isinstance(result, str):
        return result
    return getattr(result, "text", str(result))


def make_captions(rows, model_id, out_path):
    try:
        from mlx_vlm import load, generate
        from mlx_vlm.utils import load_config
        from mlx_vlm.prompt_utils import apply_chat_template
    except ImportError as exc:
        raise SystemExit(
            "Caption mode needs your existing mlx-vlm environment. "
            "Run without --caption to make the contact sheet, or install mlx-vlm "
            "in a dedicated virtual environment. Original error: " + str(exc)
        )

    print(f"Loading local model: {model_id}", flush=True)
    model, processor = load(model_id)
    config = load_config(model_id)
    prompt = (
        "Describe ONLY visible facts in this single robot RGB image. "
        "Mention identifiable objects, visible attributes (such as on/off if visually "
        "clear), and direct spatial relationships. State when something is ambiguous. "
        "Do not invent unique object IDs, unseen objects, previous locations, "
        "motion, or state changes based on the task instruction. "
        "Keep the observation under 70 words."
    )
    formatted_prompt = apply_chat_template(processor, config, prompt, num_images=1)
    with out_path.open("w", encoding="utf-8") as stream:
        for idx, row in enumerate(rows):
            print(f"Captioning {idx + 1}/{len(rows)} (t={row['timestep']})", flush=True)
            result = generate(
                model, processor, formatted_prompt,
                [row["local_path"]],
                max_tokens=115, temp=0.0, verbose=False,
            )
            result_row = {
                "task_id": row["task_id"],
                "timestep": row["timestep"],
                "source_frame_path": row["local_path"],
                "model": model_id,
                "raw_vlm_caption": get_text(result).strip(),
                "is_structured_state_observation": False,
                "note": "Captions are unverified perceptual descriptions, NOT calibrated StateMem evidence.",
            }
            stream.write(json.dumps(result_row, ensure_ascii=False) + "\n")
            stream.flush()
    print("Caption output:", out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path,
                    default=Path.home() / "Downloads/lmee_rgb_pilot/lmee_rgb_pilot_manifest.json")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--caption", action="store_true", help="run optional local MLX VLM over all downloaded frames")
    ap.add_argument("--model", default="mlx-community/Qwen2.5-VL-7B-Instruct-4bit")
    args = ap.parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    if not manifest_path.is_file():
        ap.error(f"Manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("pilot_only") or not manifest.get("selected_frames"):
        ap.error("Expected a nonempty pilot manifest from lmee_rgb_pilot_sampler.py")
    task_id = manifest["task_id"]
    rows = []
    seen_steps = set()
    for source in manifest["selected_frames"]:
        step = int(source["timestep"])
        if step in seen_steps:
            ap.error(f"Repeated timestep {step}")
        seen_steps.add(step)
        path = Path(source["local_path"]).expanduser().resolve()
        if not path.is_file():
            ap.error(f"Missing downloaded RGB: {path}")
        rows.append({"task_id": task_id, "timestep": step, "local_path": str(path)})
    rows.sort(key=lambda row: row["timestep"])
    out_dir = (args.out_dir or manifest_path.parent / "local_probe").expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "task_id": task_id,
        "scene": manifest.get("scene"),
        "downloaded_front_frames": len(rows),
        "all_available_front_frames_in_selected_trial": manifest.get("total_available_front_frames"),
        "timesteps": [r["timestep"] for r in rows],
        "official_benchmark_score": None,
        "limitations": [
            "One selected trial and front-camera frames only.",
            "Not an official LMEE score or proof of dynamic-state tracking.",
            "No evaluation answers or oracle-reference image paths used.",
            "VLM captions require a separate, validated entity/state extractor before StateMem.",
        ],
    }
    (out_dir / "pilot_validation.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (out_dir / "chronological_frames.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    contact_sheet = out_dir / "contact_sheet.jpg"
    build_contact_sheet(rows, contact_sheet)
    print("Validated RGB frames:", len(rows))
    print("Timesteps:", summary["timesteps"])
    print("Contact sheet:", contact_sheet)
    if args.caption:
        make_captions(rows, args.model, out_dir / "per_frame_captions.jsonl")
    else:
        print("CPU-only validation complete. Add --caption for optional local MLX inference.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
