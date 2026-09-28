#!/usr/bin/env python3
"""
FindingDory temporal real-RGB -> StateMem
-----------------------------------------
Uses the already-tested integration utilities from
`findingdory_rgb_statemem_integration.py`, but changes the experiment in two
important ways:

1) Every RGB observation is counterbalanced:
     prompt 1: A=start receptacle, B=goal receptacle
     prompt 2: A=goal receptacle,  B=start receptacle
   The VLM never sees the words START/GOAL.

2) Multiple official RGB answer frames are kept as separate observations and
   processed in actual frame order. StateMem propagates its belief through the
   real frame gaps instead of seeing only two collapsed phase bundles.

This is still an oracle-frame experiment because FindingDory task_58/task_59
answer annotations select informative frames.

Run first:
  python3 findingdory_rgb_statemem_temporal.py --prepare-only

Then:
  python3 findingdory_rgb_statemem_temporal.py --pilot

After we inspect the pilot, use --full only if justified.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import findingdory_rgb_statemem_integration as base

MODEL_ID = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
STAYS = [0.55, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
DECAYS = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00]
METHODS = [
    "Latest",
    "Majority",
    "Recency-Calibrated",
    "Bayes-NoDynamics",
    "StateMem-Uniform",
    "StateMem-Calibrated",
]


def pick_temporal_frames(xs, k):
    xs = sorted(set(xs))
    if len(xs) <= k:
        return xs
    if k <= 1:
        return [xs[len(xs) // 2]]
    idxs = [round(i * (len(xs) - 1) / (k - 1)) for i in range(k)]
    return sorted(set(xs[i] for i in idxs))


def temporalize(eligible, frames_per_phase):
    out = {"train": {}, "val": {}}
    stats = {"train": Counter(), "val": Counter()}

    for split in ("train", "val"):
        for eid, ep in eligible[split].items():
            kept = []
            for ev in ep["events"]:
                sf = sorted(set(ev["start_frames"]))
                gf = sorted(set(ev["goal_frames"]))
                if not sf or not gf:
                    continue
                if max(sf) >= min(gf):
                    stats[split]["non_monotonic_or_overlapping"] += 1
                    continue
                s = pick_temporal_frames(sf, frames_per_phase)
                g = pick_temporal_frames(gf, frames_per_phase)
                kept.append({
                    **ev,
                    "start_frames": s,
                    "goal_frames": g,
                    "temporal_gap_frames": min(g) - max(s),
                })
                stats[split]["usable_events"] += 1
                stats[split]["selected_observations"] += len(s) + len(g)

            if kept:
                out[split][eid] = {**ep, "events": kept}
                stats[split]["eligible_episodes"] += 1

    return out, {k: dict(v) for k, v in stats.items()}


def choose_events(episodes, allowed_scenes, cap, seed, tag):
    return base.round_robin_events(
        episodes,
        allowed_scenes=allowed_scenes,
        cap=cap,
        seed=seed,
        tag=tag,
    )


def paired_key(x):
    return json.dumps({
        "model_id": x["model_id"],
        "split": x["split"],
        "episode_id": x["episode_id"],
        "interaction_index": x["interaction_index"],
        "phase": x["phase"],
        "frame_idx": x["frame_idx"],
        "prompt_order": x["prompt_order"],
    }, sort_keys=True)


def load_paired_cache(path):
    d = {}
    if not path.exists():
        return d
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                d[paired_key(r)] = r
    return d


def semantic(raw, a_state, b_state):
    if raw == "A":
        return a_state
    if raw == "B":
        return b_state
    return None


def reconcile_pair(r1, r2):
    # order1: A=START, B=GOAL
    # order2: A=GOAL, B=START
    s1 = semantic(r1, "START", "GOAL")
    s2 = semantic(r2, "GOAL", "START")
    if s1 and s2:
        if s1 == s2:
            return s1, "consistent"
        return None, "conflict"
    if s1 or s2:
        return s1 or s2, "single_judged"
    return None, "double_abstain"


def cached_prompt(cache, cache_f, model, processor, cfg, model_id,
                  split, ev, phase, frame_idx, prompt_order, image_path):
    if prompt_order == 1:
        oa, ob = ev["start_receptacle_category"], ev["goal_receptacle_category"]
        a_state, b_state = "START", "GOAL"
    else:
        oa, ob = ev["goal_receptacle_category"], ev["start_receptacle_category"]
        a_state, b_state = "GOAL", "START"

    row = {
        "model_id": model_id,
        "split": split,
        "episode_id": ev["episode_id"],
        "scene_id": ev["scene_id"],
        "interaction_index": ev["interaction_index"],
        "object_handle": ev["object_handle"],
        "phase": phase,
        "frame_idx": int(frame_idx),
        "prompt_order": prompt_order,
        "option_a": oa,
        "option_b": ob,
        "a_state": a_state,
        "b_state": b_state,
    }
    key = paired_key(row)
    if key in cache:
        return cache[key]

    raw, raw_text = base.vlm_classify(
        model, processor, cfg, [str(image_path)], oa, ob
    )
    row.update({
        "raw_label": raw,
        "semantic_state": semantic(raw, a_state, b_state),
        "raw_output": raw_text,
    })
    cache_f.write(json.dumps(row, ensure_ascii=False) + "\n")
    cache_f.flush()
    cache[key] = row
    return row


def generate_obs(events, split, out_dir, cache_path,
                 model, processor, cfg, model_id):
    cache = load_paired_cache(cache_path)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    result = []
    total = sum(len(e["start_frames"]) + len(e["goal_frames"]) for e in events)
    done = 0

    with cache_path.open("a", encoding="utf-8") as cache_f:
        for ev in events:
            timeline = [("START", x) for x in ev["start_frames"]] + \
                       [("GOAL", x) for x in ev["goal_frames"]]
            timeline.sort(key=lambda z: z[1])

            for phase, frame_idx in timeline:
                img = frames_dir / (
                    f"{split}_ep_{ev['episode_id']}_obj_{ev['interaction_index']:02d}_"
                    f"{phase.lower()}_f{frame_idx:02d}.jpg"
                )
                if not img.exists():
                    rgb = base.read_frame(Path(ev["video_path"]), frame_idx)
                    base.save_rgb(img, rgb)

                p1 = cached_prompt(cache, cache_f, model, processor, cfg,
                                   model_id, split, ev, phase, frame_idx, 1, img)
                p2 = cached_prompt(cache, cache_f, model, processor, cfg,
                                   model_id, split, ev, phase, frame_idx, 2, img)

                pred, consistency = reconcile_pair(
                    p1["raw_label"], p2["raw_label"]
                )
                result.append({
                    "split": split,
                    "episode_id": ev["episode_id"],
                    "scene_id": ev["scene_id"],
                    "interaction_index": ev["interaction_index"],
                    "object_handle": ev["object_handle"],
                    "frame_idx": int(frame_idx),
                    "gt_state": phase,
                    "pred_state": pred,
                    "consistency": consistency,
                    "raw1": p1["raw_label"],
                    "raw2": p2["raw_label"],
                    "correct": pred == phase if pred else False,
                    "temporal_gap_frames": ev["temporal_gap_frames"],
                })
                done += 1
                if done % 20 == 0 or done == total:
                    print(f"[{split}] observations {done}/{total}")
    return result


def fit_calibration(obs, min_n=10, prior=8.0):
    judged = [o for o in obs if o["pred_state"] in {"START", "GOAL"}]
    if not judged:
        raise RuntimeError("No judged calibration observations")
    global_r = sum(o["correct"] for o in judged) / len(judged)
    g = defaultdict(lambda: [0, 0])
    for o in judged:
        k = (o["consistency"], o["pred_state"])
        g[k][0] += 1
        g[k][1] += int(o["correct"])

    table, rows = {}, []
    for k, (n, c) in sorted(g.items(), key=lambda x: str(x[0])):
        if n < min_n:
            r, fallback = global_r, True
        else:
            r = (c + prior * global_r) / (n + prior)
            fallback = False
        r = max(0.05, min(0.95, r))
        table[k] = r
        rows.append({
            "consistency": k[0], "pred_state": k[1], "n": n,
            "correct": c, "empirical_accuracy": c / n,
            "calibrated_reliability": r, "fallback_to_global": fallback,
        })
    return {
        "global_reliability": global_r,
        "judged_n": len(judged),
        "coverage": len(judged) / len(obs),
        "groups": rows,
    }, table


def reliability(o, table, global_r):
    if o["pred_state"] not in {"START", "GOAL"}:
        return None
    return table.get((o["consistency"], o["pred_state"]), global_r)


def recency(history, current_frame, decay, time_scale):
    if not history:
        return None, 0.5
    score = 0.0
    for frame_idx, state, r in history:
        age = max(0, current_frame - frame_idx) / max(1e-9, time_scale)
        w = math.exp(-decay * age)
        signed = base.logit(base.clamp(r, 0.01, 0.99))
        score += w * signed * (1 if state == "GOAL" else -1)
    p = base.sigmoid(score)
    return base.hard_state_from_p(p), p


def group_transitions(obs):
    d = defaultdict(list)
    for o in obs:
        d[(o["episode_id"], o["interaction_index"], o["object_handle"])].append(o)
    out = []
    for k, rows in d.items():
        rows.sort(key=lambda x: x["frame_idx"])
        if any(r["gt_state"] == "START" for r in rows) and \
           any(r["gt_state"] == "GOAL" for r in rows):
            out.append((k, rows))
    return out


def evaluate(obs, table, global_r, stay, decay, time_scale, split_name,
             write_steps=False):
    agg = {m: Counter() for m in METHODS}
    brier = {m: [0.0, 0] for m in METHODS}
    transitions, steps = [], []

    for key, rows in group_transitions(obs):
        eid, idx, handle = key
        latest = None
        labels = []
        hist = []
        p_no = p_u = p_c = 0.5
        prev_frame = None
        goal_idx = 0
        first_goal_hit = {m: None for m in METHODS}
        last_pred = {}

        for o in rows:
            f = o["frame_idx"]
            gt = o["gt_state"]
            if prev_frame is not None:
                delta = max(0, f - prev_frame) / max(1e-9, time_scale)
                p_u = base.transition_predict(p_u, stay, delta)
                p_c = base.transition_predict(p_c, stay, delta)
            else:
                delta = 0.0

            r = reliability(o, table, global_r)
            if o["pred_state"] in {"START", "GOAL"}:
                latest = o["pred_state"]
                labels.append(latest)
                hist.append((f, latest, r))
                p_no = base.bayes_update(p_no, latest, r)
                p_u = base.bayes_update(p_u, latest, global_r)
                p_c = base.bayes_update(p_c, latest, r)

            rec_s, rec_p = recency(hist, f, decay, time_scale)
            maj = base.majority_prediction(labels)
            preds = {
                "Latest": (latest, 1.0 if latest == "GOAL" else 0.0 if latest == "START" else 0.5),
                "Majority": (maj, None),
                "Recency-Calibrated": (rec_s, rec_p),
                "Bayes-NoDynamics": (base.hard_state_from_p(p_no), p_no),
                "StateMem-Uniform": (base.hard_state_from_p(p_u), p_u),
                "StateMem-Calibrated": (base.hard_state_from_p(p_c), p_c),
            }

            this_goal_idx = None
            if gt == "GOAL":
                this_goal_idx = goal_idx
                goal_idx += 1

            for m, (pred, pg) in preds.items():
                a = agg[m]
                ok = pred == gt
                a["n"] += 1; a["correct"] += int(ok)
                a[f"{gt.lower()}_n"] += 1; a[f"{gt.lower()}_correct"] += int(ok)
                if gt == "GOAL":
                    if this_goal_idx == 0:
                        a["immediate_n"] += 1; a["immediate_correct"] += int(ok)
                    else:
                        a["late_n"] += 1; a["late_correct"] += int(ok)
                    if first_goal_hit[m] is None and pred == "GOAL":
                        first_goal_hit[m] = this_goal_idx
                if pg is not None:
                    y = 1.0 if gt == "GOAL" else 0.0
                    brier[m][0] += (pg - y) ** 2; brier[m][1] += 1
                last_pred[m] = pred
                if write_steps:
                    steps.append({
                        "split": split_name, "episode_id": eid,
                        "interaction_index": idx, "object_handle": handle,
                        "frame_idx": f, "gt_state": gt, "method": m,
                        "prediction": pred, "correct": int(ok),
                        "prob_goal": pg, "vlm_state": o["pred_state"],
                        "vlm_consistency": o["consistency"],
                        "vlm_reliability": r, "raw1": o["raw1"], "raw2": o["raw2"],
                        "delta_frame_units": delta,
                    })
            prev_frame = f

        for m in METHODS:
            a = agg[m]
            a["transitions"] += 1
            final_ok = last_pred.get(m) == "GOAL"
            a["final_n"] += 1; a["final_correct"] += int(final_ok)
            lag = first_goal_hit[m]
            if lag is None:
                a["misses"] += 1
            else:
                a["lag_sum"] += lag; a["lag_n"] += 1
            transitions.append({
                "split": split_name, "episode_id": eid,
                "interaction_index": idx, "object_handle": handle,
                "method": m,
                "last_start_frame": max(r["frame_idx"] for r in rows if r["gt_state"] == "START"),
                "first_goal_frame": min(r["frame_idx"] for r in rows if r["gt_state"] == "GOAL"),
                "num_observations": len(rows),
                "final_prediction": last_pred.get(m),
                "final_correct": int(final_ok),
                "goal_recovery_lag_observations": lag,
                "goal_missed": int(lag is None),
            })

    summary = {}
    for m, a in agg.items():
        def ratio(c, n): return a[c] / a[n] if a[n] else None
        summary[m] = {
            "accuracy": ratio("correct", "n"),
            "start_accuracy": ratio("start_correct", "start_n"),
            "goal_accuracy": ratio("goal_correct", "goal_n"),
            "immediate_goal_accuracy": ratio("immediate_correct", "immediate_n"),
            "late_goal_accuracy": ratio("late_correct", "late_n"),
            "final_goal_accuracy": ratio("final_correct", "final_n"),
            "brier": brier[m][0] / brier[m][1] if brier[m][1] else None,
            "goal_recovery_lag_observations": a["lag_sum"] / a["lag_n"] if a["lag_n"] else None,
            "transition_miss_rate": a["misses"] / a["transitions"] if a["transitions"] else None,
            "num_steps": a["n"], "num_transitions": a["transitions"],
        }
    return summary, transitions, steps


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8"); return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def pct(x):
    return "NA" if x is None else f"{100*x:.2f}%"


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--pilot", action="store_true")
    mode.add_argument("--full", action="store_true")
    ap.add_argument("--metadata-root", default="findingdory_probe/metadata/findingdory")
    ap.add_argument("--transitions", default="findingdory_transitions/real_object_transitions.jsonl")
    ap.add_argument("--cache-dir", default="findingdory_rgb_data")
    ap.add_argument("--out-dir", default="findingdory_rgb_statemem_temporal")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--seed", type=int, default=20260916)
    ap.add_argument("--frames-per-phase", type=int, default=None)
    ap.add_argument("--max-calibration-events", type=int, default=None)
    ap.add_argument("--max-tuning-events", type=int, default=None)
    ap.add_argument("--max-test-events", type=int, default=None)
    args = ap.parse_args()

    if not (args.prepare_only or args.pilot or args.full):
        args.prepare_only = True
    if args.frames_per_phase is None:
        args.frames_per_phase = 3 if args.pilot else 4
    if args.pilot:
        args.max_calibration_events = args.max_calibration_events or 20
        args.max_tuning_events = args.max_tuning_events or 20
        args.max_test_events = args.max_test_events or 24
    if args.full:
        args.max_calibration_events = args.max_calibration_events or 200
        args.max_tuning_events = args.max_tuning_events or 200

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "vlm_cache_counterbalanced.jsonl"

    print("=" * 96)
    print("FINDINGDORY TEMPORAL REAL-RGB -> STATEMEM")
    print("=" * 96)

    native, native_report = base.build_native_events(
        Path(args.metadata_root), Path(args.transitions)
    )
    supervision, supervision_stats = base.load_hf_supervision()
    eligible0, eligibility0 = base.attach_rgb_supervision(
        native, supervision, Path(args.cache_dir)
    )
    eligible, temporal_stats = temporalize(eligible0, args.frames_per_phase)

    train_scenes = {e["scene_id"] for e in eligible["train"].values()}
    val_scenes = {e["scene_id"] for e in eligible["val"].values()}
    cal_scenes, tune_scenes = base.scene_split_train(eligible["train"], args.seed)

    prepare = {
        "native_report": native_report,
        "supervision_stats": supervision_stats,
        "base_eligibility": eligibility0,
        "temporal_coverage": temporal_stats,
        "frames_per_phase": args.frames_per_phase,
        "eligible_train_episodes": len(eligible["train"]),
        "eligible_val_episodes": len(eligible["val"]),
        "train_val_scene_overlap": len(train_scenes & val_scenes),
        "calibration_tuning_scene_overlap": len(cal_scenes & tune_scenes),
        "counterbalanced_perception": True,
        "oracle_frame_selection": True,
        "temporal_rule": "retain only events with max(START frame) < min(GOAL frame); process selected frames individually in frame order",
    }
    (out_dir / "prepare_report.json").write_text(json.dumps(prepare, indent=2), encoding="utf-8")

    for split in ("train", "val"):
        s = temporal_stats[split]
        print(f"{split}: usable={s.get('usable_events',0)} eps={s.get('eligible_episodes',0)} "
              f"non_monotonic={s.get('non_monotonic_or_overlapping',0)} "
              f"obs={s.get('selected_observations',0)}")
    print("train/val scene overlap:", len(train_scenes & val_scenes))
    print("calibration/tuning scene overlap:", len(cal_scenes & tune_scenes))

    if args.prepare_only:
        print("\nNo VLM inference run. Upload:", out_dir / "prepare_report.json")
        return

    cal_events = choose_events(eligible["train"], cal_scenes, args.max_calibration_events, args.seed, "cal")
    tune_events = choose_events(eligible["train"], tune_scenes, args.max_tuning_events, args.seed, "tune")
    test_events = choose_events(eligible["val"], None, args.max_test_events, args.seed, "test")

    if not cal_events or not tune_events or not test_events:
        raise SystemExit("A required split is empty; inspect prepare_report.json")

    gaps = []
    for ev in cal_events + tune_events:
        fs = sorted(ev["start_frames"] + ev["goal_frames"])
        gaps += [b-a for a,b in zip(fs, fs[1:]) if b > a]
    time_scale = statistics.median(gaps) if gaps else 1.0

    config = {
        "mode": "pilot" if args.pilot else "full",
        "model_id": args.model,
        "frames_per_phase": args.frames_per_phase,
        "calibration_events": len(cal_events),
        "tuning_events": len(tune_events),
        "test_events": len(test_events),
        "time_scale_median_train_frame_gap": time_scale,
        "counterbalanced": True,
        "oracle_frame_selection": True,
        "validation_used_for_tuning": False,
    }
    (out_dir / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("\nselected events:", len(cal_events), len(tune_events), len(test_events))
    print("train-only frame-gap scale:", time_scale)

    model, processor, cfg = base.load_vlm(args.model)
    print("\nCalibration stream...")
    cal_obs = generate_obs(cal_events, "train_calibration", out_dir, cache_path, model, processor, cfg, args.model)
    print("\nTuning stream...")
    tune_obs = generate_obs(tune_events, "train_tuning", out_dir, cache_path, model, processor, cfg, args.model)
    print("\nOfficial validation stream...")
    test_obs = generate_obs(test_events, "official_val", out_dir, cache_path, model, processor, cfg, args.model)

    cal_report, cal_table = fit_calibration(cal_obs)
    global_r = cal_report["global_reliability"]
    (out_dir / "calibration_table.json").write_text(json.dumps(cal_report, indent=2), encoding="utf-8")

    rec_rows = []; best_decay = None; best_key = None
    for d in DECAYS:
        s,_,_ = evaluate(tune_obs, cal_table, global_r, 0.75, d, time_scale, "train_tuning")
        x = s["Recency-Calibrated"]
        rec_rows.append({"decay": d, **x})
        key = (x["goal_accuracy"], x["late_goal_accuracy"], x["accuracy"], -(x["brier"] or 999))
        if best_key is None or key > best_key:
            best_key, best_decay = key, d
    write_csv(out_dir / "tuning_recency_sweep.csv", rec_rows)

    stay_rows = []; best_stay = None; best_key = None
    for st in STAYS:
        s,_,_ = evaluate(tune_obs, cal_table, global_r, st, best_decay, time_scale, "train_tuning")
        x = s["StateMem-Calibrated"]
        stay_rows.append({"stay": st, **x})
        key = (x["goal_accuracy"], x["late_goal_accuracy"], x["accuracy"], -x["transition_miss_rate"], -(x["brier"] or 999))
        if best_key is None or key > best_key:
            best_key, best_stay = key, st
    write_csv(out_dir / "tuning_stay_sweep.csv", stay_rows)

    print("\nFrozen train-only choices:")
    print("  recency decay:", best_decay)
    print("  StateMem stay:", best_stay)
    print("  paired RGB reliability:", round(global_r, 4))
    print("  paired judgment coverage:", round(cal_report["coverage"], 4))

    summary, tr_rows, step_rows = evaluate(
        test_obs, cal_table, global_r, best_stay, best_decay,
        time_scale, "official_val", write_steps=True
    )
    result = {
        "selected_stay": best_stay,
        "selected_recency_decay": best_decay,
        "global_calibration_reliability": global_r,
        "test_events": len(test_events),
        "methods": summary,
        "important_limitation": "real counterbalanced RGB evidence is temporal, but task_58/task_59 annotations still select informative frames",
    }
    (out_dir / "test_overall.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_csv(out_dir / "test_transition_results.csv", tr_rows)
    write_csv(out_dir / "test_step_results.csv", step_rows)

    print("\n" + "="*96)
    print("TEMPORAL REAL-RGB -> STATEMEM RESULT")
    print("="*96)
    for m in METHODS:
        x = summary[m]
        print(f"{m:<22} overall={pct(x['accuracy'])} start={pct(x['start_accuracy'])} "
              f"goal={pct(x['goal_accuracy'])} immediate={pct(x['immediate_goal_accuracy'])} "
              f"late={pct(x['late_goal_accuracy'])} final={pct(x['final_goal_accuracy'])} "
              f"miss={pct(x['transition_miss_rate'])}")
    print("\nUpload test_overall.json, prepare_report.json, calibration_table.json, and the two tuning CSVs.")
    if args.pilot:
        print("Do not run --full until we inspect this pilot.")


if __name__ == "__main__":
    main()
