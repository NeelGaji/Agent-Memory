# FindingDory — Phase 0 Notes
Paper: FindingDory: A Benchmark to Evaluate Memory in Embodied Agents, arXiv 2506.15635, 2025 (published as a conference paper at ICLR 2026)
Repo: https://github.com/findingdory-benchmark/findingdory-habitat
Read on: 2026-09-23
Read by: claude code

## One-line summary
FindingDory is a Habitat-simulator benchmark of 60 templated navigation/pick-and-place recall tasks that isolates and evaluates a VLM-driven embodied agent's long-horizon memory of a prior scripted interaction history, scoring it with automatic, PDDL-verified goal-frame-selection and navigation metrics rather than free-form or multiple-choice QA [Abstract; Section 1].

## Task / benchmark definition — what is the task, precisely:
- **Input the model receives:** (1) a recorded "interaction history" — RGB(-D) frames, agent pose, and past discrete actions — captured during a separate, earlier "experience collection" phase in which a scripted oracle agent performs 2–11 pick-and-place interactions over 400–3500 frames in an HSSD indoor scene, and (2) a templated natural-language instruction that references that history (e.g. "navigate to a soft toy that you did not rearrange yesterday") [Section 3.1, Figure 1].
- **Output the model must produce:** the high-level module predicts the index (or, for multi-goal tasks, a list of indices) of the frame(s) in the interaction video that represent a valid goal viewpoint; the full hierarchical agent then converts the selected frame into low-level discrete navigation actions (MOVE_FORWARD 0.25m, TURN_LEFT/TURN_RIGHT 10°, STOP) to physically reach it [Section 3.2; Section 4.1.1–4.1.2].
- **Metric(s) used to score it:** High-Level Success Rate (HL-SR), Low-Level Success Rate (LL-SR), Success-weighted-by-Path-Length at both levels (HL-SPL / LL-SPL), and two relaxed diagnostic variants — Distance-to-Goal-only SR (DTG-SR) and Semantic Coverage SR (SC-SR) [Section 4.2; Section B.3].

## Task taxonomy (all 60 tasks)
The paper's own taxonomy is 11 task categories under 3 top-level groups — Spatial (30 single-goal tasks), Temporal (21 single-goal tasks), Multi-Goal (9 tasks) — totaling 60 templates [Section 3.1; Table 2; Figure 2 x-axis]. Mapped onto the suggested buckets:

- **Temporal reasoning** (16+2+3 = 21): *Interaction Order* (16) — e.g. "object interacted with immediately after X"; *Time-Based* (2) — e.g. "object interacted with at {HH:MM} yesterday"; *Duration Tracking* (3) — e.g. "object which took the longest time to rearrange" [Table 2; Table 5; Figure 2].
- **Spatial reasoning** (5+5 = 10): *Spatial Relationship* (5) — e.g. "object which is the farthest from your current location"; *Room Visitation* (5) — e.g. "navigate to a room that you did not visit yesterday" [Table 2; Table 5].
- **State-change detection** (6+7 = 13): *Interaction* (6) — e.g. "navigate to any object that you did not interact with yesterday" (interacted vs. uninteracted); *Conditional Interaction* (7) — e.g. "navigate to a receptacle you picked an object from" (object–receptacle relational state) [Table 2; Table 5].
- **Simple recall** (2+5 = 7): *Object Recall* (2) — e.g. "navigate to a {category}"; *Object Attribute* (5) — e.g. "navigate back to a {color} colored object you interacted with yesterday" [Table 2; Table 5].
- **Other — Multi-Goal** (6+3 = 9): *Unordered Revisitation* (6) — revisit all receptacles picked from, any order; *Ordered Revisitation* (3) — revisit all interacted objects in a specific sequence [Table 2; Figure 2].

Counts sum to 60 (21+10+13+7+9) and match the per-category task counts printed on the Figure 2 x-axis and the exhaustive per-template list in Table 5 [Figure 2; Table 5].

## Data
- **Ships pre-collected trajectories? PARTIAL.** Two separate data artifacts exist and they answer this differently:
  - The HuggingFace dataset `yali30/findingdory` (linked from the repo README) **does** ship pre-rendered trajectory data for VLM training/eval: ~85,083 rows / ~32.5 GB in parquet, with a `video` field (relative path to a rendered egocentric video clip), an `answer` field (ground-truth list of goal frame indices), and metadata (episode ID, question, task category, interaction count); a subsampled 96-frames-per-episode version is also provided [README.md dataset badge; verified via HF dataset page].
  - The main Habitat evaluation harness in this repo does **not** replay stored video — it ships only episode/task *definitions* (object/receptacle placements, PDDL goal expressions) via the `yali30/findingdory-habitat` HF dataset, and regenerates the actual RGB-D experience-collection footage live for every run by driving a scripted `OracleAgent` through the Habitat simulator (`findingdory/policies/heuristic/oracle_agent.py`), recorded by `findingdory/dataset/data_collector.py` and orchestrated from `findingdory/run_findingdory_eval.py`. Confirmed by reading these three files directly — the eval loop calls `env.reset()` / `env.step()` and `data_collector.update_metadata(observations)` inside a live Habitat `Env`, there is no code path that loads pre-rendered frames for this harness.
- **Format of one episode:** `FindingDoryEpisode` (subclass of Habitat's `RearrangeEpisode`), a JSON-serialized dataclass with fields: `object_category`, `start_recep_category`, `goal_recep_category`, `candidate_objects[_noninteracted]`, `candidate_start_receps[_noninteracted]`, `candidate_goal_receps[_noninteracted]` (each with view points/poses), `instructions` (list of `InstructionInfo`: `task_id`, `lang`, `task_type`, `goal_expr` (PDDL), `sampled_objects`, `sampled_receps`, `sequential_goals`), `rigid_objs` transforms, `nav_goal_pos`/`nav_goal_rot` [`findingdory/dataset/findingdory_dataset.py`].
- **Scene dataset used:** Habitat Synthetic Scenes Dataset (HSSD) [Section 3.2].
- **Scale:** 107 train / 30 val scenes; 1,478 train / 100 val episodes; objects from OVMM — 839/247 object instances spanning 84/72 categories (train/val); 17/16 receptacle categories (train/val); 79,213 train / 5,876 val task instances total [Section 3.2].
- **How to load one episode into Python:** `findingdory/dataset/findingdory_dataset.py` — class `FindingDoryDatasetV0(OVMMDatasetV0)`, registered as Habitat dataset `"FindingDoryDataset-v0"`, with `.from_json()` parsing the episode file and `FindingDoryEpisodeIterator` yielding `FindingDoryEpisode` objects; loaded transparently when `habitat.core.env.Env(config=cfg)` is constructed in `findingdory/run_findingdory_eval.py`.
- **QA format:** Free-form structurally — instructions are natural-language template strings, not multiple-choice, and the model must emit a JSON-like `frame_indices` list that is parsed and scored against valid index sets (see `extract_frame_indices_from_response` in `findingdory/policies/llm/vlm_agent.py`), not a text answer scored by string/semantic match. The HF training dataset stores ground truth the same way (list of acceptable frame indices), not as answer text.

## Their baseline
- **What baseline(s) do they publish numbers for:** Video LM Agent (frozen VLMs, zero-shot), Textual Memory Agent (chunk-and-summarize-then-LLM-select), Supervised Fine-Tuned Qwen ("Qwen SFT"), plus an Oracle upper bound; separately, a hierarchical Qwen2.5-VL-7B high-level module paired with either an ImageNav or Mapping-based low-level navigation policy [Section 4.1; Table 4; Figure 3d].
- **Baseline VLM/model used:** Proprietary — GPT-4o, Gemini-2.0-Flash; open-source — Qwen2.5-VL, Gemma-3, GLM-4.1V-Thinking [Section 4.1.1].
- **How many frames sampled per episode:** Default 96 frames (`chunk_size: 96` in `findingdory/config/baseline/qwen_mapper.yaml` and `qwen_imagenav.yaml`); a separate ablation sweeps 16/24/48/96/192/384/768 frames [Figure 4a; repo config files].
- **Baseline score per task category (HL-SR %, Table 4, "All Tasks" and selected categories):**
  - Oracle: 95.92 overall (up to 100 on Object Recall)
  - GPT-4o: 27.33 overall (best frozen VLM)
  - Gemini-2.0-Flash: 25.73
  - GLM-4.1V-Thinking: 23.49
  - Qwen2.5-VL: 15.14
  - Gemma-3: 13.04
  - Text Agent: 9.45 (worst)
  - Qwen SFT: 52.44 overall (best non-oracle)
  [Table 4]

## Numbers to beat / reference
- **Their strongest frozen-VLM baseline:** GPT-4o, HL-SR = 27.33% overall [Table 4; Section 5].
- **Best-performing method in their paper:** Qwen SFT, HL-SR = 52.44% overall (up to 88.5% on Time-Based, 70.6% on Interaction), vs. an Oracle ceiling of 95.92% overall / up to ~99–100% per category [Table 4].
- **Task categories where the gap between naive and best is largest:** Multi-Goal tasks (Unordered/Ordered Revisitation) — all frozen VLMs score near-zero (0.33–9.33% HL-SR) vs. Oracle ~99–100% and even the fine-tuned Qwen SFT reaches only ~20–30%, the single largest unclosed gap in the benchmark [Section 5 "Poor performance on multi-goal tasks"; Table 4].

## Direct implications for our project
- **What we can steal (idea):** the two-phase design — a separate scripted "experience collection" phase followed by an "interaction phase" — that deliberately isolates memory evaluation from exploration, which the paper argues most prior embodied-memory benchmarks fail to do [Section 3.1; Table 1 "Isolates Memory" column]. Also the idea of relaxed/diagnostic metrics (DTG-SR, SC-SR) that separate "did it find the semantically right target" from "did it localize precisely," which is a useful failure-decomposition pattern independent of Habitat.
- **What we can steal (code / repo modules — exact file paths):** `findingdory/task/measures.py` (metric/measure implementations — `HighLevelGoalSuccess`, `HighLevelDTGSuccess`, `HighLevelSemCovSuccess`, `FindingDorySPL`, `FindingDoryHighLevelSPL`); `findingdory/policies/llm/vlm_agent.py` (frame-subsampling and frame-index-extraction-from-VLM-response logic, `extract_frame_indices_from_response`); `findingdory/dataset/findingdory_dataset.py` (episode schema pattern, `InstructionInfo` dataclass) as a reference for how to structure templated, procedurally-verifiable task instructions.
- **What we must NOT copy:** the Habitat/PDDL-specific machinery (`third_party/habitat-lab` submodule, `OracleAgent`, physics-based pick-and-place, HSSD/OVMM asset dependencies) — this is a large, simulator-coupled stack not reusable outside Habitat; also should not copy their frame-index-prediction output format wholesale if our pipeline is meant to score free-text LLM answers rather than frame indices.
- **Open question this raises for A5 (which specific memory field is our differentiator):** FindingDory's memory representation is purely "raw indexed video history + pose/action log" — there is no explicit structured/compressed memory field being tested; all baselines re-derive everything from scratch each time via VLM context. This raises the open question of whether our project's differentiator should be an explicit structured memory store (vs. their implicit, re-derive-from-raw-frames approach) — worth resolving before any design work.

## Red flags / concerns
- **Trajectory collection requires meaningful compute/setup we may not have on hand:** the eval harness depends on the Habitat simulator + `habitat-lab` (git submodule, `third_party/habitat-lab`), HSSD scenes, and OVMM object assets, all downloaded via `findingdory/scripts/download_data.sh` (multi-GB HF dataset pulls, `hf download`, `git clone --recursive`); running or regenerating trajectories means running full Habitat physics simulation, not just downloading files. The pre-rendered `yali30/findingdory` HF dataset (32.5GB) partially avoids this for VLM-only experiments, but the "official" eval path (`run_findingdory_eval.py`) does not use it.
- **QA answer format is index-based, not open-ended text:** models must output frame indices scored against ground-truth index sets, not natural-language answers scored by text/semantic match. If our pipeline is "LLM answers questions in free-form text," FindingDory's evaluation code is not directly reusable and would need a translation layer (or we'd need to answer a different formulation of the same tasks).
- **LMEE-Bench overlap: could not verify.** The paper does not mention "LMEE-Bench" anywhere (checked full text via targeted query) — no basis to claim or rule out overlap. The paper's own related-work comparison (Table 1) is against MemoryMaze, MemoryGym, MultiON, SIF, OpenEQA (Active), GoatBench, and Excalibur, not LMEE-Bench — this needs a separate lookup if LMEE-Bench is a real benchmark we care about.
- **Training code for SFT/RL (Qwen SFT, GRPO cross-benchmark experiments) is not in this repo.** `findingdory-habitat` only contains the Habitat simulation + zero-shot VLM eval harness; grepping the repo found no references to loading the `yali30/findingdory` parquet dataset or any SFT/GRPO training loop. That code (used for Table 3/4 SFT numbers) appears to live elsewhere and was not located.

## Citations
- Abstract / Section 1: benchmark motivation, 60 tasks, contributions.
- Section 3.1: two-phase design (experience collection + interaction phase), task taxonomy overview, 60 templates, single vs. multi-goal.
- Section 3.2: Habitat/HSSD setup, scene/episode/object counts, agent embodiment, PDDL procedural generation.
- Table 2: the 11 task categories with example instructions and memory requirement, referenced in Section 3.1.
- Table 5: full enumeration of all templated instructions per category (Appendix).
- Section 4.1 / 4.1.1 / 4.1.2: baseline architecture, VLM list, low-level navigation policies, solvability/Oracle definition.
- Section 4.2 / Section B.3: metric definitions (HL-SR, LL-SR, HL-SPL/LL-SPL, DTG-SR, SC-SR).
- Table 4: per-category HL-SR numbers for all baselines + Oracle.
- Figure 2: task performance plot with per-category task counts (matches taxonomy counts used above).
- Figure 3(a–d): relaxed metrics, HL-SR vs HL-SPL, hierarchical policy degradation.
- Figure 4(a–b): frame-count ablation, ImageNav failure modes.
- Table 3: VSI-Bench cross-benchmark transfer results.
- Table 1 / Section 2: related-work comparison table (no LMEE-Bench mention).
- Repo files as cited inline above (`findingdory/dataset/findingdory_dataset.py`, `findingdory/dataset/data_collector.py`, `findingdory/policies/heuristic/oracle_agent.py`, `findingdory/policies/llm/vlm_agent.py`, `findingdory/task/measures.py`, `findingdory/run_findingdory_eval.py`, `findingdory/config/baseline/*.yaml`, `findingdory/scripts/download_data.sh`, `README.md`).
