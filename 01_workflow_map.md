# Workflow Map — Agent-Aware Episodic Visual Memory
Post-decisions (2026-09-01): single-agent scope, FindingDory as substrate, no
new benchmark. This document lays out the phases, the papers each phase leans
on, what to take from each paper, and the concrete deliverables.

Timing is deliberately not specified per phase — the Sept 24 check-in is a
credibility milestone, not a completion milestone. See "Sept 24 subset"
callouts at the end for what to prioritize before that date.

Golden rule: **do not skip Phase 0.** Every phase after it assumes we have
actually read the three anchor papers and their repos. Skipping Phase 0 is
how you end up like the friend's doc — a clean plan that quietly walked past
every landmine.

---

## Anchor papers (the three we keep returning to)

| Short name | ID | Role in this project |
|---|---|---|
| **FindingDory** | arXiv 2506.15635 | The benchmark. Provides episodes, tasks, ground-truth QA, and a published baseline number to beat. |
| **LMEE / MemoryExplorer** | arXiv 2601.10744 (CVPR 2026) | The closest competitor. Their method is training-based (RL-finetuned MLLM). Our angle is the training-free counterpart. |
| **GOAT-Bench** | arXiv 2404.06609 (CVPR 2024) | The infrastructure & competitive landscape reference. Where most zero-shot scene-graph memory methods (GSMem, MSGNav, etc.) get benchmarked — tells us what "structured memory" claims are already staked. |

Secondary references — pull individual ideas from these, don't read cover to
cover:
- MA-EgoQA — original multi-agent shared/per-agent memory framing (deferred; multi-agent is out for MVP)
- WorldMM — multiple memory types (episodic + semantic + visual) as a design pattern
- Ego-R1 — retrieval-as-iterative-decision (informs A5 option 3)
- UniversalRAG — RAG over multiple modalities/granularities
- Zero-shot scene-graph memory papers on GOAT-Bench (GSMem, MSGNav, SSMG-Nav, EvoMemNav, DyNaVLM) — READ THESE to know what "structured memory beats flat retrieval" claims are already published

---

## Phase 0 — Ground truth reading & setup
**Goal:** Stop guessing. Nail down what FindingDory actually ships, what LMEE actually claims, and what our machine can actually run.

### 0.1 Read the three anchor papers properly
Not the abstract. The whole paper + skim of their released code.

- **FindingDory** — extract:
  - Task taxonomy: exactly which of the 60 tasks require *temporal* reasoning vs. *spatial* vs. *state-change* vs. simple recall. This directly informs A5.
  - Data format: do they ship pre-collected trajectories (frames + poses + timestamps + agent metadata) or must we re-run collection? This is the single biggest scope question left.
  - Baseline numbers: their VLM-baseline scores on val (per task category, if reported). These become our reference to beat, not just "the whole benchmark score."
  - Eval protocol: exact metric definitions, split sizes, seeds.
  - Code repo: entry points for (a) loading an episode, (b) running the baseline, (c) computing metrics.

- **LMEE / MemoryExplorer** — extract:
  - Exact task categories they evaluate on (do they overlap with FindingDory's, or is LMEE-Bench different?).
  - Their memory representation — is it structured, unstructured, learned, or something else? This is what our method is directly compared against.
  - What their "active memory querying" actually looks like at inference time — is it tool-use style, or a retrieval scoring head?
  - Which of their results are training-dependent (RL) vs. would still hold zero-shot. That's the gap we position into.

- **GOAT-Bench** — extract:
  - Only what we need: how the community reports numbers on this benchmark, and how the zero-shot scene-graph memory papers built on top of it structure their memory. We are NOT running on GOAT-Bench — this is competitive-landscape reading.
  - Skim 2-3 of GSMem/MSGNav/SSMG-Nav to see: what memory fields do THEY store? If they already store object + location + timestamp, then "our" structured schema isn't novel and A5 must resolve to temporal/state-change or iterative retrieval as the differentiator.

### 0.2 Compute reality check on the cluster
- Confirm we can install Habitat-Lab + HSSD assets on the cluster (headless rendering).
- Confirm we can load one FindingDory episode end-to-end and see the frames.
- Time a single VLM caption call (whichever we're planning to use) — this determines whether we can caption every frame or must subsample.
- Pick one VLM commitment and stick with it (GPT-4V / Claude / Gemini / open-weight like LLaVA/Qwen-VL). Switching later is expensive.

### 0.3 Deliverables from Phase 0
- `docs/01_findingdory_notes.md` — task taxonomy + which tasks target temporal reasoning; baseline numbers per category; what their code ships.
- `docs/02_lmee_notes.md` — their memory rep, their eval, the training-free gap.
- `docs/03_prior_art_scan.md` — one page per scene-graph memory paper: what they store, what they claim, what we can/can't reuse.
- `docs/04_env_check.md` — Habitat installs, one episode loads, one caption returns.

### 0.4 What we CAN'T write in stone until Phase 0 completes
- The exact task subset for evaluation (depends on 0.1 findings)
- The A4 margin (depends on FindingDory's baseline numbers)
- The A5 claim (depends on prior-art scan)
- Whether Phase 1 needs a data-collection step (depends on whether FindingDory ships trajectories)

---

## Phase 1 — Data spine & baseline reproduction
**Goal:** Reproduce FindingDory's own baseline number on a small subset. If we can't reproduce their published result, nothing we build on top will be trusted.

### 1.1 Task subset selection
- Choose ~5-10 tasks from FindingDory that specifically require memory (per Phase 0 taxonomy notes).
- Choose ~20-30 val episodes as our working subset. Not the full 100 — cluster time isn't free.
- Freeze this subset. Write it down. Don't drift.

### 1.2 Data loading pipeline
- Load episodes into a canonical internal format we own: `{episode_id, frames[], timestamps[], poses[], objects_present[], ground_truth_QA[]}`.
- This is our data spine — every method downstream reads from it.
- Refer to FindingDory's repo for the loader; wrap it in our own class so we can swap benchmarks later without rewriting method code.

### 1.3 Reproduce the paper's VLM baseline
- Run FindingDory's own VLM baseline on our subset.
- Compare our reproduced number vs. their published number for the same task.
- Acceptable delta: within a few points (VLM version and sampling seed cause drift).
- If it doesn't reproduce, STOP and debug — do not proceed to Phase 2 with a broken baseline.

### 1.4 Deliverables
- `data/subset_manifest.json` — the exact episode IDs and task IDs we're using.
- `src/data/loader.py` — canonical loader.
- `src/baselines/vlm_native.py` — reproduces FindingDory's VLM baseline.
- `results/phase1_reproduction.md` — table: our number vs. their number.

---

## Phase 2 — Caption + embedding retrieval baseline (the bar we must beat)
**Goal:** Build the *exact* pipeline the PhD student named as the bar to beat: caption every frame → embed captions → top-k retrieval → LLM answer.

This is not our method. This is the strawman. It has to be strong so beating it means something.

### 2.1 Frame sampling
- Fixed interval (e.g., every N frames) as the default.
- Cache captions to disk — captioning is the expensive step; we'll re-run retrieval and answering many times.

### 2.2 Captioning
- Use the VLM chosen in Phase 0.2.
- Store captions with `{episode_id, frame_id, timestamp, caption_text}` — minimum viable structure so we can later compare against our richer schema.

### 2.3 Embedding + retrieval
- Standard text embedding (e.g., OpenAI, sentence-transformers, or whatever matches the VLM stack).
- Vector index (FAISS or similar).
- Top-k retrieval on the question text.

### 2.4 Answering
- Feed top-k retrieved captions + the question into an LLM. Same LLM family used everywhere in the project (don't mix providers for the answering step — that's a hidden confound).

### 2.5 Baseline results
- Run this pipeline on the same Phase 1 subset.
- Report accuracy per task category.
- These numbers ARE what our method must beat.

### 2.6 Deliverables
- `src/baselines/caption_retrieval.py`
- `data/captions_cache/` — cached captions to save cluster time
- `results/phase2_baseline.md` — the target scores broken down by task category (temporal / spatial / state-change / recall)

### Papers to refer to in Phase 2
- UniversalRAG — general RAG-over-modalities pattern (concept-level)
- Ego-R1 — how egocentric retrieval is typically structured
- Do NOT overengineer. The point is a *fair, strong strawman*.

---

## Phase 3 — Structured memory method (the actual contribution)
**Goal:** Build the structured, temporally-aware memory system and beat the Phase 2 baseline on temporal / state-change tasks specifically.

Exact schema deferred until A5 is confirmed post-Phase 0, but the minimum set:
`{frame_id, timestamp, location_estimate, objects_detected, caption, embedding, state_change_flag}`.

### 3.1 Enrichment layer
- On top of the captioned frames from Phase 2, add:
  - Object-level extraction (from caption text via LLM, or from a detector — decide based on Phase 0 findings)
  - Timestamp (already have from Phase 1 loader)
  - Location estimate (from pose in Phase 1 loader)
  - State-change detection: compare successive observations of the same object

### 3.2 Structured retrieval
- Route queries by type:
  - Temporal ("last seen", "before/after") → filter by object, sort by timestamp
  - Spatial ("in the kitchen") → filter by location
  - Recall ("did I see X") → structured lookup, fall back to embedding
- Falls back to Phase 2's embedding retrieval when structured filters don't match — this keeps the comparison honest.

### 3.3 Answering
- Same LLM as Phase 2.4. Only the retrieved evidence changes.
- Critical: keep the answering step identical between baseline and method, so any accuracy delta is attributable to memory, not to prompt engineering.

### 3.4 Deliverables
- `src/method/structured_memory.py`
- `src/method/query_router.py`
- `results/phase3_method_vs_baseline.md` — head-to-head table

### Papers to refer to in Phase 3
- **LMEE / MemoryExplorer** — their memory schema is the closest thing to what we're building. Steal the *shape* of their representation, not their training pipeline.
- **Scene-graph memory papers (GSMem, MSGNav, SSMG-Nav)** — how they represent object-relation-location structure. Take the representation ideas, not the full 3D scene-graph machinery.
- **MA-EgoQA** — per-agent field design (useful even for single-agent, because it teaches you what "identity-tagged memory entries" look like structurally; we just have one identity for now).
- **WorldMM** — the concept of separating episodic vs. semantic memory. Even in single-agent we probably want event log + object-level summary.

---

## Phase 4 — Ablations (why does it work / does it actually work)
**Goal:** Prove that the win in Phase 3 comes from a specific memory field, not from lucky prompt shuffling.

### 4.1 Field ablations
Turn off one field at a time from the structured schema and re-run:
- No timestamp → tests whether temporal structure matters
- No location → tests whether spatial structure matters
- No state-change → tests whether change detection matters
- No object extraction → collapses to Phase 2 baseline (sanity check)

### 4.2 Retrieval budget ablations
- Vary top-k
- Vary caption sampling rate

### 4.3 Deliverables
- `results/phase4_ablations.md` — per-field, per-task-category impact table
- This is the section a reviewer will look at hardest. Don't cut corners.

---

## Phase 5 — Robustness (optional for MVP, expected for the paper)
- Noisy captions: intentionally degrade caption quality, measure how gracefully the structured memory recovers vs. flat retrieval.
- Missing detections: drop N% of frames.
- Distractor objects: add unrelated objects to the scene.

### Papers to refer to in Phase 5
- Grounded Multi-Hop VideoQA — evidence-grounding evaluation ideas

---

## Phase 6 — Write-up & reproducibility
- Internal report (not the paper — the checkpoint document for the professor).
- Repo cleanup, environment pinning, one-command reproduction of Phase 3's headline table.
- A `RESULTS.md` that a reviewer can read in 5 minutes.

---

## Sept 24 subset — what to prioritize before the check-in

The Sept 24 milestone is a **credibility checkpoint, not a finished paper**. The minimum honest artifact that shows real progress is:

- Phase 0 complete (all four deliverables)
- Phase 1 complete (baseline reproduces within reason)
- Phase 2 complete (caption+embedding baseline numbers on our subset)
- Phase 3 at least *started* with one honest data point — even a single task type where structured memory beats or loses to the baseline is a real result

What would be *bad* to bring on Sept 24:
- A big system architecture with no numbers
- Numbers from an unreproduced baseline
- Multi-agent experiments (out of scope; will look unfocused)
- Any claim we haven't ablated

What would be *good* even if partial:
- "We reproduced FindingDory's baseline within X points on Y episodes"
- "Caption+embedding baseline gets Z on temporal tasks"
- "Preliminary structured-memory method changes that by ∆ on N questions — here's the table, here's what we plan to ablate next"

Honest partial > polished imaginary.

---

## Dependencies between phases
```
Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6
                                   │
                                   └─ blocks on A5 (which memory field is
                                      the differentiator), which resolves
                                      after Phase 0 prior-art scan
```

## What's still open going into Phase 0
- **A4 (success margin)** — cannot be set until Phase 1.3 reveals FindingDory's own baseline scores on our subset. Propose a margin then, don't guess now.
- **A5 (which specific claim)** — resolves after Phase 0.1 prior-art scan.
- Everything else is decided.
