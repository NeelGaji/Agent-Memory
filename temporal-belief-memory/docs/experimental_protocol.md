# Experimental protocol and limitations

**Status as of 2026-09-26:** completed frozen real-RGB FindingDory offline state evaluation, temporal-stress replays, multi-object downstream QA; local LMEE metadata/RGB ingestion pilot. **Not** yet an official online Habitat or LMEE comparative evaluation.

## Datasets and split integrity

- Native FindingDory-Habitat train metadata: **1,478** usable episodes and **7,379** verified object-transition events. Official validation: **100** episodes and **650** native transitions at the native extraction stage. These larger counts are **not** the denominator for the real-RGB evaluation.
- RGB supervision eligibility: after START/GOAL annotation availability and temporal filtering, **1,180 training episodes** (2,284 usable transitions; 16,269 selected RGB observations) and **67 validation episodes** (**148 transitions**, **963 selected RGB observations**). The train and validation scenes do not overlap. Train calibration and train tuning scene subsets do not overlap.
- Temporal ordering rule: retain only transitions where `max(START annotated frame) < min(GOAL annotated frame)`. Select up to **4** annotated frames per phase; evaluate individually in actual frame order. Official `task_58`/`task_59` answer annotations **select informative RGB frames**. This is **oracle frame selection**, not natural agent-driven perception. The VLM does not receive ground-truth START/GOAL words in its counterbalanced prompt, but the frame *selection* is annotation-assisted.
- Local LMEE pilot: downloaded metadata subset contains **58 tasks**, **145 QAs**, **32 scenes**. Eight pilot front-camera frames were captioned; **no complete LMEE evaluation or cross-benchmark accuracy is available**. Do not use QA answer-supporting `image_path` for frame selection.

## Completed real-RGB protocol

1. Extract native start/goal receptacle transitions and align eligible annotated RGB frames.
2. For every selected RGB observation, query the same MLX Qwen2.5-VL-7B-Instruct-4bit model twice with the candidate labels counterbalanced (`A/B`, then `B/A`); neither query uses the tokens START or GOAL. Convert agreement, single judgment, conflict, and abstention to structured evidence.
3. Fit **hierarchical consistency calibration on training scenes only**. Global judged reliability `0.794344`; judged response coverage on the calibration sample `0.787449`. State-specific consistent groups and pooled-shrunk single-judged group have separate train-derived reliabilities. Do not replace calibrated reliability with the VLM's unvalidated raw confidence.
4. Select recency decay and StateMem persistence on a *different* training-scene tuning split. Frozen reported values: `recency_decay=2.0`, `stay=0.8`, `frame_time_scale=2.0`. The source code's train-only selection objective uses GOAL and late-GOAL accuracy before overall accuracy and Brier for tie-breaking. Once selected, do not retune on validation.
5. Run frozen Latest, Majority, Recency-Calibrated, Bayes-NoDynamics, StateMem-Uniform, and StateMem-Calibrated on identical 148 validation transitions (963 state-evaluation steps across 67 episodes). Report accuracy, post-transition behavior, Brier where available, transition-recovery lag, and miss rate.

## Robustness replay (exploratory)

On the **same previously examined official validation set**, re-run all memory methods from saved VLM outputs. No new VLM inference, model change, parameter retuning, or confidence recalibration occurs. Scenarios: hide the first 1/2/3 post-transition GOAL observations; or retain 75%/50%/33% of observations after each initial START anchor, averaging 20 deterministic label-independent masks. Use paired **episode-cluster bootstrap** with 20,000 resamples; do not treat 963 evaluation steps as independent. This stress experiment was designed after viewing baseline validation results: results are **exploratory**, not fresh confirmatory validation.

## Downstream multi-object QA (exploratory)

Use the same saved VLM observations and frozen methods. Restrict to episodes with at least two tracked objects; start querying only after all tracked objects have an initial observation. At the union of selected evidence frames, evaluate per-object START/GOAL classification, exact entire tracked-world state, exact GOAL-object count, set-F1, and GOAL-count MAE. Completed evaluation contains **335 query times across 44 episodes per method**. Paired inference clusters by episode with 20,000 resamples. It is **offline structured QA**; it does **not** demonstrate embodied navigation or official benchmark task success.

## Interpretation safeguards

- Do not claim that StateMem beats LMEE/MEMORA/other agents: none has been directly re-run in an equivalent environment with the same perception and planning stack.
- Overall StateMem's full-RGB gain over Latest and Recency is small and its paired 95% confidence intervals cross zero. Recency has a slightly lower (better) Brier score; StateMem has lower *immediate* GOAL accuracy and a higher transition miss rate in the full stream.
- Sparse/delayed stress effects are mechanism-consistent and some confidence intervals exclude zero **within this exploratory replay**. They are not an independent benchmark test.
- The two-state START/GOAL representation simplifies full embodied world-state reasoning; online frame acquisition, perception calibration under distribution shift, additional object states, and second-environment independent validation remain open.
- Pilot LMEE captions describe observed scenes and uncertain labels; they are not calibrated state observations and do not establish real temporal world changes.

## Reproducibility and source references

Primary scripts are under `experiments/findingdory/`; summary results under `results/findingdory/`. Main frozen run: `findingdory_rgb_statemem_temporal_calfix.py`; post-hoc analyses: `statemem_temporal_stress_test.py` and `statemem_downstream_world_qa.py`. Dataset/agent repositories: [FindingDory Habitat](https://github.com/findingdory-benchmark/findingdory-habitat), [LMEE](https://github.com/wangsen99/LMEE). Record repository commits and downloaded dataset revisions before a final paper submission.
