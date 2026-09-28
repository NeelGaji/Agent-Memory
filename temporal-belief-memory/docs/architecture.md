# StateMem: architecture and scope

**Working project name:** StateMem (final public name to be checked for collisions).  
**Research question:** Can an embodied agent maintain a more reliable estimate of the *current* world state by treating observations as uncertain historical evidence rather than immediately committing each observation as fact?

![Architecture](../figures/fig01_architecture.png)

## Design

The proposed system separates **Episodic Evidence Memory (EEM)** from **World Belief Memory (WBM)**. EEM stores time-stamped observations and provenance; WBM holds a probabilistic estimate of the present state of each tracked entity. The **Temporal Belief Reconciler** propagates beliefs through elapsed time and incorporates observations with reliability estimated on separate calibration data. A downstream query/router or planner can use WBM while retrieving supporting historical evidence from EEM.

```text
single-agent RGB observations
      | perceptual model (current experiments: counterbalanced local Qwen VLM)
      v
structured observations with frame/time and provenance
      v
Episodic Evidence Memory ----> Temporal Belief Reconciler ----> World Belief Memory
       |                                  ^                            |
       |                  calibrated observation reliability           |
       +------------------------ evidence retrieval ------------------+
                                                                    |
                                                     query/QA/navigation interface
```

The *full* diagram depicts the architectural target. The completed FindingDory experiments implement the **perception -> evidence -> reconciler -> belief** path and evaluate its state estimates offline. A lightweight multi-object QA probe reads those estimates. **Online query routing, a Habitat navigation controller driven by StateMem, and LMEE official benchmark integration are not yet implemented.**

## Belief update

For tracked entity `e` with candidate states `s`, the world belief is:

`B_t^e(s) = P(S_t^e=s | z_1, ..., z_t)`.

A conceptual filtering step first predicts state persistence across the observed time gap and then updates with the likelihood of the newest observation:

```text
predicted_B_t(s) = sum_{s_prev} P(s | s_prev, delta_t) B_(t-1)(s_prev)
B_t(s) proportional_to P(z_t | s, calibrated_reliability_t) predicted_B_t(s)
```

In the *completed FindingDory RGB evaluation*, candidate states are **START** and **GOAL** receptacles, not arbitrary full 3-D semantic states. The symmetric two-state persistence parameter is `stay=0.8`; a train-derived frame-gap scale of `2.0` converts elapsed frame gaps into effective transition time. Recency baseline decay is `2.0`. A single train-only calibrator supplies evidence reliability; the reported global judged reliability is `0.794344` (not a universal confidence for every observation). Consistent and single-judged counterbalanced responses receive different train-derived reliability treatment. Contradictory or double-abstaining responses do not force a state update. See `experimental_protocol.md` for exact scope.

## Why two memory stores?

An observation is a statement about *what was perceived at a given time*, not an unconditional fact about the present. A newly reported state may be wrong or obsolete; repeated images may be redundant, and labels may differ between views. EEM preserves what was actually observed and when, including contradictions. WBM captures a revisable present-state hypothesis while retaining links to evidence. This is a modular mechanism that could sit alongside retrieval-based agents; it is **not** a claim that probabilistic filtering, episodic memory, or belief states are novel individually.

## LMEE integration status

The local LMEE adapter has indexed **58 tasks, 145 questions, and 32 scenes** in its mini subset (metadata only). One eight-frame chronological RGB pilot was captioned on an Apple Silicon Mac. The captions alternated between `picture`, `whiteboard`, `screen`, and `map` for a visually similar wall-mounted graphic. This is a useful **entity-association smoke test**, not evidence of a real object-state transition or an official LMEE result. LMEE-specific confidence is **not** calibrated by reusing the FindingDory calibrator.

## Intended downstream interface (not yet benchmarked online)

The planned API exposes `(entity_id, current_state_distribution, evidence_refs, last_update_frame)` to the query layer. For navigation, a query identifies candidate receptacles/locations; WBM supplies state probabilities; the same low-level policy/planner can act on that state in controlled method comparisons. An end-to-end success claim requires executing this interface inside the official simulator.
