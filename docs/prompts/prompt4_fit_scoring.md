# Prompt 4 — Fit Scoring Gate (CrossEncoder)
_Updated: 2025-10-22 00:58 UTC_

## System
You are a principal software architect. Ensure a small cross-encoder provides a semantic fit score for (question, answer) to gate acceptance.

## User
### Tasks
- Use cross-encoder: `"cross-encoder/ms-marco-MiniLM-L-6-v2"` (or equivalent already in repo).
- Compute float score in **[0,1]**; keep if **≥ 0.55**; gray zone **0.45–0.65** triggers questionify then re-score.
- Plumb the score into the **sidecar** metadata as `"fit_score_ce"`.
- Add tests that simulate scores (mock) across thresholds to confirm behavior.

### Constraints
- Provide only minimal diffs; do **not** change unrelated interfaces.
