# Phase Acceptance Criteria
_Updated: 2025-10-22 01:02 UTC_

## Phase 1 — Export & Schema
- ✅ Training JSONL is **messages-only**; last message role is **assistant**.
- ✅ Sidecar JSONL exists, **1:1** aligned; contains `pair_id`, `episode_id`, `source_file`, spans, gates, `fit_score_ce`.
- ✅ Validator prints **“OK”** on both train/val.
- ✅ No change to train/val split.

## Phase 2 — Local-first Heuristics
- ✅ Allow-list enforced: only **Fr Stephen De Young** or **Jonathan Pageau** answers.
- ✅ Merge same-speaker turns with **gap ≤ 1.0s**; cap answer length **≤ 1200 chars**.
- ✅ Trim guard: trimmed answer is a **contiguous substring**.
- ✅ Questionify only if needed; **≤ 25 words**, ends `?`, valid JSON.
- ✅ Pair-fit populated; **keep if ≥ 0.50** (gray zone logic applied).
- ✅ Sidecar flags populated: `speaker_check_passed`, `pairfit_passed`, `fit_score_ce`, `questionify_used`, `answer_trimmed`.

## Phase 3 — Validation & QC
- ✅ Strict validator integrated into CI; fails build on any error.
- ✅ QC audit (3%) with fixed seed; **pass-rate ≥ 95%**; JSON/CSV reports.
- ✅ Determinism: same seed ⇒ identical file hashes; `split_manifest.json` stable.
- ✅ No leakage: no overlap of `pair_id`/`dedup_key` across train/val.

## Phase 4 — Tests & Hardening (no feature changes)
- ✅ Unit tests for: allow-list, merge threshold, substring-trim guard, questionify contract, pair-fit thresholds.
- ✅ Split tests: determinism and no leakage.
- ✅ QC sampler determinism with fixed seed.
