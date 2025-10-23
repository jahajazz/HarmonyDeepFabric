# Harmony Requirements — Symbolic Commentary Pipeline
_Updated: 2025-10-22 01:02 UTC_

## Allowed Speakers (Answers)
- **Fr Stephen De Young**
- **Jonathan Pageau**

Answers must be **verbatim** or a **contiguous substring** of the allowed speaker's utterances.

## Answer Assembly
- Merge adjacent turns by the **same allowed speaker** if `gap_sec ≤ 1.0`.
- Do not cross episode boundaries.
- Maximum answer length (pre-trim): **1200 characters**.
- Optional trim permitted only if the result is a **contiguous substring** of the original merged answer.

## Question Sourcing
- Prefer explicit `?` or interrogatives in **nearby non-target context** preceding the answer.
- If none: call **gpt-4o-mini** with a “Questionify” prompt.
  - Must return JSON: `{"question": "<text>"}`.
  - Enforce **≤ 25 words** and **ends with `?`**.
  - Invalid/missing ⇒ **drop** the pair.

## Pair Fit (Semantic Gate)
- Use a small **CrossEncoder** (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`).  
- Compute `fit_score_ce ∈ [0,1]` and keep if **≥ 0.55**.
- Gray zone `0.45–0.65`: allow a single questionify attempt then re-score; if still < 0.55 ⇒ **drop**.

## Output (Harmony-STRICT)
- **Training JSONL**: top-level **`messages` only**; 2–3 entries (system optional), **last role = `assistant`**.
- **Sidecar JSONL**: 1:1 line alignment with training; include `pair_id`, `episode_id`, `source_file`, **spans**, **gates**, `fit_score_ce`.
- Never include sidecar fields in training JSONL.

## Split & Idempotence
- Maintain current **train/val** split; prefer **episode-level isolation**.
- Determinism with fixed `--seed`: same inputs ⇒ identical outputs (except timestamps).
- Provide `reports/split_manifest.json` for reproducible tracking.

## Validation & QC
- **Strict validator** (in CI) must print **“OK”** for train and val.
- **QC audit**: 3% random sample (fixed seed); checks: groundedness, format, speaker allow-list, duplication/near-duplication.
- QC pass-rate target **≥ 95%**; failures logged to JSON/CSV.

## Model Use Policy
- Local heuristics first.
- **gpt-4o-mini** only for **questionify** (and optional QC copy where configured).
- No escalation to larger models in the pipeline by default.
