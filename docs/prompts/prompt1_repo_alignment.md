# Prompt 1 — Repo Alignment Review (No Changes Yet)
_Updated: 2025-10-22 00:58 UTC_

## System
You are a principal software architect. Audit the repository for alignment with the Symbolic Commentary pipeline and Harmony output spec. Propose changes ONLY where the implementation is misaligned with the requirements below. Preserve existing behavior when already compliant. Return a concrete, minimal change plan.

## User
### Repository Goals & Rules
1. Local-first heuristics for harvesting Q→A from diarized transcripts; **ALLOWED answer speakers only**: `["Fr Stephen De Young","Jonathan Pageau"]`.
2. Question detection: prefer explicit `?` or interrogatives from non-target context; fallback to small-model **questionify** (gpt-4o-mini) ONLY when necessary. Enforce **≤25 words** and terminal `?`.
3. Answer assembly: merge contiguous target turns when **gap ≤ 1.0s** and **total chars ≤ 1200**; optional substring trim allowed but must remain a **contiguous substring**.
4. Fit gating: compute CrossEncoder fit score for (question, answer); keep if **≥ 0.55**; allow a gray zone (**0.45–0.65**) to trigger questionify then re-score.
5. Output: Harmony-STRICT training JSONL with **ONLY `messages`**; last role is **`assistant`**. Sidecar JSONL (1:1) holds spans/provenance/gates; **never** included in training.
6. Validation: strict validator must return “OK” for train/val files; CI fails otherwise. Idempotent outputs for same inputs & seed.
7. Sample inputs to cover in tests:
   - `/mnt/data/SW - #089 - 2020-01-15 - The Road From Equality to Inversion.jsonl`
   - `/mnt/data/LOS - #007 - 2020-11-13 - His Ministers Flaming Fire.jsonl`

### Deliverables
- A **short diff-oriented change plan** listing: files to edit, functions to touch, minimal code deltas, and tests to add or adjust.
- **Do NOT implement yet**; propose smallest safe steps.
