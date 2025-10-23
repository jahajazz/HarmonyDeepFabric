# Controller Prompt (Prompt 0 — keep chat minimal)
Read and follow these docs instead of duplicating specs in chat.

READ FIRST:
- `docs/spec/harmony_requirements.md`
- `docs/spec/phase_acceptance.md`

HARD CONSTRAINTS (do not change unless tests fail):
- Allowed answer speakers: ["Fr Stephen De Young","Jonathan Pageau"].
- Answers verbatim or contiguous substring; merge gap ≤ 1.0s; max_answer_chars=1200.
- Harmony training JSONL: messages-only; last role = assistant. Sidecar 1:1 aligned.
- Do NOT change train/val splitting unless determinism or leakage checks fail.
- Heuristics first; use gpt-4o-mini only for questionify.

PHASE GATES:
- P1 exporters/validator OK → P2 heuristics → P3 QC+determinism → P4 tests.

PER-STEP BUDGET:
- ≤ 2 edits, ≤ 1000 output tokens. If exceeded: STOP and summarize blockers.

OUTPUT EACH STEP:
- Plan (≤10 lines): files, functions, expected diffs
- Edits: paths only
- Commands run
- Result: OK / fail + reason
