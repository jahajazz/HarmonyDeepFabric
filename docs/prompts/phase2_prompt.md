# Phase 2 Prompt — Local-first Heuristics
- Enforce allowed speakers for answers (Fr Stephen De Young, Jonathan Pageau).
- Merge adjacent same-speaker turns if **gap ≤ 1.0s**; cap **≤ 1200 chars**.
- Trim guard: result must be a **contiguous substring**.
- Questionify only if no clean context question; **≤25 words**, ends `?`, JSON { "question": "<...>" }.
- CrossEncoder score: keep if **≥ 0.55**; gray zone 0.45–0.65: single questionify attempt then re-score.
- Update sidecar flags.
- Add unit tests for these gates.
