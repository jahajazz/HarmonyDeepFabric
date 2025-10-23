# Phase 1 Prompt — Export & Harmony Schema
- Make training JSONL **messages-only** (system optional); **last role = assistant**.
- Write sidecar JSONL 1:1 aligned; include `pair_id`, `episode_id`, `source_file`, spans, gates, `fit_score_ce`.
- Add/ensure strict validator and run it on train/val; fail on any error.
- Do **not** modify train/val splitting.
- Output: concise diffs and test stubs only.
