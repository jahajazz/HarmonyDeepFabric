# Prompt 5 — Harmony Export & Sidecar Alignment
_Updated: 2025-10-22 00:58 UTC_

## System
You are a principal software architect. Verify exporters conform exactly to Harmony-STRICT and sidecar schemas. Fix only misalignments.

## User
### Tasks
- **Training JSONL**: ONLY top-level key `messages`; 2–3 entries (system optional), last role = **assistant**.
- **Sidecar JSONL**: same line count as training; include `pair_id`, `episode_id`, `source_file`, **spans**, **gates**, `fit_score_ce`.
- Add strict validator to CI; pipeline must **fail** on any error.
- Add integration tests using the two sample files; assert validator prints **“OK”** and line counts match.

### Output
- Return concise diffs and tests only.
