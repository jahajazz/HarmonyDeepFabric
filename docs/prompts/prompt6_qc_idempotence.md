# Prompt 6 — QC Audit & Idempotence
_Updated: 2025-10-22 00:58 UTC_

## System
You are a principal software architect. Add/verify QC auditing and idempotence with minimal changes.

## User
### Tasks
- Implement a **3%** random sample QC auditor (fixed seed) that checks: groundedness, format, speaker allow-list, duplication.
- Report failure items to **JSON/CSV** with episode/timecodes.
- Ensure re-running with the same inputs & seed produces **identical outputs** (except timestamps). Write `split_manifest.json` and compare on re-runs.
- Add tests for deterministic output under fixed seed and variation under changed seed.

### Constraints
- Provide minimal diffs and tests only.
