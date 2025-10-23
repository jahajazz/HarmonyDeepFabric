# Phase 3 Prompt — Validation & QC
- Wire strict validator in CI; fail build on error.
- Add 3% QC audit (fixed seed) for groundedness, format, allow-list, duplication.
- Write JSON/CSV reports; enforce **≥ 95%** pass-rate target.
- Determinism: file hashes and `split_manifest.json` must be identical for same seed.
- Ensure no pair_id/dedup_key overlap across splits.
