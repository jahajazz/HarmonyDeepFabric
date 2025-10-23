# Prompt 2 — Enforce Allowed Speakers & Merging (Minimal Diffs)
_Updated: 2025-10-22 00:58 UTC_

## System
You are a principal software architect. Implement ONLY the smallest changes required to enforce the allowed-speaker rule and contiguous merging constraints without breaking existing behavior.

## User
### Implement with minimal diffs
- Allowed answer speakers: `["Fr Stephen De Young","Jonathan Pageau"]`. **Drop pairs** otherwise.
- Merge adjacent turns by the **same allowed speaker** if **gap ≤ 1.0s**; preserve timestamps; cap total answer length **≤ 1200 chars**.
- If trimming is applied, ensure the trimmed text is a **contiguous substring** of the original answer.
- Add unit tests using the two sample files to cover:
  - (a) allowed vs disallowed speaker answers
  - (b) merge at **0.8s** gap
  - (c) **no-merge** at **1.2s** gap
  - (d) substring-trim guard

### Constraints
- Do not alter unrelated modules.
- Provide concise diffs and updated tests only.
