# Prompt 3 — Question Detection & Questionify Gate
_Updated: 2025-10-22 00:58 UTC_

## System
You are a principal software architect. Align question identification and the small-model questionify fallback with requirements, minimizing code churn.

## User
### Requirements
- Prefer explicit `?` or interrogative openers in nearby **non-target context** to form the question.
- If not present, call **gpt-4o-mini** with the *Questionify* prompt and enforce:
  - **≤ 25 words**
  - Ends with `?`
  - JSON: `{"question": "<...>"}`; **drop** if empty or invalid
- Recompute CrossEncoder fit score after questionify; keep only if **≥ 0.55**.

### Tests
- A context line without `?` becomes a 1-sentence question via questionify.
- Malformed/empty JSON from the model results in **dropping** the pair.

### Output
- Return diffs and tests only; do **not** refactor working code that already meets these rules.
