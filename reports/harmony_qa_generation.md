## Harmony Q/A Generation Workflow

### Overview

The script `scripts/generators/harmony_qa_from_transcripts.py` converts diarised podcast transcripts into Harmony-formatted question/answer pairs ready for OSS GPT fine-tuning. It preserves the original theological tone and restricts assistant answers to an explicit speaker whitelist (for example, Fr Stephen De Young and Jonathan Pageau).

Key capabilities:
- Loads one or more JSONL transcripts with `{episode_id,start,end,speaker,text}` records.
- Builds context windows around lines spoken by approved answer speakers while retaining nearby dialogue for grounding.
- Calls an OpenAI model (default `gpt-4o`) to propose faithful Q/A pairs that quote the whitelist speakers verbatim or with minimal trimming.
- Validates that every answer appears in the source text and emits Harmony messages with system/user/assistant channels plus rich metadata.
- Optionally rewrites preceding non-question host turns into concise questions via a lightweight model (`gpt-4o-mini` by default).
- Auto-trims long, filler-heavy answers to the shortest faithful substring while preserving provenance indices.
- Optionally double-checks that the answer speaker is on the allow-list and that no other voices leak into the final response.
- Samples 1–5% of records for a strict GPT-4o QC audit, logging any failures before you kick off fine-tuning.
- Runs a second lightweight judge to ensure each question/answer pair is supported by the cited transcript spans before inclusion.
- Optionally writes an audit markdown file summarising each generated sample for quick spot-checking.

### Prerequisites

- Ensure `.env` (or your shell) exposes `OPENAI_API_KEY`.
- Transcripts should live under `deepfabric/data/transcripts/*.jsonl` (or any directory passed via `--input_glob`).
- Install project dependencies (the repo already pins `openai>=1.107.2`).

### Recommended command

```bash
python scripts/generators/harmony_qa_from_transcripts.py \
  --input_glob "deepfabric/data/transcripts/*.jsonl" \
  --out_dir data/harmony_ready \
  --answer_speakers "Fr Stephen De Young,Jonathan Pageau" \
  --model gpt-4o \
  --context_turns 6 \
  --questions_per_window 1 \
  --train_split 0.95 \
  --seed 7 \
  --audit_path reports/harmony_qa_audit.md
```

This writes `data/harmony_ready/train.jsonl`, `data/harmony_ready/val.jsonl`, and an audit table at `reports/harmony_qa_audit.md`. Add `--dry_run_windows 5` the first time you run it to limit API usage.

### Important CLI options

- `--answer_speakers`: Comma-separated whitelist of speakers whose words may appear in answers. Adjust the list without editing code.
- `--context_turns` and `--max_context_tokens`: Control how much surrounding conversation the model sees. Larger values retain more context but consume more tokens.
- `--questions_per_window`: Upper bound on Q/A pairs requested per context window (each candidate still faces post-validation).
- `--model`: Pick the highest-fidelity OpenAI model available to your account (e.g., `gpt-4o`, newer releases when they arrive).
- `--questionify_model`, `--questionify_temperature`, `--questionify_max_output_tokens`, `--disable_questionify`: Configure or disable the light-weight rewrite that turns non-interrogative host turns into <=25-word questions.
- `--answer_trim_model`, `--answer_trim_temperature`, `--answer_trim_max_output_tokens`, `--answer_trim_threshold`, `--disable_answer_trim`: Control the extractive trimming pass that shortens long answers while keeping them contiguous substrings.
- `--speaker_check_model`, `--speaker_check_temperature`, `--speaker_check_max_output_tokens`, `--disable_speaker_check`: Optional allow-list guard that asks a tiny model to confirm the final answer stays within the approved speaker voices.
- `--qc_model`, `--qc_temperature`, `--qc_max_output_tokens`, `--qc_sample_rate`, `--qc_min_samples`, `--qc_pass_threshold`, `--qc_seed`, `--qc_output_path`, `--disable_qc_audit`: Manage the random-sample dataset auditor and its pass/fail thresholds.
- `--pairfit_model`, `--pairfit_temperature`, `--pairfit_max_output_tokens`, `--disable_pairfit`: Control the semantic judge that verifies each Q→A pair remains grounded in the cited transcript spans.
- `--audit_path`: Enables markdown output summarising episode, window id, question, answer speaker, and the first citation timestamp.
- `--dry_run_windows`: Limits the number of windows processed, useful while tuning prompts.

### Output structure

Each Harmony JSONL record includes:
- `messages[0]`: system instruction reminding the fine-tuned model to maintain tone and respect the speaker whitelist.
- `messages[1]`: user message containing the generated question plus the transcript excerpt used as context.
- `messages[2]`: assistant analysis (reasoning with citations) and final answer (verbatim quote).
- `metadata`: episode id, window id, answer speaker, cited spans, and the original segments so you can trace every token back to the source transcript.
- `metadata.question_origin`: whether the user question came directly from context (`context`), via the small-model rewrite (`questionify`), or from the generation model fallback (`model`). When available, `metadata.question_context` and `metadata.question_context_speaker` capture the pre-answer speaker turn that seeded the rewrite.
- `metadata.answer_trim`: start/end character indices (plus lengths and thresholds) for any answer trimming that was applied.
- `metadata.speaker_check`: result of the allow-list compliance check when enabled (e.g., `{"allowed": true, "reason": null}`).
- `metadata.qc_audit`: pass/fail verdict from the random-sample QC judge, including issue codes and a short comment when available.
- `metadata.pair_validation`: status and attempt logs from the semantic judge (`accepted`, `rejected`, `skipped`, etc.), plus any suggested rewrites that were applied.

### Post-generation validation

1. Run the repository validator to confirm schema compliance and token budgets:
   ```bash
   python scripts/validators/validate_harmony_inputs.py \
     --input data/harmony_ready/train.jsonl \
     --input data/harmony_ready/val.jsonl \
     --max_user_tokens 1800 \
     --max_final_tokens 450
   ```
2. Spot-check samples from `reports/harmony_qa_audit.md` inside ChatGPT or another review tool to ensure the answers faithfully reflect the cited speaker's words.
3. Once satisfied, proceed with Harmony fine-tuning using the generated JSONL shards.

### Extending the workflow

- Adjust `--answer_speakers` or prepare small wrapper configs to manage different speaker groups.
- Increase `--questions_per_window` or reduce `--context_turns` to trade dataset volume against citation fidelity.
- Add extra heuristics (topic detection, manual episode filtering) upstream before invoking the generator if you need tighter control.
