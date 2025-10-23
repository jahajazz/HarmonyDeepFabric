## Harmony Fine-Tune Compatibility Report

### Can the existing pipeline ingest the raw transcripts?

No. Harmony training data in this repository is built around the three-message schema emitted by `harmony_writer.write_harmony_record`: a `system` preamble, a `user` content block, and at least one `assistant` turn that carries `analysis` and `final` strings. The diarised JSONL transcripts contain only per-speaker narration `{episode_id, start, end, speaker, text}` with no system prompt, no user/assistant role split, and no target response channels. The current tooling (for example `convert_to_harmony.py` and `validate_harmony.py`) refuse entries that lack those fields, so the transcripts cannot be used directly.

Additional gaps observed:
- Harmony converters assume user blocks of roughly 900-1400 tokens and assistant outputs under 400 tokens. The raw turns are much shorter and need grouping to stay within those budgets.
- Every assistant turn must provide `analysis` and `final` text; the transcripts only provide spoken content.
- Downstream reports expect provenance metadata (`chunk_id`, `token_count`, timestamps), none of which exist in the source JSONL.

### Expected Harmony schema

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {
      "role": "assistant",
      "analysis": "chain-of-thought text",
      "final": "user-facing answer"
    }
  ],
  "metadata": {
    "episode_id": "LOS-0002",
    "source_path": "...",
    "speaker_roles": {"Host": "user", "Guest": "assistant"},
    "token_estimates": {
      "user_total": 1240,
      "user_max": 640,
      "assistant_total": 360,
      "assistant_max": 190
    }
  }
}
```

### Mapping from transcript fields to Harmony

- `episode_id` -> `metadata.episode_id`
- `speaker` -> mapped to Harmony roles (`user`/`assistant`) when only two recurring speakers exist; otherwise rendered inline inside the user message
- `text` -> user or assistant content depending on the assigned role; multi-speaker segments flatten into a single user block
- `start` / `end` -> used for chronological ordering and merge heuristics; indirectly preserved through deterministic window indices
- `source path` -> carried through as `metadata.source_path`
- Derived fields -> system message preface, placeholder analysis text, token estimates, window counters

### Adapter and validator

1. `scripts/adapters/jsonl_to_harmony.py`  
   Converts diarised transcripts to Harmony records with options for speaker merging (`--merge_gaps_sec`), minimum character filtering, window sizes, and deterministic train/val splits. Metadata captures speaker-role mapping and simple token estimates so that long windows can be flagged quickly.

2. `scripts/validators/validate_harmony_inputs.py`  
   Confirms every record has the Harmony schema (`system` + `user` + assistant with both `analysis` and `final`), rejects whitespace-only text, and checks that user and assistant messages stay under configurable token budgets (defaults: user <= 2000 tokens, assistant <= 400 tokens). It prints record counts along with average and 95th percentile token lengths.

Both scripts rely only on the standard library plus optional `tiktoken` (used when available) and are under 200 lines each.

### Sample conversion (synthetic demo with `--max_turns 2`)

Input fragment (`tmp/sample.jsonl`):

```json
{"episode_id":"EP-001","start":0.0,"end":3.2,"speaker":"Host","text":"Welcome to the show."}
{"episode_id":"EP-001","start":3.2,"end":6.4,"speaker":"Guest","text":"Thank you for having me."}
{"episode_id":"EP-001","start":6.6,"end":10.0,"speaker":"Host","text":"Let's dive into the topic."}
{"episode_id":"EP-001","start":10.2,"end":14.5,"speaker":"Guest","text":"Absolutely, I think it's fascinating."}
{"episode_id":"EP-002","start":0.0,"end":2.1,"speaker":"Narrator","text":"Previously on our series."}
{"episode_id":"EP-002","start":2.1,"end":4.0,"speaker":"SpeakerA","text":"I had a question about the ritual."}
{"episode_id":"EP-002","start":4.0,"end":6.5,"speaker":"SpeakerB","text":"The ritual dates back centuries."}
{"episode_id":"EP-002","start":6.5,"end":8.2,"speaker":"SpeakerC","text":"And it involves several communities."}
```

Adapter output (`data/harmony_ready/train.jsonl` and `data/harmony_ready/val.jsonl`):

```json
{"messages":[
  {"role":"system","content":"Transcript mode: conversation. Episode: EP-001. Speakers observed: Host, Guest. Preserve the diarised ordering and attribution."},
  {"role":"user","content":"Host: Welcome to the show."},
  {"role":"assistant","analysis":"Direct transcript turn spoken by Guest. Reasoning not available.","final":"Thank you for having me."}
],"metadata":{"episode_id":"EP-001","source_path":"tmp/sample.jsonl","window_index":0,"windows_total":2,"speaker_roles":{"Host":"user","Guest":"assistant"},"speakers":["Host","Guest"],"mode":"conversation","strategy":"two_speaker","token_estimates":{"user_total":6,"user_max":6,"assistant_total":6,"assistant_max":6}}}
{"messages":[
  {"role":"system","content":"Transcript mode: conversation. Episode: EP-001. Speakers observed: Host, Guest. Preserve the diarised ordering and attribution."},
  {"role":"user","content":"Host: Let's dive into the topic."},
  {"role":"assistant","analysis":"Direct transcript turn spoken by Guest. Reasoning not available.","final":"Absolutely, I think it's fascinating."}
],"metadata":{"episode_id":"EP-001","source_path":"tmp/sample.jsonl","window_index":1,"windows_total":2,"speaker_roles":{"Host":"user","Guest":"assistant"},"speakers":["Host","Guest"],"mode":"conversation","strategy":"two_speaker","token_estimates":{"user_total":8,"user_max":8,"assistant_total":9,"assistant_max":9}}}
{"messages":[
  {"role":"system","content":"Transcript mode: conversation. Episode: EP-002. Speakers observed: Narrator, SpeakerA, SpeakerB, SpeakerC. Preserve the diarised ordering and attribution. This window contains three or more unique speakers, so the transcript is flattened into a single turn."},
  {"role":"user","content":"Narrator: Previously on our series.\nSpeakerA: I had a question about the ritual.\nSpeakerB: The ritual dates back centuries.\nSpeakerC: And it involves several communities."},
  {"role":"assistant","analysis":"Placeholder analysis for multi-speaker window. This window contains three or more unique speakers, so the transcript is flattened into a single turn.","final":"Narrator: Previously on our series.\nSpeakerA: I had a question about the ritual.\nSpeakerB: The ritual dates back centuries.\nSpeakerC: And it involves several communities."}
],"metadata":{"episode_id":"EP-002","source_path":"tmp/sample.jsonl","window_index":0,"windows_total":1,"speaker_roles":{},"speakers":["Narrator","SpeakerA","SpeakerB","SpeakerC"],"mode":"conversation","strategy":"flattened","token_estimates":{"user_total":42,"user_max":42,"assistant_total":42,"assistant_max":42}}}
```

### Validator results (synthetic run)

```
[validator] data/harmony_ready/train.jsonl -> 2 records | user avg=7.0 p95=8.0 | assistant-final avg=7.5 p95=9.0
[validator] data/harmony_ready/val.jsonl -> 1 records | user avg=42.0 p95=42.0 | assistant-final avg=42.0 p95=42.0
[validator] all inputs passed schema and token checks.
```

No schema violations were detected in the demo run. All messages stayed below the configured limits (user <= 2000 tokens, assistant <= 400 tokens), so no truncation was required.

### Practical considerations and recommended next steps

1. Collect or generate labelled targets (summaries, QA pairs, reflections) if the goal is to fine-tune on reasoning outputs. The adapter currently inserts deterministic placeholder analysis strings so that the Harmony schema stays valid.
2. Tune `--max_turns`, `--merge_gaps_sec`, and `--min_chars` per series. Longer windows keep more context, but they must remain under the desired token budget (~1400 tokens for user content).
3. After producing full datasets, run:
   - `python scripts/adapters/jsonl_to_harmony.py --input_glob "/mnt/data/*.jsonl" --out_dir data/harmony_ready --mode conversation --max_turns 20 --min_chars 20 --merge_gaps_sec 1.0 --train_split 0.95 --seed 7`
   - `python scripts/validators/validate_harmony_inputs.py --input data/harmony_ready/train.jsonl --input data/harmony_ready/val.jsonl`
4. If downstream analytics expect fields such as `chunk_id` or `token_count`, extend the adapter to pass through additional metadata from the source transcripts.
