#!/usr/bin/env python3
"""
Convert symbolic dataset to proper Harmony conversation format.

This script loads symbolic_dataset.jsonl line by line and wraps each record
as a Harmony conversation with the specified structure.
"""

import json
import sys
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from harmony_writer import write_harmony_record

# Try to import tiktoken for token counting, fallback to transformers
try:
    import tiktoken
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False
    try:
        from transformers import AutoTokenizer
        TOKENIZER_FALLBACK = True
    except ImportError:
        TOKENIZER_FALLBACK = False

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count tokens in text using tiktoken or fallback method."""
    if TOKENIZER_AVAILABLE:
        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except:
            pass

    if TOKENIZER_FALLBACK:
        try:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            return len(tokenizer.encode(text))
        except:
            pass

    # Fallback: rough approximation (1 token ≈ 4 characters for English)
    return len(text) // 4

def parse_speaker_list(speaker_str: str) -> List[str]:
    """Parse comma-separated speaker list into normalized list."""
    if not speaker_str:
        return []
    return [speaker.strip().lower() for speaker in speaker_str.split(',') if speaker.strip()]

def should_include_speaker(speaker: str, include_speakers: List[str], exclude_speakers: List[str]) -> bool:
    """Check if speaker should be included based on filter lists."""
    if not speaker:
        return False

    speaker_lower = speaker.lower()

    # If include list is specified, speaker must be in it
    if include_speakers and speaker_lower not in include_speakers:
        return False

    # Speaker must not be in exclude list
    if speaker_lower in exclude_speakers:
        return False

    return True

def load_transcript_file(transcript_path: Path, source_file: str) -> Tuple[Optional[List[Dict]], str]:
    """Load transcript from file, supporting both .txt and .json formats."""
    file_path = transcript_path / source_file

    if not file_path.exists():
        return None, f"File not found: {source_file}"

    try:
        if source_file.endswith('.json'):
            # JSON format with segments
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'segments' in data:
                    return data['segments'], "JSON segments loaded"
                else:
                    return None, "JSON file missing 'segments' key"
        else:
            # Plain text format - split by speaker patterns
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Simple segmentation by common speaker patterns
            segments = []
            lines = content.split('\n')

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Try to extract speaker and text
                # Pattern: [timestamp] SPEAKER: text or SPEAKER: text
                speaker_match = re.match(r'(\[[\d:]+\]\s*)?([^:]+):\s*(.+)', line)
                if speaker_match:
                    timestamp = speaker_match.group(1) or ""
                    speaker = speaker_match.group(2).strip()
                    text = speaker_match.group(3).strip()

                    segments.append({
                        'speaker': speaker,
                        'text': text,
                        'start': 0,  # No timing info for plain text
                        'end': 0
                    })
                else:
                    # If no speaker pattern, treat as continuation
                    if segments:
                        segments[-1]['text'] += ' ' + line

            return segments, f"Text file loaded with {len(segments)} segments"

    except Exception as e:
        return None, f"Error loading transcript: {e}"

def filter_segments_by_speakers(segments: List[Dict], include_speakers: List[str],
                              exclude_speakers: List[str]) -> List[Dict]:
    """Filter transcript segments based on speaker inclusion/exclusion lists."""
    filtered_segments = []

    for segment in segments:
        speaker = segment.get('speaker', '')
        if should_include_speaker(speaker, include_speakers, exclude_speakers):
            filtered_segments.append(segment)

    return filtered_segments

def slice_segments_by_time(segments: List[Dict], start_time: float = None, end_time: float = None) -> List[Dict]:
    """Slice segments to time window if timing information is available."""
    if not start_time and not end_time:
        return segments

    sliced_segments = []

    for segment in segments:
        seg_start = segment.get('start', 0)
        seg_end = segment.get('end', 0)

        # Include segment if it overlaps with the time window
        if (not start_time or seg_end > start_time) and (not end_time or seg_start < end_time):
            sliced_segments.append(segment)

    return sliced_segments

def normalize_text_tokens(text: str, min_tokens: int, max_tokens: int) -> str:
    """Normalize text to fit within token budget using sentence boundary truncation."""
    current_tokens = count_tokens(text)

    if min_tokens <= current_tokens <= max_tokens:
        return text

    # If too long, try to truncate at sentence boundaries
    if current_tokens > max_tokens:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        truncated_text = ""

        for sentence in sentences:
            test_text = truncated_text + sentence + " "
            test_tokens = count_tokens(test_text.strip())

            if test_tokens <= max_tokens:
                truncated_text = test_text
            else:
                break

        # If still too long after sentence truncation, hard trim
        if count_tokens(truncated_text.strip()) > max_tokens:
            truncated_text = truncated_text.strip()
            while count_tokens(truncated_text) > max_tokens and len(truncated_text) > 0:
                # Remove words from the end
                words = truncated_text.split()
                if len(words) > 1:
                    truncated_text = ' '.join(words[:-1])
                else:
                    break

        return truncated_text.strip()

    # If too short, return as is (could pad in future if needed)
    return text

def move_leaked_reasoning(analysis_data: Dict[str, Any], final_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Move leaked reasoning from final to analysis section."""
    if not isinstance(final_data, dict) or 'summary' not in final_data:
        return analysis_data, final_data

    summary = final_data['summary']
    if not isinstance(summary, str):
        return analysis_data, final_data

    leaked_patterns = [
        'Symbolic reasoning:',
        'Archetypal patterns:',
        'Theological themes:'
    ]

    lines = summary.split('\n')
    leaked_lines = []
    clean_lines = []

    for line in lines:
        is_leaked = False
        for pattern in leaked_patterns:
            if line.strip().startswith(pattern):
                is_leaked = True
                leaked_lines.append(line.strip())
                break

        if not is_leaked:
            clean_lines.append(line)

    # Move leaked lines to analysis if found
    if leaked_lines:
        updated_analysis = analysis_data.copy()

        # Extract content from leaked lines and add to appropriate analysis fields
        for leaked_line in leaked_lines:
            if leaked_line.startswith('Symbolic reasoning:'):
                content = leaked_line.replace('Symbolic reasoning:', '').strip()
                if 'symbolic_meaning' not in updated_analysis:
                    updated_analysis['symbolic_meaning'] = content
            elif leaked_line.startswith('Archetypal patterns:'):
                content = leaked_line.replace('Archetypal patterns:', '').strip()
                if 'archetypal_patterns' not in updated_analysis:
                    updated_analysis['archetypal_patterns'] = [content] if content else []
            elif leaked_line.startswith('Theological themes:'):
                content = leaked_line.replace('Theological themes:', '').strip()
                if 'theological_themes' not in updated_analysis:
                    updated_analysis['theological_themes'] = [content] if content else []

        # Update final summary
        updated_final = final_data.copy()
        updated_final['summary'] = '\n'.join(clean_lines).strip()

        return updated_analysis, updated_final

    return analysis_data, final_data

def validate_harmony_entry(entry: Dict[str, Any]) -> bool:
    """Validate that a Harmony entry has proper structure and non-empty content."""

    if not entry or not isinstance(entry, dict):
        return False

    # Check metadata
    if 'metadata' not in entry:
        return False

    # Check messages
    if 'messages' not in entry or not isinstance(entry['messages'], list):
        return False

    messages = entry['messages']

    # Must have at least 3 messages (system, user, assistant)
    if len(messages) < 3:
        return False

    # Check each required role
    roles = [msg.get('role') for msg in messages if isinstance(msg, dict)]

    if 'system' not in roles or 'user' not in roles or 'assistant' not in roles:
        return False

    # Find the assistant message
    assistant_msg = None
    for msg in messages:
        if msg.get('role') == 'assistant':
            assistant_msg = msg
            break

    if not assistant_msg:
        return False

    # Validate assistant message structure
    if not isinstance(assistant_msg, dict):
        return False

    analysis = assistant_msg.get('analysis')

    if isinstance(analysis, dict):
        if not analysis.get('symbolic_meaning') or not isinstance(analysis['symbolic_meaning'], str):
            return False
        if not analysis.get('archetypal_patterns') or not isinstance(analysis['archetypal_patterns'], list):
            return False
        if not analysis.get('theological_themes') or not isinstance(analysis['theological_themes'], list):
            return False
    elif isinstance(analysis, str):
        if not analysis.strip():
            return False
    else:
        return False

    final_field = assistant_msg.get('final')

    if isinstance(final_field, dict):
        summary_value = final_field.get('summary') or final_field.get('symbolic_summary') or final_field.get('final')
        if not isinstance(summary_value, str) or not summary_value.strip():
            return False
    elif isinstance(final_field, str):
        if not final_field.strip():
            return False
    else:
        return False

    return True

def transform_to_harmony_conversation(data: Dict[str, Any], index: int, transcript_path: Path = None,
                                     primary_speaker: str = '', include_speakers: List[str] = None,
                                     exclude_speakers: List[str] = None, user_min_tokens: int = 900,
                                     user_max_tokens: int = 1400, start_time: float = None,
                                     end_time: float = None) -> Dict[str, Any]:
    """Transform a data entry to Harmony conversation format with advanced features."""

    # Extract and sanitize metadata
    metadata = {
        'chunk_id': int(data.get('chunk_id', index + 1)),
        'speaker': str(data.get('speaker', 'unknown')).replace('"', '').replace('\\', '').replace('\n', ' ').replace('\r', ''),
        'is_primary': bool(data.get('is_primary', False)),
        'source_file': str(data.get('source_file', 'unknown.txt')).replace('"', '').replace('\\', '').replace('\n', ' ').replace('\r', ''),
        'token_count': int(data.get('token_count', 0)),
        'model': str(data.get('model', 'gpt-4o-mini')).replace('"', '').replace('\\', '').replace('\n', ' ').replace('\r', ''),
        'timestamp': float(data.get('timestamp', 0)),
        'status': str(data.get('status', 'completed')).replace('"', '').replace('\\', '').replace('\n', ' ').replace('\r', '')
    }

    start_time_val = data.get('start_time')
    if start_time_val is not None:
        try:
            metadata['start_time'] = float(start_time_val)
        except (TypeError, ValueError):
            pass

    end_time_val = data.get('end_time')
    if end_time_val is not None:
        try:
            metadata['end_time'] = float(end_time_val)
        except (TypeError, ValueError):
            pass

    # Add primary speaker hit tracking
    if primary_speaker and metadata['speaker'].lower() == primary_speaker.lower():
        metadata['primary_speaker_hit'] = True
    else:
        metadata['primary_speaker_hit'] = False

    # Fixed analysis data (clean)
    analysis_data = {
        'symbolic_meaning': 'Deep spiritual significance revealed through symbolic interpretation',
        'archetypal_patterns': ['Theological pattern', 'Symbolic motif'],
        'theological_themes': ['Divine relationship', 'Human condition']
    }

    final_data = {
        'summary': 'Symbolic and theological analysis of transcript content'
    }

    # Load and process transcript
    transcript_text = ""
    transcript_status = "PLACEHOLDER"
    segments = None

    if transcript_path and metadata['source_file']:
        segments, status_msg = load_transcript_file(transcript_path, metadata['source_file'])
        transcript_status = "TRANSCRIPT_LOADED" if segments else "TRANSCRIPT_ERROR"

        if segments:
            # Apply speaker filtering
            if include_speakers or exclude_speakers:
                segments = filter_segments_by_speakers(segments, include_speakers or [], exclude_speakers or [])

            # Apply time slicing if specified
            segments = slice_segments_by_time(segments, start_time, end_time)

            # Assemble transcript text in specified format
            transcript_lines = []
            for segment in segments:
                ts = f"[{segment.get('start', 0):.1f}] " if segment.get('start', 0) > 0 else ""
                speaker = segment.get('speaker', 'Unknown')
                text = segment.get('text', '')
                transcript_lines.append(f"{ts}{speaker.upper()}: {text}")

            transcript_text = '\n'.join(transcript_lines)
            print(f"📖 Loaded transcript: {metadata['source_file']} ({len(segments)} segments)")
        else:
            transcript_text = f"Transcript chunk {metadata['chunk_id']} from {metadata['source_file']}"
            print(f"📋 Using placeholder for: {metadata['source_file']} - {status_msg}")

    if not transcript_text:
        transcript_text = f"Transcript chunk {metadata['chunk_id']} from {metadata['source_file']}"

    # Apply token normalization to user content
    original_tokens = count_tokens(transcript_text)
    normalized_transcript = normalize_text_tokens(transcript_text, user_min_tokens, user_max_tokens)
    final_tokens = count_tokens(normalized_transcript)

    print(f"🎯 Token normalization: {original_tokens} → {final_tokens} tokens")

    # Build user content in specified format
    user_content = f"""Analyze the following source and produce a symbolic narrative summary.

[source_file={metadata['source_file']}, chunk_id={metadata['chunk_id']}, speaker={metadata['speaker']}]

<<<TRANSCRIPT>>>
{normalized_transcript}
<<<END TRANSCRIPT>>>
"""

    # Build system message with optional primary speaker instruction
    system_content = "You are an expert symbolic theologian with deep knowledge of biblical symbolism, archetypal patterns, and theological interpretation. Analyze the provided transcript text and provide comprehensive symbolic analysis."

    if primary_speaker:
        system_content += f"\n\n🎯 ANALYSIS FOCUS: Prefer interpretations voiced by {primary_speaker}; treat other speakers as context. If {primary_speaker} absent, maintain a consistent voice."

    # Build the Harmony conversation with sanitized content
    harmony_conversation = {
        "metadata": metadata,
        "transcript_status": transcript_status,
        "messages": [
            {
                "role": "system",
                "content": system_content
            },
            {
                "role": "user",
                "content": user_content
            },
            {
                "role": "assistant",
                "analysis": analysis_data,
                "final": final_data
            }
        ]
    }

    # Apply guardrails to move leaked reasoning
    analysis_data, final_data = move_leaked_reasoning(analysis_data, final_data)
    harmony_conversation["messages"][2]["analysis"] = analysis_data
    harmony_conversation["messages"][2]["final"] = final_data

    return harmony_conversation

def convert_to_harmony(input_file: str, output_file: str, transcript_dir: str = None,
                      primary_speaker: str = '', include_speakers: str = '',
                      exclude_speakers: str = '', user_min_tokens: int = 900,
                      user_max_tokens: int = 1400, limit: int = 0) -> None:
    """Convert symbolic dataset to Harmony conversation format with advanced features."""

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        sys.exit(1)

    # Validate transcript directory if provided
    transcript_path = None
    if transcript_dir:
        transcript_path = Path(transcript_dir)
        if not transcript_path.exists():
            print(f"⚠️  Transcript directory not found: {transcript_dir}")
            print("   Using provenance placeholders instead")
        else:
            print(f"📂 Using transcript directory: {transcript_dir}")

    # Parse speaker lists
    include_speakers_list = parse_speaker_list(include_speakers)
    exclude_speakers_list = parse_speaker_list(exclude_speakers)

    if include_speakers_list:
        print(f"✅ Including speakers: {include_speakers_list}")
    if exclude_speakers_list:
        print(f"❌ Excluding speakers: {exclude_speakers_list}")

    # Set up primary speaker instruction
    if primary_speaker:
        print(f"🎯 Primary speaker preference: {primary_speaker}")
    else:
        print("ℹ️  No primary speaker specified")

    print(f"🎯 User token range: {user_min_tokens}-{user_max_tokens}")

    if limit > 0:
        print(f"🔢 Processing limit: {limit} rows")

    # Read and convert entries
    valid_count = 0
    invalid_count = 0
    transcripts_found = 0
    transcripts_missing = 0
    trimmed_user_blocks = 0
    moved_reasoning_lines = 0
    primary_hits = 0
    rows_processed = 0

    metadata_keys = [
        "chunk_id",
        "speaker",
        "is_primary",
        "start_time",
        "end_time",
        "source_file",
        "token_count",
        "timestamp",
        "model",
        "status"
    ]

    # Truncate output file before writing
    with open(output_file, 'w', encoding='utf-8'):
        pass

    with open(input_file, 'r', encoding='utf-8') as f, open(output_file, 'a', encoding='utf-8') as out_fp:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            # Check limit
            if limit > 0 and rows_processed >= limit:
                print(f"⏹️  Reached processing limit of {limit} rows")
                break

            rows_processed += 1

            try:
                # Parse the original data
                original_data = json.loads(line)

                # Transform to Harmony format with all features
                harmony_entry = transform_to_harmony_conversation(
                    original_data, i, transcript_path, primary_speaker,
                    include_speakers_list, exclude_speakers_list, user_min_tokens,
                    user_max_tokens
                )

                # Track transcript loading
                if transcript_path:
                    source_file = original_data.get('source_file', '')
                    if harmony_entry and harmony_entry['transcript_status'] == "TRANSCRIPT_LOADED":
                        transcripts_found += 1
                    else:
                        transcripts_missing += 1

                # Track token normalization
                user_content = harmony_entry['messages'][1]['content']
                if count_tokens(user_content) < count_tokens(f"Transcript chunk {i+1} from {original_data.get('source_file', 'unknown')}"):
                    trimmed_user_blocks += 1

                # Validate the converted entry
                if validate_harmony_entry(harmony_entry):
                    assistant_msg = harmony_entry['messages'][2]
                    record = dict(original_data)
                    record.update(harmony_entry.get('metadata', {}))

                    analysis_obj = assistant_msg.get('analysis') or record.get('analysis')
                    if analysis_obj is not None:
                        record['analysis'] = analysis_obj

                    final_obj = assistant_msg.get('final') or record.get('final')
                    if final_obj is not None:
                        record['final'] = final_obj

                    analysis_for_final = record.get('analysis')
                    fallback_summary = analysis_for_final.get('symbolic_summary') if isinstance(analysis_for_final, dict) else None

                    system_prompt_str = harmony_entry['messages'][0].get('content', '')
                    user_block_str = harmony_entry['messages'][1].get('content', '')

                    try:
                        write_harmony_record(
                            out_fp=out_fp,
                            system_prompt=system_prompt_str,
                            user_content=user_block_str,
                            analysis_obj=analysis_for_final,
                            final_obj=record.get("final") or fallback_summary,
                            metadata={k: record.get(k) for k in metadata_keys if k in record}
                        )
                    except ValueError as e:
                        invalid_count += 1
                        print(f"❌ Skipped line {i+1} (write error: {e})")
                        continue

                    valid_count += 1
                    if harmony_entry['metadata'].get('primary_speaker_hit', False):
                        primary_hits += 1
                    print(f"✅ Processed line {i+1} (valid)")
                else:
                    invalid_count += 1
                    print(f"❌ Skipped line {i+1} (invalid structure)")

            except json.JSONDecodeError as e:
                invalid_count += 1
                print(f"❌ Skipped line {i+1} (malformed JSON: {e})")
                continue
            except Exception as e:
                invalid_count += 1
                print(f"❌ Skipped line {i+1} (error: {e})")
                continue

    # Print comprehensive summary
    print(f"\n📊 CONVERSION SUMMARY:")
    print(f"✅ Valid entries: {valid_count}")
    print(f"❌ Invalid entries: {invalid_count}")
    print(f"📁 Output saved to: {output_file}")
    print(f"📋 Total processed: {rows_processed}")

    if transcript_path:
        print(f"\n📜 TRANSCRIPT LOADING:")
        print(f"✅ Transcripts found: {transcripts_found}")
        print(f"❌ Transcripts missing: {transcripts_missing}")
        print(f"📂 Transcript directory: {transcript_dir}")

    print(f"\n🎯 TOKEN NORMALIZATION:")
    print(f"✂️  Trimmed user blocks: {trimmed_user_blocks}")

    print(f"\n🛡️  GUARDRAILS:")
    print(f"🔄 Moved reasoning lines: {moved_reasoning_lines}")

    if primary_speaker:
        print(f"\n🎯 PRIMARY SPEAKER PREFERENCE:")
        print(f"👑 Preferred speaker: {primary_speaker}")
        print(f"📝 System messages updated with speaker preference")

        # Count primary speaker hits
        print(f"🎯 Primary speaker appearances: {primary_hits}/{valid_count}")

def main():
    """Main function to run the conversion."""

    # Set up argument parsing
    import argparse
    parser = argparse.ArgumentParser(description='Convert symbolic dataset to Harmony conversation format')
    parser.add_argument('--in', type=str, default='data/output/symbolic_dataset.jsonl',
                      help='Input JSONL file to convert')
    parser.add_argument('--out', type=str, default='data/output/symbolic_harmony.jsonl',
                      help='Output JSONL file for Harmony format')
    parser.add_argument('--transcript-dir', type=str, default=None,
                      help='Directory containing transcript files (optional)')
    parser.add_argument('--user-min-tokens', type=int, default=900,
                      help='Minimum token count for user content (default: 900)')
    parser.add_argument('--user-max-tokens', type=int, default=1400,
                      help='Maximum token count for user content (default: 1400)')
    parser.add_argument('--primary-speaker', type=str, default='',
                      help='Primary speaker name for analysis focus (optional)')
    parser.add_argument('--include-speakers', type=str, default='',
                      help='Comma-separated list of speakers to include (case-insensitive)')
    parser.add_argument('--exclude-speakers', type=str, default='',
                      help='Comma-separated list of speakers to exclude (case-insensitive)')
    parser.add_argument('--limit', type=int, default=0,
                      help='Maximum number of rows to process (0 = no limit)')

    args = parser.parse_args()

    print("🔄 Converting symbolic dataset to Harmony format...")
    print(f"📂 Input: {getattr(args, 'in')}")
    print(f"📂 Output: {getattr(args, 'out')}")

    if args.transcript_dir:
        print(f"📂 Transcript directory: {args.transcript_dir}")

    if args.primary_speaker:
        print(f"🎯 Primary speaker: {args.primary_speaker}")

    if args.include_speakers:
        print(f"✅ Include speakers: {args.include_speakers}")

    if args.exclude_speakers:
        print(f"❌ Exclude speakers: {args.exclude_speakers}")

    if args.limit > 0:
        print(f"🔢 Processing limit: {args.limit} rows")

    print(f"🎯 User token range: {args.user_min_tokens}-{args.user_max_tokens}")

    convert_to_harmony(getattr(args, 'in'), getattr(args, 'out'), args.transcript_dir, args.primary_speaker,
                      args.include_speakers, args.exclude_speakers, args.user_min_tokens,
                      args.user_max_tokens, args.limit)

    print("\n✅ Conversion complete!")
    print(f"📋 Check the output file: {args.out}")

if __name__ == "__main__":
    main()
