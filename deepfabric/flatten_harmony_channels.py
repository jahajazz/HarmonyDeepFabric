#!/usr/bin/env python3
"""
Flatten Harmony conversation channels.

This script processes Harmony JSONL files and flattens the structured
analysis and final fields into simple strings, optionally removing
the content field when both channels exist.
"""

import json
import sys
import os
import argparse
from typing import Dict, Any, List, Tuple

def flatten_analysis(analysis_data: Any) -> str:
    """Convert analysis dict to formatted string."""
    if not isinstance(analysis_data, dict):
        return str(analysis_data) if analysis_data else ""

    symbolic_meaning = analysis_data.get('symbolic_meaning', '')
    archetypal_patterns = analysis_data.get('archetypal_patterns', [])
    theological_themes = analysis_data.get('theological_themes', [])

    # Format as specified
    lines = []

    if symbolic_meaning:
        lines.append(f"Symbolic reasoning:\n{symbolic_meaning}")

    if archetypal_patterns:
        patterns_str = ', '.join(str(p) for p in archetypal_patterns)
        lines.append(f"Archetypal patterns: {patterns_str}")

    if theological_themes:
        themes_str = ', '.join(str(t) for t in theological_themes)
        lines.append(f"Theological themes: {themes_str}")

    return '\n\n'.join(lines)

def flatten_final(final_data: Any) -> str:
    """Convert final dict to string from summary or related fields."""
    if not isinstance(final_data, dict):
        return str(final_data) if final_data else ""

    # Try different possible field names for the final content
    for field_name in ['summary', 'symbolic_summary', 'final']:
        if field_name in final_data and final_data[field_name]:
            return str(final_data[field_name])

    return ""

def should_remove_content(analysis_str: str, final_str: str) -> bool:
    """Determine if content field should be removed."""
    return bool(analysis_str.strip() and final_str.strip())

def process_harmony_entry(entry: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Process a single Harmony entry and return (processed_entry, was_modified)."""
    if not isinstance(entry, dict) or 'messages' not in entry:
        return entry, False

    messages = entry.get('messages', [])
    modified = False

    for message in messages:
        if not isinstance(message, dict) or message.get('role') != 'assistant':
            continue

        # Process analysis field
        if 'analysis' in message:
            original_analysis = message['analysis']
            flattened_analysis = flatten_analysis(original_analysis)

            if flattened_analysis != str(original_analysis):
                message['analysis'] = flattened_analysis
                modified = True

        # Process final field
        if 'final' in message:
            original_final = message['final']
            flattened_final = flatten_final(original_final)

            if flattened_final != str(original_final):
                message['final'] = flattened_final
                modified = True

        # Remove content field if both analysis and final are non-empty strings
        if 'analysis' in message and 'final' in message and 'content' in message:
            analysis_str = str(message['analysis'])
            final_str = str(message['final'])

            if should_remove_content(analysis_str, final_str):
                del message['content']
                modified = True

    return entry, modified

def main():
    """Main flattening function."""
    parser = argparse.ArgumentParser(description='Flatten Harmony conversation channels')
    parser.add_argument('--in', type=str, required=True,
                      help='Input JSONL file to flatten')
    parser.add_argument('--out', type=str, required=True,
                      help='Output JSONL file for flattened format')

    args = parser.parse_args()

    print(f"🔄 Flattening Harmony channels: {getattr(args, 'in')}")
    print(f"📂 Output: {args.out}")

    # Check input file exists
    if not os.path.exists(getattr(args, 'in')):
        print(f"❌ Input file not found: {getattr(args, 'in')}")
        sys.exit(1)

    # Process entries
    converted_count = 0
    skipped_count = 0
    total_processed = 0

    with open(getattr(args, 'in'), 'r', encoding='utf-8') as infile, \
         open(args.out, 'w', encoding='utf-8') as outfile:

        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue

            total_processed += 1

            try:
                entry = json.loads(line)
                processed_entry, was_modified = process_harmony_entry(entry)

                # Write processed entry
                outfile.write(json.dumps(processed_entry, ensure_ascii=False) + '\n')

                if was_modified:
                    converted_count += 1
                    print(f"✅ Processed line {line_num} (modified)")
                else:
                    skipped_count += 1
                    print(f"⏭️  Processed line {line_num} (unchanged)")

            except json.JSONDecodeError as e:
                skipped_count += 1
                print(f"❌ Skipped line {line_num} (malformed JSON: {e})")
                continue
            except Exception as e:
                skipped_count += 1
                print(f"❌ Skipped line {line_num} (error: {e})")
                continue

    # Print summary
    print(f"\n📊 FLATTENING SUMMARY:")
    print(f"📋 Total processed: {total_processed}")
    print(f"✅ Modified entries: {converted_count}")
    print(f"⏭️  Unchanged entries: {skipped_count}")
    print(f"📁 Output saved to: {args.out}")

    if converted_count > 0:
        print(f"\n🎯 FLATTENING RESULTS:")
        print(f"📝 Analysis fields converted to strings")
        print(f"📄 Final fields extracted from summaries")
        print(f"🗑️  Content fields removed where both channels exist")

    print(f"\n✅ Flattening complete!")

if __name__ == "__main__":
    main()
