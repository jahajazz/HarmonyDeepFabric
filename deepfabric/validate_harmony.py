#!/usr/bin/env python3
"""
Validate Harmony conversation format files.

This script randomly samples n items from a Harmony JSONL file and validates
their structure, content quality, and token budgets.
"""

import json
import sys
import os
import random
import argparse
import re
from typing import Dict, Any, List, Tuple
from pathlib import Path

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

def load_harmony_entries(input_file: str) -> List[Dict[str, Any]]:
    """Load all Harmony entries from JSONL file."""
    entries = []

    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        sys.exit(1)

    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
                if isinstance(entry, dict):
                    entries.append((line_num, entry))
            except json.JSONDecodeError as e:
                print(f"⚠️  Skipping line {line_num}: Invalid JSON - {e}")

    return entries

def validate_message_order(messages: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """Validate that messages follow system→user→assistant order."""
    if len(messages) < 3:
        return False, f"Only {len(messages)} messages (need at least 3)"

    expected_order = ["system", "user", "assistant"]
    actual_order = [msg.get("role") for msg in messages[:3]]

    if actual_order != expected_order:
        return False, f"Expected {expected_order}, got {actual_order}"

    return True, "Valid order"

def validate_assistant_structure(assistant_msg: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate assistant message has required analysis and final fields."""
    if not isinstance(assistant_msg, dict):
        return False, "Assistant message is not a dict"

    missing_fields = []

    if "analysis" not in assistant_msg:
        missing_fields.append("analysis")
    elif not isinstance(assistant_msg["analysis"], dict):
        return False, "analysis field is not a dict"

    if "final" not in assistant_msg:
        missing_fields.append("final")
    elif not isinstance(assistant_msg["final"], dict):
        return False, "final field is not a dict"

    if missing_fields:
        return False, f"Missing fields: {missing_fields}"

    return True, "Valid structure"

def detect_reasoning_markers(text: str) -> List[str]:
    """Detect obvious reasoning markers in text."""
    markers = []

    # Check for reasoning headers
    reasoning_patterns = [
        r'^Reasoning:',
        r'^Let\'s think',
        r'^\d+\.',  # Numbered steps like "1.", "2.", etc.
        r'^\*\s*\d+\.',  # Bold numbered steps
    ]

    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        for pattern in reasoning_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                markers.append(line[:50] + "..." if len(line) > 50 else line)

    return markers

def validate_final_content(final_content: str, max_tokens: int) -> Tuple[bool, str]:
    """Validate final content doesn't contain reasoning markers and fits token budget."""
    # Check token budget
    tokens = count_tokens(final_content)
    if tokens > max_tokens:
        return False, f"Exceeds token budget: {tokens} > {max_tokens}"

    # Check for reasoning markers
    markers = detect_reasoning_markers(final_content)
    if markers:
        return False, f"Contains reasoning markers: {markers[:2]}"  # Show first 2

    return True, f"Valid content ({tokens} tokens)"

def print_compact_table(entries: List[Tuple[int, Dict[str, Any]]], max_tokens: int) -> None:
    """Print compact validation table."""
    print("\n📊 VALIDATION TABLE")
    print("=" * 80)
    print(f"{'Idx':<5} {'Order':<8} {'Len(A)':<8} {'Len(F)':<8} {'U-Tok':<8} {'F-Tok':<8} {'Status'}")
    print("-" * 80)

    for idx, (line_num, entry) in enumerate(entries, 1):
        # Extract data
        messages = entry.get("messages", [])
        has_system = any(msg.get("role") == "system" for msg in messages)
        has_user = any(msg.get("role") == "user" for msg in messages)
        has_assistant = any(msg.get("role") == "assistant" for msg in messages)

        # Get assistant message
        assistant_msg = None
        for msg in messages:
            if msg.get("role") == "assistant":
                assistant_msg = msg
                break

        if not assistant_msg:
            status = "❌ No assistant"
            len_analysis = len_final = u_tokens = f_tokens = 0
        else:
            # Validate structure
            structure_valid, structure_msg = validate_assistant_structure(assistant_msg)
            if not structure_valid:
                status = f"❌ {structure_msg}"
                len_analysis = len_final = u_tokens = f_tokens = 0
            else:
                # Get lengths and tokens
                analysis = assistant_msg.get("analysis", {})
                final = assistant_msg.get("final", {})

                len_analysis = len(analysis) if analysis else 0
                len_final = len(final) if final else 0

                # Get content for token counting
                user_content = ""
                for msg in messages:
                    if msg.get("role") == "user":
                        user_content = msg.get("content", "")
                        break

                final_content = final.get("summary", "") if final else ""

                u_tokens = count_tokens(user_content)
                f_tokens = count_tokens(final_content)

                # Validate content
                content_valid, content_msg = validate_final_content(final_content, max_tokens)

                if not content_valid:
                    status = f"❌ {content_msg}"
                else:
                    status = "✅ Valid"

        # Print row
        order_status = "✅" if (has_system and has_user and has_assistant) else "❌"
        print(f"{idx:<5} {order_status:<8} {len_analysis:<8} {len_final:<8} {u_tokens:<8} {f_tokens:<8} {status}")

def print_statistics(entries: List[Tuple[int, Dict[str, Any]]], max_tokens: int) -> None:
    """Print overall statistics and histograms."""
    if not entries:
        print("❌ No valid entries to analyze")
        return

    print("\n📈 STATISTICS")
    print("=" * 50)

    # Collect data for analysis
    user_tokens_list = []
    final_tokens_list = []
    valid_count = 0
    invalid_count = 0

    for line_num, entry in entries:
        messages = entry.get("messages", [])
        assistant_msg = None

        for msg in messages:
            if msg.get("role") == "assistant":
                assistant_msg = msg
                break

        if not assistant_msg:
            invalid_count += 1
            continue

        # Validate structure
        structure_valid, _ = validate_assistant_structure(assistant_msg)
        if not structure_valid:
            invalid_count += 1
            continue

        # Get content for token counting
        user_content = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_content = msg.get("content", "")
                break

        final = assistant_msg.get("final", {})
        final_content = final.get("summary", "") if final else ""

        user_tokens = count_tokens(user_content)
        final_tokens = count_tokens(final_content)

        # Check if valid
        content_valid, _ = validate_final_content(final_content, max_tokens)
        if content_valid:
            valid_count += 1
            user_tokens_list.append(user_tokens)
            final_tokens_list.append(final_tokens)
        else:
            invalid_count += 1

    total_count = len(entries)
    print(f"📊 Total entries analyzed: {total_count}")
    print(f"✅ Valid entries: {valid_count}")
    print(f"❌ Invalid entries: {invalid_count}")
    print(f"📈 Success rate: {valid_count/total_count*100:.1f}%" if total_count > 0 else "N/A")

    if user_tokens_list:
        print("\n🎯 USER TOKENS DISTRIBUTION:")
        print(f"   Min: {min(user_tokens_list)}")
        print(f"   Max: {max(user_tokens_list)}")
        print(f"   Avg: {sum(user_tokens_list)/len(user_tokens_list):.1f}")

        # Simple histogram
        print("   Histogram (buckets of 200):")
        buckets = {}
        for tokens in user_tokens_list:
            bucket = (tokens // 200) * 200
            buckets[bucket] = buckets.get(bucket, 0) + 1

        for bucket in sorted(buckets.keys()):
            count = buckets[bucket]
            bar = "█" * min(count * 2, 20)  # Scale bars
            print(f"   {bucket:4d}-{bucket+199:4d}: {bar} ({count})")

    if final_tokens_list:
        print("\n📝 FINAL TOKENS DISTRIBUTION:")
        print(f"   Min: {min(final_tokens_list)}")
        print(f"   Max: {max(final_tokens_list)}")
        print(f"   Avg: {sum(final_tokens_list)/len(final_tokens_list):.1f}")

        # Check against max_tokens
        over_budget = sum(1 for t in final_tokens_list if t > max_tokens)
        print(f"   Over budget ({max_tokens}): {over_budget}/{len(final_tokens_list)} ({over_budget/len(final_tokens_list)*100:.1f}%)")

def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description='Validate Harmony conversation format')
    parser.add_argument('--in', type=str, required=True,
                      help='Input JSONL file to validate')
    parser.add_argument('--n', type=int, default=12,
                      help='Number of items to sample (default: 12)')
    parser.add_argument('--final-max-tokens', type=int, default=400,
                      help='Maximum tokens allowed in final content (default: 400)')

    args = parser.parse_args()

    print(f"🔍 Validating Harmony format: {getattr(args, 'in')}")
    print(f"📊 Sampling {args.n} items, max final tokens: {args.final_max_tokens}")

    # Load entries
    all_entries = load_harmony_entries(getattr(args, 'in'))
    print(f"📂 Loaded {len(all_entries)} entries from {getattr(args, 'in')}")

    if len(all_entries) == 0:
        print("❌ No valid entries found")
        sys.exit(1)

    # Randomly sample n entries
    if len(all_entries) <= args.n:
        sampled_entries = all_entries
        print(f"📋 Using all {len(all_entries)} entries (less than requested {args.n})")
    else:
        sampled_entries = random.sample(all_entries, args.n)
        print(f"🎲 Randomly sampled {args.n} entries")

    # Validate each entry and check fail conditions
    failed_entries = []

    for i, (line_num, entry) in enumerate(sampled_entries, 1):
        messages = entry.get("messages", [])

        # Check message order
        order_valid, order_msg = validate_message_order(messages)
        if not order_valid:
            failed_entries.append((i, f"Line {line_num}: Invalid message order - {order_msg}"))
            continue

        # Find assistant message
        assistant_msg = None
        for msg in messages:
            if msg.get("role") == "assistant":
                assistant_msg = msg
                break

        if not assistant_msg:
            failed_entries.append((i, f"Line {line_num}: No assistant message found"))
            continue

        # Check assistant structure
        structure_valid, structure_msg = validate_assistant_structure(assistant_msg)
        if not structure_valid:
            failed_entries.append((i, f"Line {line_num}: Invalid assistant structure - {structure_msg}"))
            continue

        # Check final content
        final = assistant_msg.get("final", {})
        final_content = final.get("summary", "") if final else ""
        content_valid, content_msg = validate_final_content(final_content, args.final_max_tokens)

        if not content_valid:
            failed_entries.append((i, f"Line {line_num}: Invalid final content - {content_msg}"))

    # Print results table
    print_compact_table(sampled_entries, args.final_max_tokens)

    # Print statistics
    print_statistics(sampled_entries, args.final_max_tokens)

    # Check fail conditions
    if failed_entries:
        print(f"\n❌ VALIDATION FAILED")
        print("=" * 50)
        for idx, error in failed_entries:
            print(f"  [{idx}] {error}")
        print(f"\n🔥 {len(failed_entries)}/{len(sampled_entries)} entries failed validation")
        sys.exit(1)

    print("\n✅ VALIDATION PASSED")
    print("=" * 50)
    print(f"🎉 All {len(sampled_entries)} sampled entries passed validation!")

if __name__ == "__main__":
    main()
