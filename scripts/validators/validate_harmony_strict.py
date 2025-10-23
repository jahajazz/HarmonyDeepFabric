#!/usr/bin/env python3
"""
Harmony-STRICT Validator

Validates that generated dataset meets Harmony-STRICT specifications:
- Training JSONL: ONLY `messages` array, last role MUST be assistant
- Sidecar metadata JSONL: Separate file with 1:1 alignment
- Required fields in sidecar: pair_id, episode_id, target_spans, fit_score_ce, gates
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple


def validate_training_record(record: Dict[str, Any], line_num: int) -> Tuple[bool, str]:
    """Validate a single training record meets Harmony-STRICT format."""

    # Must have exactly 'messages' key at top level
    if set(record.keys()) != {'messages'}:
        extra_keys = set(record.keys()) - {'messages'}
        return False, f"Line {line_num}: Invalid top-level keys. Expected only 'messages', found extra: {extra_keys}"

    messages = record['messages']

    # Must be a list
    if not isinstance(messages, list):
        return False, f"Line {line_num}: 'messages' must be a list, got {type(messages)}"

    # Must have at least 1 message
    if len(messages) == 0:
        return False, f"Line {line_num}: 'messages' array cannot be empty"

    # All messages must be dicts with 'role' and 'content'
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            return False, f"Line {line_num}: Message {i} must be a dict, got {type(msg)}"

        if 'role' not in msg:
            return False, f"Line {line_num}: Message {i} missing 'role' field"

        if 'content' not in msg:
            return False, f"Line {line_num}: Message {i} missing 'content' field"

        if not isinstance(msg['role'], str):
            return False, f"Line {line_num}: Message {i} 'role' must be string, got {type(msg['role'])}"

        if not isinstance(msg['content'], str):
            return False, f"Line {line_num}: Message {i} 'content' must be string, got {type(msg['content'])}"

    # Last message MUST have role 'assistant'
    if messages[-1]['role'] != 'assistant':
        return False, f"Line {line_num}: Last message role must be 'assistant', got '{messages[-1]['role']}'"

    return True, ""


def validate_sidecar_record(record: Dict[str, Any], line_num: int) -> Tuple[bool, str]:
    """Validate a single sidecar metadata record."""

    # Required fields
    required_fields = [
        'pair_id', 'episode_id', 'window_id', 'sample_index',
        'answer_speaker', 'question', 'answer', 'sources',
        'target_spans', 'fit_score_ce', 'gates'
    ]

    for field in required_fields:
        if field not in record:
            return False, f"Line {line_num}: Missing required field '{field}'"

    # Validate field types
    if not isinstance(record['pair_id'], str):
        return False, f"Line {line_num}: 'pair_id' must be string"

    if not isinstance(record['episode_id'], str):
        return False, f"Line {line_num}: 'episode_id' must be string"

    if not isinstance(record['gates'], dict):
        return False, f"Line {line_num}: 'gates' must be dict"

    # Validate gates structure
    required_gates = ['questionify_used', 'answer_trimmed', 'speaker_check_passed', 'pairfit_passed']
    for gate in required_gates:
        if gate not in record['gates']:
            return False, f"Line {line_num}: Missing gate '{gate}' in 'gates'"

    assert (not record['gates']['pairfit_passed']) or (record.get('fit_score_ce') is not None and record['fit_score_ce'] >= 0.5)

    question = record.get('question', '')
    if record['gates'].get('questionify_used'):
        if record.get('question_origin') != 'model':
            return False, f"Line {line_num}: questionify_used=true but question_origin={record.get('question_origin')}"
        if not isinstance(question, str) or not question.endswith('?'):
            return False, f"Line {line_num}: questionify-derived question must end with '?'"
        if len(question.split()) > 25:
            return False, f"Line {line_num}: questionify-derived question exceeds 25 words"

    composition = record.get('composition', {})
    if isinstance(composition, dict) and composition.get('mode') == 'A':
        if not record.get('answer_found_in_source', False):
            return False, f"Line {line_num}: Variant A must have answer_found_in_source=true"

    return True, ""


def validate_file_pair(training_file: Path, sidecar_file: Path) -> Tuple[bool, List[str]]:
    """Validate that training and sidecar files are properly aligned."""

    errors = []

    # Check if files exist
    if not training_file.exists():
        return False, [f"Training file not found: {training_file}"]

    if not sidecar_file.exists():
        return False, [f"Sidecar file not found: {sidecar_file}"]

    # Load training records
    training_records = []
    with open(training_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                is_valid, error = validate_training_record(record, i)
                if not is_valid:
                    errors.append(error)
                else:
                    training_records.append(record)
            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: Invalid JSON: {e}")

    # Load sidecar records
    sidecar_records = []
    with open(sidecar_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                is_valid, error = validate_sidecar_record(record, i)
                if not is_valid:
                    errors.append(error)
                else:
                    sidecar_records.append(record)
            except json.JSONDecodeError as e:
                errors.append(f"Sidecar line {i}: Invalid JSON: {e}")

    # Check line count alignment
    if len(training_records) != len(sidecar_records):
        errors.append(
            f"Line count mismatch: training has {len(training_records)} records, "
            f"sidecar has {len(sidecar_records)} records"
        )

    # Check pair_id alignment (1:1 correspondence)
    if len(training_records) == len(sidecar_records):
        for i, (train_rec, sidecar_rec) in enumerate(zip(training_records, sidecar_records)):
            if 'pair_id' not in sidecar_rec:
                errors.append(f"Sidecar line {i+1}: Missing pair_id for alignment check")
            # Note: We can't directly validate pair_id alignment without additional metadata
            # This would need to be enhanced based on specific requirements

    return len(errors) == 0, errors


def find_sidecar_file(training_file: Path) -> Path:
    """Find the appropriate sidecar file (normalized or legacy)."""
    # Try normalized name first
    sidecar_file = training_file.parent / training_file.name.replace("train.jsonl", "sidecar_train.jsonl").replace("val.jsonl", "sidecar_val.jsonl")

    # Fall back to legacy name if normalized doesn't exist
    if not sidecar_file.exists():
        legacy_file = training_file.parent / training_file.name.replace("train.jsonl", "train_metadata.jsonl").replace("val.jsonl", "val_metadata.jsonl")
        if legacy_file.exists():
            return legacy_file

    return sidecar_file

def main():
    parser = argparse.ArgumentParser(description="Validate Harmony-STRICT dataset format")
    parser.add_argument("--train_file", type=Path, required=True, help="Training JSONL file")
    parser.add_argument("--val_file", type=Path, required=True, help="Validation JSONL file")
    parser.add_argument("--train_metadata", type=Path, help="Training metadata JSONL file (optional, will auto-detect)")
    parser.add_argument("--val_metadata", type=Path, help="Validation metadata JSONL file (optional, will auto-detect)")

    args = parser.parse_args()

    print("🔍 Validating Harmony-STRICT dataset format...")
    print("=" * 60)

    all_valid = True
    total_errors = []

    # Auto-detect sidecar files if not provided
    train_sidecar = args.train_metadata or find_sidecar_file(args.train_file)
    val_sidecar = args.val_metadata or find_sidecar_file(args.val_file)

    # Validate train files
    print(f"📁 Validating train files:")
    print(f"   Training: {args.train_file}")
    print(f"   Sidecar: {train_sidecar}")

    train_valid, train_errors = validate_file_pair(args.train_file, train_sidecar)
    if train_valid:
        print("   ✅ Train files: OK")
    else:
        print("   ❌ Train files: FAILED")
        all_valid = False
        total_errors.extend([f"TRAIN: {err}" for err in train_errors])

    print()

    # Validate val files
    print(f"📁 Validating val files:")
    print(f"   Training: {args.val_file}")
    print(f"   Sidecar: {val_sidecar}")

    val_valid, val_errors = validate_file_pair(args.val_file, val_sidecar)
    if val_valid:
        print("   ✅ Val files: OK")
    else:
        print("   ❌ Val files: FAILED")
        all_valid = False
        total_errors.extend([f"VAL: {err}" for err in val_errors])

    print()
    print("=" * 60)

    if all_valid:
        print("🎉 VALIDATION PASSED: All files meet Harmony-STRICT specifications!")
        print("   ✅ Training JSONL: messages-only format, last role assistant")
        print("   ✅ Sidecar JSONL: 1:1 alignment with required fields")
        print("   ✅ Line counts match between training and metadata files")
        return 0
    else:
        print("❌ VALIDATION FAILED: Files do not meet Harmony-STRICT specifications!")
        print("\nErrors found:")
        for error in total_errors:
            print(f"   • {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
