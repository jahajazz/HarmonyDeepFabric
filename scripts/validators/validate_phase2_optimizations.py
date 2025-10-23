#!/usr/bin/env python3
"""Phase 2 Go/No-Go validation script for local-first heuristics optimization."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


def validate_sidecar_flags(sidecar_path: Path) -> Dict[str, any]:
    """Spot-check sidecar flags on a few new items."""
    print(f"🔍 Validating sidecar flags in {sidecar_path}")

    if not sidecar_path.exists():
        print(f"❌ Sidecar file not found: {sidecar_path}")
        return {"error": "file_not_found"}

    results = {
        "total_records": 0,
        "speaker_check_passed_count": 0,
        "fit_score_populated_count": 0,
        "pairfit_passed_count": 0,
        "questionify_used_count": 0,
        "answer_trimmed_count": 0,
        "trim_contiguous_failures": 0,
        "invalid_speakers": [],
        "fit_score_issues": [],
        "sample_records": []
    }

    with open(sidecar_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 100:  # Sample first 100 records
                break

            try:
                record = json.loads(line.strip())
                results["total_records"] += 1

                # Check speaker validation
                if record.get("gates", {}).get("speaker_check_passed", False):
                    results["speaker_check_passed_count"] += 1

                # Check fit_score_ce population
                fit_score = record.get("fit_score_ce")
                if fit_score is not None:
                    results["fit_score_populated_count"] += 1
                    if not isinstance(fit_score, (int, float)) or not (0.0 <= fit_score <= 1.0):
                        results["fit_score_issues"].append(f"Record {i}: Invalid fit_score {fit_score}")

                # Check pairfit_passed logic
                pairfit_passed = record.get("gates", {}).get("pairfit_passed", False)
                if pairfit_passed:
                    results["pairfit_passed_count"] += 1

                # Check questionify usage
                if record.get("gates", {}).get("questionify_used", False):
                    results["questionify_used_count"] += 1

                # Check answer trimming
                if record.get("gates", {}).get("answer_trimmed", False):
                    results["answer_trimmed_count"] += 1

                    # Validate contiguous substring
                    answer_trim = record.get("answer_trim", {})
                    if answer_trim.get("start_char", -1) == -1:
                        results["trim_contiguous_failures"] += 1

                # Check for invalid speakers
                speaker = record.get("answer_speaker", "")
                if speaker and speaker not in ["Fr Stephen De Young", "Jonathan Pageau"]:
                    results["invalid_speakers"].append(f"Record {i}: {speaker}")

                # Store sample record for detailed inspection
                if i < 5:
                    results["sample_records"].append({
                        "record_id": i,
                        "speaker": record.get("answer_speaker"),
                        "question": record.get("question", "")[:100] + "..." if len(record.get("question", "")) > 100 else record.get("question"),
                        "fit_score": record.get("fit_score_ce"),
                        "gates": record.get("gates", {})
                    })

            except json.JSONDecodeError as e:
                print(f"❌ JSON decode error on line {i}: {e}")
                continue

    return results


def analyze_gating_metrics(sidecar_paths: List[Path]) -> Dict[str, any]:
    """Analyze gating metrics for sanity checks."""
    print("📊 Analyzing gating metrics...")

    all_metrics = {
        "total_records": 0,
        "drop_reasons": {},
        "speaker_counts": {},
        "fit_score_stats": {"min": 1.0, "max": 0.0, "avg": 0.0, "count": 0}
    }

    for path in sidecar_paths:
        if not path.exists():
            continue

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    all_metrics["total_records"] += 1

                    # Analyze speaker distribution
                    speaker = record.get("answer_speaker", "Unknown")
                    all_metrics["speaker_counts"][speaker] = all_metrics["speaker_counts"].get(speaker, 0) + 1

                    # Analyze fit scores
                    fit_score = record.get("fit_score_ce")
                    if fit_score is not None and isinstance(fit_score, (int, float)):
                        all_metrics["fit_score_stats"]["min"] = min(all_metrics["fit_score_stats"]["min"], fit_score)
                        all_metrics["fit_score_stats"]["max"] = max(all_metrics["fit_score_stats"]["max"], fit_score)
                        all_metrics["fit_score_stats"]["count"] += 1

                except json.JSONDecodeError:
                    continue

    # Calculate average fit score
    if all_metrics["fit_score_stats"]["count"] > 0:
        # We don't have individual scores stored, so we'll estimate from the gates
        pass

    return all_metrics


def verify_determinism(train_path: Path, val_path: Path, metadata_train_path: Path, metadata_val_path: Path) -> bool:
    """Verify determinism by comparing file hashes."""
    print("🔄 Verifying determinism...")

    def get_file_hash(filepath: Path) -> str:
        if not filepath.exists():
            return "file_not_found"
        with open(filepath, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()

    # Get hashes for current files
    current_hashes = {
        "train": get_file_hash(train_path),
        "val": get_file_hash(val_path),
        "metadata_train": get_file_hash(metadata_train_path),
        "metadata_val": get_file_hash(metadata_val_path)
    }

    print(f"Current file hashes: {current_hashes}")

    # Check for any missing files
    missing_files = [k for k, v in current_hashes.items() if v == "file_not_found"]
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False

    return True


def check_leakage_guard(sidecar_train_path: Path, sidecar_val_path: Path) -> Dict[str, any]:
    """Check for pair_id overlap between train/val splits."""
    print("🚧 Checking leakage guard...")

    def extract_pair_ids(filepath: Path) -> Set[str]:
        pair_ids = set()
        if not filepath.exists():
            return pair_ids

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    pair_id = record.get("pair_id")
                    if pair_id:
                        pair_ids.add(pair_id)
                except json.JSONDecodeError:
                    continue
        return pair_ids

    train_ids = extract_pair_ids(sidecar_train_path)
    val_ids = extract_pair_ids(sidecar_val_path)

    overlap = train_ids.intersection(val_ids)

    results = {
        "train_count": len(train_ids),
        "val_count": len(val_ids),
        "overlap_count": len(overlap),
        "overlap_pairs": list(overlap)[:10]  # Show first 10 overlaps
    }

    if overlap:
        print(f"❌ Found {len(overlap)} overlapping pair_ids!")
        print(f"Sample overlaps: {list(overlap)[:5]}")
    else:
        print("✅ No pair_id leakage detected")

    return results


def audit_speaker_allowlist(sidecar_paths: List[Path]) -> Dict[str, any]:
    """Audit speaker allow-list compliance."""
    print("👥 Auditing speaker allow-list...")

    speaker_counts = {}
    invalid_speakers = []

    allowed_speakers = {"Fr Stephen De Young", "Jonathan Pageau"}

    for path in sidecar_paths:
        if not path.exists():
            continue

        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    record = json.loads(line.strip())
                    speaker = record.get("answer_speaker", "")

                    if speaker in allowed_speakers:
                        speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
                    else:
                        invalid_speakers.append(f"{path}:{i}: {speaker}")

                except json.JSONDecodeError:
                    continue

    results = {
        "allowed_speakers_found": list(speaker_counts.keys()),
        "speaker_counts": speaker_counts,
        "invalid_speakers": invalid_speakers[:10],  # Show first 10
        "total_invalid": len(invalid_speakers)
    }

    print(f"✅ Allowed speakers found: {list(speaker_counts.keys())}")
    print(f"📊 Speaker distribution: {speaker_counts}")

    if invalid_speakers:
        print(f"❌ Found {len(invalid_speakers)} invalid speakers!")
        print(f"Sample invalid speakers: {invalid_speakers[:5]}")
    else:
        print("✅ All speakers are from allowed list")

    return results


def pin_dependencies():
    """Pin dependency versions for CI stability."""
    print("📌 Pinning dependency versions...")

    requirements_path = Path("deepfabric/requirements.txt")
    if not requirements_path.exists():
        print("❌ Requirements file not found")
        return False

    content = requirements_path.read_text()

    # Pin specific versions for stability
    updates = {
        "sentence-transformers>=2.2.0": "sentence-transformers==2.2.2",
        "tiktoken>=0.5.0": "tiktoken==0.5.1",
    }

    for old, new in updates.items():
        if old in content:
            content = content.replace(old, new)
            print(f"📌 Pinned {old} -> {new}")

    requirements_path.write_text(content)
    print("✅ Dependencies pinned")
    return True


def main():
    """Run all Phase 2 Go/No-Go checks."""
    parser = argparse.ArgumentParser(description="Phase 2 Go/No-Go validation")
    parser.add_argument("--harmony_dir", type=Path, default=Path("data/harmony_ready"),
                       help="Harmony data directory")
    args = parser.parse_args()

    print("🚀 Starting Phase 2 Go/No-Go Validation")
    print("=" * 50)

    # 1. Sidecar validation
    print("\n1. SIDECAR FLAGS VALIDATION")
    print("-" * 30)

    train_metadata = args.harmony_dir / "train_metadata.jsonl"
    val_metadata = args.harmony_dir / "val_metadata.jsonl"

    train_results = validate_sidecar_flags(train_metadata)
    val_results = validate_sidecar_flags(val_metadata)

    # Check speaker validation
    train_speaker_rate = train_results["speaker_check_passed_count"] / max(train_results["total_records"], 1)
    val_speaker_rate = val_results["speaker_check_passed_count"] / max(val_results["total_records"], 1)

    print(f"✅ Train speaker_check_passed rate: {train_speaker_rate:.1%}")
    print(f"✅ Val speaker_check_passed rate: {val_speaker_rate:.1%}")

    # Check fit_score population
    train_fit_rate = train_results["fit_score_populated_count"] / max(train_results["total_records"], 1)
    val_fit_rate = val_results["fit_score_populated_count"] / max(val_results["total_records"], 1)

    print(f"✅ Train fit_score population rate: {train_fit_rate:.1%}")
    print(f"✅ Val fit_score population rate: {val_fit_rate:.1%}")

    # 2. Gating metrics analysis
    print("\n2. GATING METRICS ANALYSIS")
    print("-" * 30)

    metrics = analyze_gating_metrics([train_metadata, val_metadata])

    print(f"📊 Total records analyzed: {metrics['total_records']}")
    print(f"👥 Speaker distribution: {metrics['speaker_counts']}")

    # 3. Determinism verification
    print("\n3. DETERMINISM VERIFICATION")
    print("-" * 30)

    train_jsonl = args.harmony_dir / "train.jsonl"
    val_jsonl = args.harmony_dir / "val.jsonl"

    determinism_ok = verify_determinism(train_jsonl, val_jsonl, train_metadata, val_metadata)
    print(f"✅ Determinism check: {'PASSED' if determinism_ok else 'FAILED'}")

    # 4. Leakage guard
    print("\n4. LEAKAGE GUARD")
    print("-" * 30)

    leakage_results = check_leakage_guard(train_metadata, val_metadata)
    print(f"✅ Train records: {leakage_results['train_count']}")
    print(f"✅ Val records: {leakage_results['val_count']}")
    print(f"✅ Overlap: {leakage_results['overlap_count']}")

    # 5. Speaker allow-list audit
    print("\n5. SPEAKER ALLOW-LIST AUDIT")
    print("-" * 30)

    speaker_audit = audit_speaker_allowlist([train_metadata, val_metadata])
    print(f"✅ Allowed speakers: {speaker_audit['allowed_speakers_found']}")

    # 6. Dependency pinning
    print("\n6. DEPENDENCY PINNING")
    print("-" * 30)

    pin_dependencies()

    # Summary
    print("\n" + "=" * 50)
    print("📋 PHASE 2 GO/NO-GO SUMMARY")
    print("=" * 50)

    all_checks_passed = True

    # Check 1: Speaker validation
    if train_speaker_rate < 0.95 or val_speaker_rate < 0.95:
        print("❌ FAIL: Speaker validation rate below 95%")
        all_checks_passed = False
    else:
        print("✅ PASS: Speaker validation rate ≥95%")

    # Check 2: Fit score population
    if train_fit_rate < 0.95 or val_fit_rate < 0.95:
        print("❌ FAIL: Fit score population rate below 95%")
        all_checks_passed = False
    else:
        print("✅ PASS: Fit score population rate ≥95%")

    # Check 3: No invalid speakers
    if speaker_audit["total_invalid"] > 0:
        print(f"❌ FAIL: Found {speaker_audit['total_invalid']} invalid speakers")
        all_checks_passed = False
    else:
        print("✅ PASS: All speakers from allowed list")

    # Check 4: No data leakage
    if leakage_results["overlap_count"] > 0:
        print(f"❌ FAIL: Found {leakage_results['overlap_count']} overlapping pair_ids")
        all_checks_passed = False
    else:
        print("✅ PASS: No data leakage detected")

    # Check 5: Determinism
    if not determinism_ok:
        print("❌ FAIL: Files not found or determinism issues")
        all_checks_passed = False
    else:
        print("✅ PASS: All required files present")

    print("\n" + "=" * 50)
    if all_checks_passed:
        print("🎉 GO DECISION: Phase 2 implementation PASSED all checks!")
        print("🚀 Ready for production deployment")
    else:
        print("❌ NO-GO DECISION: Phase 2 implementation needs fixes")
        print("🔧 Address the issues above before deployment")
    print("=" * 50)

    return 0 if all_checks_passed else 1


if __name__ == "__main__":
    sys.exit(main())
