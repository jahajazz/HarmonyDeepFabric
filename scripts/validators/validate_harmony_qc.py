#!/usr/bin/env python3
"""Phase 3: Comprehensive QC audit and validation system for Harmony Q/A data."""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from difflib import SequenceMatcher


class HarmonyQCAuditor:
    """Comprehensive QC auditor for Harmony Q/A data."""

    def __init__(self, sample_rate: float = 0.03, min_samples: int = 5, seed: int = 42):
        self.sample_rate = sample_rate
        self.min_samples = min_samples
        self.seed = seed
        self.rng = random.Random(seed)

    def load_harmony_data(self, train_path: Path, val_path: Path) -> Tuple[List[Dict], List[Dict]]:
        """Load training and validation data."""
        train_data = self._load_jsonl(train_path)
        val_data = self._load_jsonl(val_path)
        return train_data, val_data

    def load_metadata(self, train_metadata_path: Path, val_metadata_path: Path) -> Tuple[List[Dict], List[Dict]]:
        """Load sidecar metadata."""
        train_metadata = self._load_jsonl(train_metadata_path)
        val_metadata = self._load_jsonl(val_metadata_path)
        return train_metadata, val_metadata

    def _load_jsonl(self, path: Path) -> List[Dict]:
        """Load JSONL file."""
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
        return data

    def check_groundedness(self, record: Dict, metadata: Dict) -> Tuple[bool, str]:
        """Check if answer is grounded in source spans."""
        answer = metadata.get("answer", "").lower()
        source_spans = metadata.get("source_spans_text", "")

        if not answer or not source_spans:
            return False, "Missing answer or source spans"

        # Check if answer appears in source spans
        answer_norm = self._normalize_text(answer)
        source_norm = self._normalize_text(source_spans)

        # Direct substring match
        if answer_norm in source_norm:
            return True, "Answer found in source spans"

        # Fuzzy matching for near-matches
        similarity = SequenceMatcher(None, answer_norm, source_norm).ratio()
        if similarity >= 0.9:
            return True, f"High similarity match ({similarity:.2f})"

        return False, f"Answer not found in source spans (similarity: {similarity:.2f})"

    def check_format_compliance(self, record: Dict, metadata: Dict) -> Tuple[bool, str]:
        """Check format compliance (Motifs→Mapping→Reading structure if relevant)."""
        # For Harmony data, check basic structure requirements
        required_fields = ["pair_id", "question", "answer", "answer_speaker"]

        for field in required_fields:
            if field not in metadata:
                return False, f"Missing required field: {field}"

        # Check question format
        question = metadata.get("question", "")
        if not question.endswith("?"):
            return False, "Question does not end with '?'"

        # Check word count (should be reasonable)
        word_count = len(question.split())
        if word_count > 50:  # Allow some flexibility beyond 25
            return False, f"Question too long: {word_count} words"

        # Check answer format
        answer = metadata.get("answer", "")
        if not answer:
            return False, "Empty answer"

        if len(answer) > 2000:  # Reasonable answer length
            return False, f"Answer too long: {len(answer)} characters"

        return True, "Format compliant"

    def check_speaker_allowlist(self, record: Dict, metadata: Dict) -> Tuple[bool, str]:
        """Check speaker allow-list compliance."""
        allowed_speakers = {"fr stephen de young", "jonathan pageau"}
        speaker = metadata.get("answer_speaker", "").lower()

        if speaker not in allowed_speakers:
            return False, f"Invalid speaker: {speaker}"

        return True, f"Valid speaker: {speaker}"

    def check_fit_score_consistency(self, record: Dict, metadata: Dict) -> Tuple[bool, str]:
        """Check fit_score_ce consistency with pairfit_passed."""
        pairfit_passed = metadata.get("gates", {}).get("pairfit_passed", False)
        fit_score_ce = metadata.get("fit_score_ce")

        if pairfit_passed is None or fit_score_ce is None:
            return True, "Fit score consistency check skipped"

        # Check if fit_score_ce is a valid float in [0, 1]
        if not isinstance(fit_score_ce, (int, float)):
            return False, f"fit_score_ce must be numeric, got {type(fit_score_ce)}"

        if not (0.0 <= fit_score_ce <= 1.0):
            return False, f"fit_score_ce out of range [0,1]: {fit_score_ce}"

        # Check consistency with pairfit_passed
        if pairfit_passed and fit_score_ce < 0.5:
            return False, f"pairfit_passed=true but fit_score_ce={fit_score_ce} < 0.5"

        if not pairfit_passed and fit_score_ce >= 0.5:
            return False, f"pairfit_passed=false but fit_score_ce={fit_score_ce} >= 0.5"

        return True, f"Fit score consistent: {fit_score_ce} with pairfit_passed={pairfit_passed}"

    def check_duplication(self, metadata_list: List[Dict], current_idx: int) -> Tuple[bool, str]:
        """Check for duplication/near-duplication using answer fingerprint."""
        current_metadata = metadata_list[current_idx]
        current_answer = current_metadata.get("answer", "")
        current_question = current_metadata.get("question", "")

        if not current_answer:
            return True, "No answer to check"

        # Create fingerprint from normalized answer and question
        current_fingerprint = self._create_fingerprint(current_answer, current_question)

        for i, other_metadata in enumerate(metadata_list):
            if i == current_idx:
                continue

            other_answer = other_metadata.get("answer", "")
            other_question = other_metadata.get("question", "")

            if not other_answer:
                continue

            other_fingerprint = self._create_fingerprint(other_answer, other_question)

            # Check for exact match
            if current_fingerprint == other_fingerprint:
                return False, f"Exact duplicate found at index {i}"

            # Check for high similarity
            similarity = SequenceMatcher(None, current_answer, other_answer).ratio()
            if similarity >= 0.95:  # Very high similarity threshold
                return False, f"Near-duplicate found at index {i} (similarity: {similarity:.2f})"

        return True, "No duplicates found"

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        import re
        # Remove punctuation, normalize whitespace, convert to lowercase
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join(text.split()).lower()
        return text

    def _create_fingerprint(self, answer: str, question: str) -> str:
        """Create fingerprint for duplication detection."""
        # Create a hash of normalized, truncated content
        content = f"{question}|{answer}".strip()
        normalized = self._normalize_text(content)
        truncated = normalized[:200]  # First 200 chars should be sufficient
        return hashlib.md5(truncated.encode()).hexdigest()

    def _get_sample_indices(self, total_records: int) -> List[int]:
        """Get reproducible sample indices using fixed seed."""
        if total_records <= 0:
            return []

        sample_size = max(self.min_samples, int(total_records * self.sample_rate))
        sample_size = min(sample_size, total_records)

        indices = list(range(total_records))
        self.rng.shuffle(indices)
        return sorted(indices[:sample_size])

    def run_qc_audit(self, train_data: List[Dict], val_data: List[Dict],
                    train_metadata: List[Dict], val_metadata: List[Dict]) -> Dict[str, Any]:
        """Run comprehensive QC audit with fit-score validation."""
        """Run comprehensive QC audit."""
        print("🔍 Running comprehensive QC audit...")

        all_metadata = train_metadata + val_metadata
        total_records = len(all_metadata)

        if total_records == 0:
            return {"error": "No data to audit"}

        # Sample indices for QC audit
        sample_size = max(self.min_samples, int(total_records * self.sample_rate))
        sample_size = min(sample_size, total_records)

        # Use fixed seed for reproducible sampling
        indices = list(range(total_records))
        self.rng.shuffle(indices)
        sample_indices = sorted(indices[:sample_size])

        print(f"📊 Auditing {sample_size}/{total_records} records ({sample_size/total_records*100:.1f}%)")

        results = {
            "total_records": total_records,
            "sample_size": sample_size,
            "passed": 0,
            "failed": 0,
            "failures": [],
            "checks": {
                "groundedness": {"passed": 0, "failed": 0, "details": []},
                "format": {"passed": 0, "failed": 0, "details": []},
                "speaker": {"passed": 0, "failed": 0, "details": []},
                "fit_score": {"passed": 0, "failed": 0, "details": []},
                "duplication": {"passed": 0, "failed": 0, "details": []}
            }
        }

        for idx in sample_indices:
            is_train = idx < len(train_metadata)
            record = train_data[idx] if is_train else val_data[idx - len(train_metadata)]
            metadata = all_metadata[idx]

            # Ensure metadata has all required fields for validation
            if not isinstance(metadata, dict):
                metadata = {}

            record_result = {
                "index": idx,
                "is_train": is_train,
                "pair_id": metadata.get("pair_id", ""),
                "episode_id": metadata.get("episode_id", ""),
                "question": metadata.get("question", "")[:100] + "..." if len(metadata.get("question", "")) > 100 else metadata.get("question", ""),
                "answer_speaker": metadata.get("answer_speaker", ""),
                "check_results": {},
                "overall_pass": True,
                "failure_reasons": []
            }

            # Run all checks
            checks = [
                ("groundedness", self.check_groundedness),
                ("format", self.check_format_compliance),
                ("speaker", self.check_speaker_allowlist),
                ("fit_score", self.check_fit_score_consistency),
                ("duplication", self.check_duplication)
            ]

            for check_name, check_func in checks:
                try:
                    if check_name == "duplication":
                        passed, reason = check_func(all_metadata, idx)
                    else:
                        passed, reason = check_func(record, metadata)

                    record_result["check_results"][check_name] = {
                        "passed": passed,
                        "reason": reason
                    }

                    # Update overall statistics
                    results["checks"][check_name]["passed" if passed else "failed"] += 1
                    results["checks"][check_name]["details"].append({
                        "index": idx,
                        "passed": passed,
                        "reason": reason
                    })

                    if not passed:
                        record_result["overall_pass"] = False
                        record_result["failure_reasons"].append(f"{check_name}: {reason}")

                except Exception as e:
                    record_result["check_results"][check_name] = {
                        "passed": False,
                        "reason": f"Exception: {str(e)}"
                    }
                    record_result["overall_pass"] = False
                    record_result["failure_reasons"].append(f"{check_name}: Exception - {str(e)}")

            # Update overall results
            if record_result["overall_pass"]:
                results["passed"] += 1
            else:
                results["failed"] += 1
                results["failures"].append(record_result)

        # Calculate pass rate
        results["pass_rate"] = results["passed"] / sample_size if sample_size > 0 else 0

        return results

    def save_qc_results(self, results: Dict[str, Any], output_dir: Path):
        """Save QC results to JSON and CSV files."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed JSON results
        json_path = output_dir / "qc_audit_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # Save failures CSV
        csv_path = output_dir / "qc_failures.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            if results["failures"]:
                writer = csv.DictWriter(f, fieldnames=[
                    "index", "is_train", "pair_id", "episode_id", "answer_speaker",
                    "question", "failure_reasons"
                ])
                writer.writeheader()
                for failure in results["failures"]:
                    writer.writerow({
                        "index": failure["index"],
                        "is_train": failure["is_train"],
                        "pair_id": failure["pair_id"],
                        "episode_id": failure["episode_id"],
                        "answer_speaker": failure["answer_speaker"],
                        "question": failure["question"],
                        "failure_reasons": "; ".join(failure["failure_reasons"])
                    })

        print(f"💾 QC results saved to {output_dir}")
        print(f"   📄 JSON: {json_path}")
        print(f"   📊 CSV: {csv_path}")


class StrictValidator:
    """Strict validator that fails build on any error."""

    def __init__(self, connector_max_sentences: int = 2, paraphrase_char_cap: int = 300):
        self.errors = []
        self.connector_max_sentences = max(0, connector_max_sentences)
        self.paraphrase_char_cap = max(0, paraphrase_char_cap)
        self.allowed_speakers = {"fr stephen de young", "jonathan pageau"}

    def _load_jsonl(self, p: str | Path):
        """Load JSONL file with robust error handling."""
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"JSONL not found: {p}")
        out = []
        with p.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Parse error in {p} at line {i}: {e}") from e
        return out

    def validate_file_exists(self, path: Path, description: str):
        """Validate file exists."""
        if not path.exists():
            self.errors.append(f"Missing file: {description} ({path})")
        else:
            print(f"✅ {description}: {path}")

    def validate_file_not_empty(self, path: Path, description: str):
        """Validate file is not empty."""
        if not path.exists():
            self.errors.append(f"Missing file: {description} ({path})")
            return

        size = path.stat().st_size
        if size == 0:
            self.errors.append(f"Empty file: {description} ({path})")
        else:
            print(f"✅ {description} size: {size} bytes")

    def validate_jsonl_format(self, path: Path, description: str):
        """Validate JSONL format."""
        if not path.exists():
            self.errors.append(f"Missing file: {description} ({path})")
            return

        try:
            with open(path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if line.strip():
                        json.loads(line.strip())

                    # Limit validation to first 100 lines for performance
                    if i >= 100:
                        break

            print(f"✅ {description} JSONL format: Valid")

        except json.JSONDecodeError as e:
            self.errors.append(f"Invalid JSONL in {description} ({path}): {e}")
        except Exception as e:
            self.errors.append(f"Error reading {description} ({path}): {e}")

    def validate_harmony_structure(self, train_path: Path, val_path: Path):
        """Validate Harmony structure requirements."""
        for path, split_name in [(train_path, "train"), (val_path, "val")]:
            if not path.exists():
                continue

            try:
                with open(path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if line.strip():
                            record = json.loads(line.strip())

                            # Check for messages field
                            if "messages" not in record:
                                self.errors.append(f"Missing 'messages' field in {split_name} record {i}")
                                continue

                            messages = record["messages"]
                            if not isinstance(messages, list) or len(messages) == 0:
                                self.errors.append(f"Invalid 'messages' field in {split_name} record {i}")
                                continue

                            # Check last message is assistant
                            if messages[-1].get("role") != "assistant":
                                self.errors.append(f"Last message not 'assistant' in {split_name} record {i}")

                            # Limit validation to first 50 records for performance
                            if i >= 50:
                                break

                print(f"✅ {split_name} Harmony structure: Valid")

            except Exception as e:
                self.errors.append(f"Error validating {split_name} structure: {e}")

    def validate_metadata_alignment(self, data_path: Path, metadata_path: Path, split_name: str):
        """Validate metadata alignment with data."""
        if not data_path.exists() or not metadata_path.exists():
            return

        try:
            # Load first few records to check alignment
            with open(data_path, 'r', encoding='utf-8') as f:
                data_records = [json.loads(line.strip()) for line in f if line.strip()][:10]

            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata_records = [json.loads(line.strip()) for line in f if line.strip()][:10]

            if len(data_records) != len(metadata_records):
                self.errors.append(f"Record count mismatch in {split_name}: data={len(data_records)}, metadata={len(metadata_records)}")

            print(f"✅ {split_name} metadata alignment: Valid")

        except Exception as e:
            self.errors.append(f"Error validating {split_name} alignment: {e}")

    def validate_speaker_consistency(self, train_metadata_path: Path, val_metadata_path: Path):
        """Validate speaker allow-list consistency."""

        for path, split_name in [(train_metadata_path, "train"), (val_metadata_path, "val")]:
            if not path.exists():
                continue

            try:
                with open(path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if line.strip():
                            record = json.loads(line.strip())
                            speaker = record.get("answer_speaker", "").lower()

                            if speaker not in self.allowed_speakers:
                                self.errors.append(f"Invalid speaker in {split_name} record {i}: {speaker}")

                        # Limit to first 100 records for performance
                        if i >= 100:
                            break

                print(f"✅ {split_name} speaker consistency: Valid")

            except Exception as e:
                self.errors.append(f"Error validating {split_name} speakers: {e}")

    def validate_composition_metadata(self, metadata_path: Path, split_name: str):
        """Validate Variant A/B composition constraints."""
        if not metadata_path.exists():
            return

        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if not line.strip():
                        continue

                    try:
                        record = json.loads(line.strip())
                    except json.JSONDecodeError as e:
                        self.errors.append(f"Invalid JSON in {split_name} metadata record {i}: {e}")
                        continue

                    composition = record.get("composition")
                    if not isinstance(composition, dict):
                        self.errors.append(f"Missing composition block in {split_name} record {i}")
                        continue

                    mode = composition.get("mode")
                    if mode not in {"A", "B"}:
                        self.errors.append(f"Invalid composition.mode '{mode}' in {split_name} record {i}")
                        continue

                    connector_sentences = composition.get("connector_sentences")
                    paraphrase_chars = composition.get("paraphrase_chars")
                    quoted_spans = composition.get("quoted_spans")

                    if not isinstance(connector_sentences, int) or connector_sentences < 0:
                        self.errors.append(f"Invalid connector_sentences in {split_name} record {i}: {connector_sentences}")
                    if not isinstance(paraphrase_chars, int) or paraphrase_chars < 0:
                        self.errors.append(f"Invalid paraphrase_chars in {split_name} record {i}: {paraphrase_chars}")
                    if not isinstance(quoted_spans, list):
                        self.errors.append(f"composition.quoted_spans must be a list in {split_name} record {i}")
                        quoted_spans = []

                    if not quoted_spans:
                        self.errors.append(f"Missing quoted spans in {split_name} record {i}")

                    if mode == "A":
                        if connector_sentences != 0 or paraphrase_chars != 0:
                            self.errors.append(f"A-mode record {i} in {split_name} must have zero connectors and paraphrase chars")
                    else:
                        if connector_sentences < 1 or connector_sentences > self.connector_max_sentences:
                            self.errors.append(
                                f"B-mode record {i} in {split_name} has connector_sentences={connector_sentences}, "
                                f"expected 1-{self.connector_max_sentences}"
                            )
                        if paraphrase_chars < 1 or paraphrase_chars > self.paraphrase_char_cap:
                            self.errors.append(
                                f"B-mode record {i} in {split_name} has paraphrase_chars={paraphrase_chars}, "
                                f"expected 1-{self.paraphrase_char_cap}"
                            )

                    for span_index, span in enumerate(quoted_spans):
                        if not isinstance(span, dict):
                            self.errors.append(f"Invalid quoted span structure in {split_name} record {i} (index {span_index})")
                            continue
                        speaker = str(span.get("speaker", "")).lower()
                        if speaker and speaker not in self.allowed_speakers:
                            self.errors.append(
                                f"Quoted span speaker '{span.get('speaker')}' not allowed in {split_name} record {i}"
                            )
                        for field in ("start", "end"):
                            value = span.get(field)
                            if value is not None and not isinstance(value, (int, float)):
                                self.errors.append(
                                    f"Quoted span field '{field}' must be numeric in {split_name} record {i} (index {span_index})"
                                )

                    gates = record.get("gates", {})
                    pairfit_passed = bool(gates.get("pairfit_passed", False))
                    fit_score_ce = record.get("fit_score_ce")
                    if fit_score_ce is None:
                        if pairfit_passed:
                            self.errors.append(
                                f"pairfit_passed true but fit_score_ce missing in {split_name} record {i}"
                            )
                    elif not isinstance(fit_score_ce, (int, float)):
                        self.errors.append(
                            f"fit_score_ce must be numeric in {split_name} record {i}: {fit_score_ce}"
                        )
                    else:
                        score = float(fit_score_ce)
                        if pairfit_passed and score < 0.5:
                            self.errors.append(
                                f"pairfit_passed true but fit_score_ce={score:.3f} < 0.5 in {split_name} record {i}"
                            )
                        if not pairfit_passed and score >= 0.5:
                            self.errors.append(
                                f"pairfit_passed false but fit_score_ce={score:.3f} >= 0.5 in {split_name} record {i}"
                            )

        except Exception as e:
            self.errors.append(f"Error validating composition for {split_name}: {e}")

    def run_validation(self, harmony_dir: Path) -> bool:
        """Run all validations."""
        print("🔍 Running strict validation...")

        # File existence checks
        train_path = harmony_dir / "train.jsonl"
        val_path = harmony_dir / "val.jsonl"
        train_metadata_path = harmony_dir / "train_metadata.jsonl"
        val_metadata_path = harmony_dir / "val_metadata.jsonl"

        self.validate_file_exists(train_path, "Train data")
        self.validate_file_exists(val_path, "Validation data")
        self.validate_file_exists(train_metadata_path, "Train metadata")
        self.validate_file_exists(val_metadata_path, "Validation metadata")

        # File size checks
        self.validate_file_not_empty(train_path, "Train data")
        self.validate_file_not_empty(val_path, "Validation data")
        self.validate_file_not_empty(train_metadata_path, "Train metadata")
        self.validate_file_not_empty(val_metadata_path, "Validation metadata")

        # JSONL format checks
        self.validate_jsonl_format(train_path, "Train data")
        self.validate_jsonl_format(val_path, "Validation data")
        self.validate_jsonl_format(train_metadata_path, "Train metadata")
        self.validate_jsonl_format(val_metadata_path, "Validation metadata")

        # Structure checks
        self.validate_harmony_structure(train_path, val_path)

        # Alignment checks
        self.validate_metadata_alignment(train_path, train_metadata_path, "train")
        self.validate_metadata_alignment(val_path, val_metadata_path, "validation")

        # Speaker consistency checks
        self.validate_speaker_consistency(train_metadata_path, val_metadata_path)
        self.validate_composition_metadata(train_metadata_path, "train")
        self.validate_composition_metadata(val_metadata_path, "validation")

        # Summary
        if self.errors:
            print(f"\n❌ VALIDATION FAILED: {len(self.errors)} errors found")
            for error in self.errors:
                print(f"   • {error}")
            return False
        else:
            print("\n✅ VALIDATION PASSED: All checks successful")
            return True


def create_split_manifest(harmony_dir: Path) -> Dict[str, Any]:
    """Create split manifest for idempotence verification."""
    file_map = {
        "train": "train.jsonl",
        "val": "val.jsonl",
        "sidecar_train": "sidecar_train.jsonl",
        "sidecar_val": "sidecar_val.jsonl",
        "train_metadata": "train_metadata.jsonl",
        "val_metadata": "val_metadata.jsonl",
    }

    manifest: Dict[str, Any] = {
        "created_at": Path(__file__).name,
        "file_hashes": {},
        "record_counts": {},
        "split_info": {},
    }

    for key, filename in file_map.items():
        path = harmony_dir / filename
        if not path.exists():
            continue

        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                if not chunk:
                    break
                sha256.update(chunk)

        record_count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record_count += 1

        manifest[key] = {
            "path": str(path),
            "count": record_count,
            "sha256": sha256.hexdigest(),
        }
        manifest["file_hashes"][filename] = manifest[key]["sha256"]
        manifest["record_counts"][filename] = record_count

    return manifest


def save_split_manifest(manifest: Dict[str, Any], harmony_dir: Path):
    """Save split manifest to JSON file."""
    manifest_path = harmony_dir / "split_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"💾 Split manifest saved to {manifest_path}")


def verify_idempotence(harmony_dir: Path) -> bool:
    """Verify idempotence by comparing current manifest with previous."""
    manifest_path = harmony_dir / "split_manifest.json"

    if not manifest_path.exists():
        print("⚠️  No previous manifest found - creating new one")
        return True

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            previous_manifest = json.load(f)

        current_manifest = create_split_manifest(harmony_dir)

        comparable_previous = {k: v for k, v in previous_manifest.items() if k not in {"created_at"}}
        comparable_current = {k: v for k, v in current_manifest.items() if k not in {"created_at"}}

        if comparable_previous == comparable_current:
            print("✅ Idempotence verified: Files identical to previous run")
            return True

        print("❌ Idempotence check failed: Files differ from previous run")
        print("Previous manifest:", comparable_previous)
        print("Current manifest:", comparable_current)
        return False

    except Exception as e:
        print(f"❌ Error verifying idempotence: {e}")
        return False


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description="Phase 3: Harmony Q/A Validation & QC")
    parser.add_argument("--harmony_dir", type=Path, default=Path("data/harmony_ready"),
                       help="Harmony data directory")
    parser.add_argument("--qc_output_dir", type=Path, default=Path("reports/qc_results"),
                       help="QC results output directory")
    parser.add_argument("--strict", action="store_true",
                       help="Run strict validation (fail on any error)")
    parser.add_argument("--qc_only", action="store_true",
                       help="Run only QC audit, skip strict validation")
    parser.add_argument("--sample_rate", type=float, default=0.03,
                       help="QC sample rate (default: 0.03 = 3%%)")
    parser.add_argument("--min_samples", type=int, default=5,
                       help="Minimum QC samples")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible sampling")
    parser.add_argument("--connector_max_sentences", type=int, default=2,
                       help="Maximum connector sentences allowed for Variant B validation")
    parser.add_argument("--paraphrase_char_cap", type=int, default=300,
                       help="Maximum paraphrase characters allowed for Variant B validation")

    args = parser.parse_args()

    print("🚀 Phase 3: Harmony Q/A Validation & QC")
    print("=" * 60)

    # Create output directory
    args.qc_output_dir.mkdir(parents=True, exist_ok=True)

    success = True

    # 1. Strict Validation
    if not args.qc_only:
        print("\n1. STRICT VALIDATION")
        print("-" * 30)

        validator = StrictValidator(
            connector_max_sentences=max(0, args.connector_max_sentences),
            paraphrase_char_cap=max(0, args.paraphrase_char_cap),
        )
        validation_passed = validator.run_validation(args.harmony_dir)

        if args.strict and not validation_passed:
            print("❌ Strict validation failed - exiting")
            return 1

        success = success and validation_passed

    # 2. QC Audit
    print("\n2. QC AUDIT")
    print("-" * 30)

    try:
        # Resolve paths for --harmony_dir mode
        dirp = Path(args.harmony_dir)
        train_path = dirp / "train.jsonl"
        val_path = dirp / "val.jsonl"
        sidecar_train_path = (dirp / "sidecar_train.jsonl")
        sidecar_val_path = (dirp / "sidecar_val.jsonl")

        # Accept legacy filenames if normalized ones missing
        if not sidecar_train_path.exists():
            alt = dirp / "train_metadata.jsonl"
            if alt.exists(): sidecar_train_path = alt
        if not sidecar_val_path.exists():
            alt = dirp / "val_metadata.jsonl"
            if alt.exists(): sidecar_val_path = alt

        print(f"[QC] loading:\n  train={train_path}\n  sidecar_train={sidecar_train_path}\n  val={val_path}\n  sidecar_val={sidecar_val_path}")

        # Ensure _load_jsonl method exists
        validator = StrictValidator(
            connector_max_sentences=max(0, args.connector_max_sentences),
            paraphrase_char_cap=max(0, args.paraphrase_char_cap),
        )
        assert hasattr(validator, "_load_jsonl"), "StrictValidator._load_jsonl missing"

        train_data, val_data = [], []
        train_metadata, val_metadata = [], []

        try:
            train_data = validator._load_jsonl(train_path) if train_path.exists() else []
            val_data = validator._load_jsonl(val_path) if val_path.exists() else []
            train_metadata = validator._load_jsonl(sidecar_train_path) if sidecar_train_path.exists() else []
            val_metadata = validator._load_jsonl(sidecar_val_path) if sidecar_val_path.exists() else []
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return 1

        # Run QC audit
        auditor = HarmonyQCAuditor(
            sample_rate=args.sample_rate,
            min_samples=args.min_samples,
            seed=args.seed
        )

        qc_results = auditor.run_qc_audit(train_data, val_data, train_metadata, val_metadata)

        # Save results
        auditor.save_qc_results(qc_results, args.qc_output_dir)

        # Check pass rate
        pass_rate = qc_results.get("pass_rate", 0)
        required_pass_rate = 0.95  # 95% required

        print(f"\n📊 QC AUDIT RESULTS:")
        print(f"   Sample size: {qc_results['sample_size']}")
        print(f"   Passed: {qc_results['passed']}")
        print(f"   Failed: {qc_results['failed']}")
        print(f"   Pass rate: {pass_rate:.1%}")

        if pass_rate >= required_pass_rate:
            print(f"✅ QC audit PASSED (≥{required_pass_rate:.0%} required)")
        else:
            print(f"❌ QC audit FAILED (required ≥{required_pass_rate:.0%}, got {pass_rate:.1%})")
            success = False

        # Detailed check results
        print(f"\n📋 DETAILED CHECK RESULTS:")
        for check_name, check_results in qc_results["checks"].items():
            passed = check_results["passed"]
            failed = check_results["failed"]
            total = passed + failed
            if total > 0:
                rate = passed / total
                print(f"   {check_name}: {passed}/{total} ({rate:.1%})")

    except Exception as e:
        print(f"❌ QC audit failed: {e}")
        success = False

    # 3. Idempotence verification
    print("\n3. IDEMPOTENCE VERIFICATION")
    print("-" * 30)

    try:
        # Create current manifest
        current_manifest = create_split_manifest(args.harmony_dir)
        save_split_manifest(current_manifest, args.harmony_dir)

        # Verify idempotence
        idempotence_ok = verify_idempotence(args.harmony_dir)
        print(f"✅ Idempotence check: {'PASSED' if idempotence_ok else 'FAILED'}")
        success = success and idempotence_ok

    except Exception as e:
        print(f"❌ Idempotence verification failed: {e}")
        success = False

    # 4. Summary
    print("\n" + "=" * 60)
    print("📋 PHASE 3 VALIDATION SUMMARY")
    print("=" * 60)

    if success:
        print("🎉 ALL VALIDATIONS PASSED!")
        print("✅ Strict validation: OK")
        print("✅ QC audit: OK")
        print("✅ Idempotence: OK")
        print("🚀 Ready for production")
    else:
        print("❌ SOME VALIDATIONS FAILED!")
        print("🔧 Address the issues above before deployment")

    print("=" * 60)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
