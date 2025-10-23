#!/usr/bin/env python3
"""Unit tests for Phase 3 QC audit and validation system."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import sys
import os

# Add the scripts directory to the path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from validators.validate_harmony_qc import (
    HarmonyQCAuditor,
    StrictValidator,
    create_split_manifest,
    verify_idempotence,
)


class TestPhase3QCValidation(unittest.TestCase):
    """Test Phase 3 QC audit and validation features."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

        # Create sample data
        self.sample_train_data = [
            {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is the meaning of life?"},
                    {"role": "assistant", "content": "The meaning of life is 42."}
                ]
            }
        ]

        self.sample_val_data = [
            {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is the capital of France?"},
                    {"role": "assistant", "content": "The capital of France is Paris."}
                ]
            }
        ]

        self.sample_train_metadata = [
            {
                "pair_id": "TEST_001",
                "episode_id": "EP001",
                "question": "What is the meaning of life?",
                "answer": "The meaning of life is 42.",
                "answer_speaker": "Fr Stephen De Young",
                "source_spans_text": "The meaning of life is 42. This is a profound truth.",
                "gates": {
                    "speaker_check_passed": True,
                    "pairfit_passed": True,
                    "questionify_used": False,
                    "answer_trimmed": False
                }
            }
        ]

        self.sample_val_metadata = [
            {
                "pair_id": "TEST_002",
                "episode_id": "EP001",
                "question": "What is the capital of France?",
                "answer": "The capital of France is Paris.",
                "answer_speaker": "Jonathan Pageau",
                "source_spans_text": "Paris is the capital and most populous city of France.",
                "gates": {
                    "speaker_check_passed": True,
                    "pairfit_passed": True,
                    "questionify_used": False,
                    "answer_trimmed": False
                }
            }
        ]

        # Create temporary files
        self.train_path = self.temp_dir / "train.jsonl"
        self.val_path = self.temp_dir / "val.jsonl"
        self.train_metadata_path = self.temp_dir / "train_metadata.jsonl"
        self.val_metadata_path = self.temp_dir / "val_metadata.jsonl"

        # Write test data to files
        with open(self.train_path, 'w', encoding='utf-8') as f:
            for item in self.sample_train_data:
                f.write(json.dumps(item) + '\n')

        with open(self.val_path, 'w', encoding='utf-8') as f:
            for item in self.sample_val_data:
                f.write(json.dumps(item) + '\n')

        with open(self.train_metadata_path, 'w', encoding='utf-8') as f:
            for item in self.sample_train_metadata:
                f.write(json.dumps(item) + '\n')

        with open(self.val_metadata_path, 'w', encoding='utf-8') as f:
            for item in self.sample_val_metadata:
                f.write(json.dumps(item) + '\n')

    def test_strict_validator_file_checks(self):
        """Test strict validator file existence and size checks."""
        validator = StrictValidator()

        # Test file existence validation
        validator.validate_file_exists(self.train_path, "Test train file")
        self.assertEqual(len(validator.errors), 0)

        # Test non-existent file
        fake_path = self.temp_dir / "nonexistent.jsonl"
        validator.validate_file_exists(fake_path, "Fake file")
        self.assertEqual(len(validator.errors), 1)
        self.assertIn("Missing file", validator.errors[0])

    def test_strict_validator_jsonl_format(self):
        """Test JSONL format validation."""
        validator = StrictValidator()

        # Test valid JSONL
        validator.validate_jsonl_format(self.train_path, "Test JSONL")
        self.assertEqual(len(validator.errors), 0)

        # Test invalid JSONL
        invalid_jsonl = self.temp_dir / "invalid.jsonl"
        with open(invalid_jsonl, 'w', encoding='utf-8') as f:
            f.write("This is not JSON\n")
            f.write('{"invalid": json}\n')  # Missing quotes

        validator.validate_jsonl_format(invalid_jsonl, "Invalid JSONL")
        self.assertGreater(len(validator.errors), 0)

    def test_strict_validator_harmony_structure(self):
        """Test Harmony structure validation."""
        validator = StrictValidator()

        # Test valid structure
        validator.validate_harmony_structure(self.train_path, self.val_path)
        self.assertEqual(len(validator.errors), 0)

        # Test invalid structure (missing messages)
        invalid_structure = self.temp_dir / "invalid_structure.jsonl"
        with open(invalid_structure, 'w', encoding='utf-8') as f:
            f.write('{"not_messages": "invalid"}\n')

        validator.validate_harmony_structure(invalid_structure, invalid_structure)
        self.assertGreater(len(validator.errors), 0)

    def test_qc_auditor_groundedness_check(self):
        """Test groundedness validation."""
        auditor = HarmonyQCAuditor()

        # Test valid groundedness
        passed, reason = auditor.check_groundedness(
            self.sample_train_data[0],
            self.sample_train_metadata[0]
        )
        self.assertTrue(passed)
        self.assertIn("Answer found", reason)

        # Test missing source spans
        metadata_no_spans = self.sample_train_metadata[0].copy()
        metadata_no_spans["source_spans_text"] = ""
        passed, reason = auditor.check_groundedness(
            self.sample_train_data[0],
            metadata_no_spans
        )
        self.assertFalse(passed)
        self.assertIn("Missing", reason)

    def test_qc_auditor_format_compliance(self):
        """Test format compliance validation."""
        auditor = HarmonyQCAuditor()

        # Test valid format
        passed, reason = auditor.check_format_compliance(
            self.sample_train_data[0],
            self.sample_train_metadata[0]
        )
        self.assertTrue(passed)
        self.assertIn("compliant", reason)

        # Test missing required fields
        metadata_missing = {}
        passed, reason = auditor.check_format_compliance(
            self.sample_train_data[0],
            metadata_missing
        )
        self.assertFalse(passed)
        self.assertIn("Missing required field", reason)

        # Test question without ?
        metadata_no_qmark = self.sample_train_metadata[0].copy()
        metadata_no_qmark["question"] = "What is the meaning of life"
        passed, reason = auditor.check_format_compliance(
            self.sample_train_data[0],
            metadata_no_qmark
        )
        self.assertFalse(passed)
        self.assertIn("does not end with", reason)

    def test_qc_auditor_speaker_allowlist(self):
        """Test speaker allow-list validation."""
        auditor = HarmonyQCAuditor()

        # Test valid speakers
        passed, reason = auditor.check_speaker_allowlist(
            self.sample_train_data[0],
            self.sample_train_metadata[0]
        )
        self.assertTrue(passed)
        self.assertIn("Valid speaker", reason)

        # Test invalid speaker
        metadata_invalid_speaker = self.sample_train_metadata[0].copy()
        metadata_invalid_speaker["answer_speaker"] = "Invalid Speaker"
        passed, reason = auditor.check_speaker_allowlist(
            self.sample_train_data[0],
            metadata_invalid_speaker
        )
        self.assertFalse(passed)
        self.assertIn("Invalid speaker", reason)

    def test_qc_auditor_duplication_detection(self):
        """Test duplication detection."""
        auditor = HarmonyQCAuditor()

        # Create metadata with duplicates
        duplicate_metadata = [
            self.sample_train_metadata[0],
            self.sample_train_metadata[0].copy()  # Exact duplicate
        ]

        # Test duplicate detection
        passed, reason = auditor.check_duplication(duplicate_metadata, 0)
        self.assertFalse(passed)
        self.assertIn("duplicate", reason)

        # Test no duplication
        unique_metadata = [
            self.sample_train_metadata[0],
            self.sample_val_metadata[0]  # Different content
        ]
        passed, reason = auditor.check_duplication(unique_metadata, 0)
        self.assertTrue(passed)
        self.assertIn("No duplicates", reason)

    def test_qc_audit_sampling_determinism(self):
        """Test QC sampling determinism with fixed seed."""
        auditor1 = HarmonyQCAuditor(seed=42)
        auditor2 = HarmonyQCAuditor(seed=42)

        # Both should produce same sample indices
        indices1 = auditor1._get_sample_indices(1000)
        indices2 = auditor2._get_sample_indices(1000)

        self.assertEqual(indices1, indices2)

        # Different seeds should produce different samples
        auditor3 = HarmonyQCAuditor(seed=123)
        indices3 = auditor3._get_sample_indices(1000)

        self.assertNotEqual(indices1, indices3)

    def test_split_manifest_creation(self):
        """Test split manifest creation and idempotence."""
        manifest = create_split_manifest(self.temp_dir)

        # Check manifest structure
        self.assertIn("file_hashes", manifest)
        self.assertIn("record_counts", manifest)
        self.assertIn("created_at", manifest)

        # Check that our test files are included
        self.assertIn("train.jsonl", manifest["file_hashes"])
        self.assertIn("train_metadata.jsonl", manifest["file_hashes"])

        # Check record counts
        self.assertEqual(manifest["record_counts"]["train.jsonl"], 1)
        self.assertEqual(manifest["record_counts"]["train_metadata.jsonl"], 1)

    def test_idempotence_verification(self):
        """Test idempotence verification."""
        # First run - should create manifest
        is_idempotent = verify_idempotence(self.temp_dir)
        self.assertTrue(is_idempotent)  # Should pass on first run

        # Second run - should verify against previous manifest
        is_idempotent = verify_idempotence(self.temp_dir)
        self.assertTrue(is_idempotent)  # Should pass if files unchanged

        # Modify a file and check that idempotence fails
        with open(self.train_path, 'a', encoding='utf-8') as f:
            f.write('{"messages": [{"role": "user", "content": "Modified"}]}\n')

        is_idempotent = verify_idempotence(self.temp_dir)
        self.assertFalse(is_idempotent)  # Should fail after modification

    def test_qc_audit_comprehensive(self):
        """Test comprehensive QC audit."""
        auditor = HarmonyQCAuditor(sample_rate=1.0, seed=42)  # 100% sample for testing

        results = auditor.run_qc_audit(
            self.sample_train_data,
            self.sample_val_data,
            self.sample_train_metadata,
            self.sample_val_metadata
        )

        # Check results structure
        self.assertIn("total_records", results)
        self.assertIn("sample_size", results)
        self.assertIn("passed", results)
        self.assertIn("failed", results)
        self.assertIn("checks", results)

        # Check that all records were audited
        self.assertEqual(results["total_records"], 2)
        self.assertEqual(results["sample_size"], 2)

        # Check individual check results
        checks = results["checks"]
        self.assertIn("groundedness", checks)
        self.assertIn("format", checks)
        self.assertIn("speaker", checks)
        self.assertIn("duplication", checks)

        # All should pass for our clean test data
        self.assertEqual(results["passed"], 2)
        self.assertEqual(results["failed"], 0)
        self.assertEqual(results["pass_rate"], 1.0)

    def test_qc_results_export(self):
        """Test QC results export to JSON and CSV."""
        auditor = HarmonyQCAuditor()

        # Create test results
        test_results = {
            "total_records": 2,
            "sample_size": 2,
            "passed": 1,
            "failed": 1,
            "failures": [
                {
                    "index": 1,
                    "is_train": False,
                    "pair_id": "TEST_002",
                    "episode_id": "EP001",
                    "question": "Test question?",
                    "answer_speaker": "Jonathan Pageau",
                    "failure_reasons": ["format: Question does not end with '?'"]
                }
            ],
            "checks": {
                "groundedness": {"passed": 2, "failed": 0, "details": []},
                "format": {"passed": 1, "failed": 1, "details": []},
                "speaker": {"passed": 2, "failed": 0, "details": []},
                "duplication": {"passed": 2, "failed": 0, "details": []}
            }
        }

        output_dir = self.temp_dir / "qc_output"
        auditor.save_qc_results(test_results, output_dir)

        # Check that files were created
        json_path = output_dir / "qc_audit_results.json"
        csv_path = output_dir / "qc_failures.csv"

        self.assertTrue(json_path.exists())
        self.assertTrue(csv_path.exists())

        # Verify JSON content
        with open(json_path, 'r', encoding='utf-8') as f:
            loaded_results = json.load(f)
            self.assertEqual(loaded_results["passed"], 1)
            self.assertEqual(loaded_results["failed"], 1)

        # Verify CSV content
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            self.assertGreater(len(lines), 1)  # Header + at least one data row

    def test_text_normalization_and_fingerprinting(self):
        """Test text normalization and fingerprinting for duplication detection."""
        auditor = HarmonyQCAuditor()

        # Test normalization
        text1 = "Hello, World! This is a test."
        text2 = "hello world this is a test"

        norm1 = auditor._normalize_text(text1)
        norm2 = auditor._normalize_text(text2)

        self.assertEqual(norm1, norm2)

        # Test fingerprinting
        fp1 = auditor._create_fingerprint("Answer 1", "Question 1")
        fp2 = auditor._create_fingerprint("Answer 1", "Question 1")
        fp3 = auditor._create_fingerprint("Answer 2", "Question 2")

        self.assertEqual(fp1, fp2)  # Same content should produce same fingerprint
        self.assertNotEqual(fp1, fp3)  # Different content should produce different fingerprint

    def test_edge_cases_and_error_handling(self):
        """Test edge cases and error handling."""
        auditor = HarmonyQCAuditor()

        # Test empty metadata
        empty_metadata = {}
        passed, reason = auditor.check_groundedness({}, empty_metadata)
        self.assertFalse(passed)
        self.assertIn("Missing", reason)

        # Test missing source spans
        metadata_no_spans = {"answer": "Some answer"}
        passed, reason = auditor.check_groundedness({}, metadata_no_spans)
        self.assertFalse(passed)
        self.assertIn("Missing", reason)

        # Test empty answer
        metadata_empty_answer = {"answer": "", "source_spans_text": "Some source"}
        passed, reason = auditor.check_groundedness({}, metadata_empty_answer)
        self.assertFalse(passed)
        self.assertIn("Missing", reason)

    def test_determinism_requirements(self):
        """Test that all random operations use fixed seeds for determinism."""
        # Test multiple auditors with same seed produce same results
        auditor1 = HarmonyQCAuditor(seed=123)
        auditor2 = HarmonyQCAuditor(seed=123)

        # Both should produce identical sampling
        indices1 = list(range(100))
        indices2 = list(range(100))

        auditor1.rng.shuffle(indices1)
        auditor2.rng.shuffle(indices2)

        self.assertEqual(indices1, indices2)

    def test_qc_audit_pass_rate_calculation(self):
        """Test QC audit pass rate calculation."""
        auditor = HarmonyQCAuditor(sample_rate=1.0, seed=42)

        # Create test data with known pass/fail ratio
        test_train_data = self.sample_train_data * 2  # 2 records
        test_val_data = self.sample_val_data * 2     # 2 records
        test_train_metadata = self.sample_train_metadata * 2  # 2 records
        test_val_metadata = self.sample_val_metadata * 2      # 2 records

        # Make one record fail format check
        test_train_metadata[0]["question"] = "Question without question mark"

        results = auditor.run_qc_audit(
            test_train_data,
            test_val_data,
            test_train_metadata,
            test_val_metadata
        )

        # Should have 4 total records, sample all of them
        self.assertEqual(results["total_records"], 4)
        self.assertEqual(results["sample_size"], 4)

        # Should have 3 passed, 1 failed (75% pass rate)
        self.assertEqual(results["passed"], 3)
        self.assertEqual(results["failed"], 1)
        self.assertEqual(results["pass_rate"], 0.75)


if __name__ == "__main__":
    unittest.main()
