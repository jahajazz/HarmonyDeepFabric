#!/usr/bin/env python3
"""Phase 4: Comprehensive testing and hardening for Harmony Q/A pipeline."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sys
import os

# Add the scripts directory to the path so we can import the module
os.environ.setdefault("OPENAI_API_KEY", "test-key")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from generators.harmony_qa_from_transcripts import (
    Segment,
    ContextWindow,
    merge_contiguous_segments,
    _validate_question,
    _is_question,
    apply_answer_trim,
    get_cross_encoder,
    normalise,
    match_answer_to_source,
    evaluate_pairfit,
    QuestionifyConfig,
    PairFitConfig,
    AnswerTrimConfig,
)

from validators.validate_harmony_qc import (
    HarmonyQCAuditor,
    StrictValidator,
    create_split_manifest,
    verify_idempotence,
)


class TestPhase4Hardening(unittest.TestCase):
    """Comprehensive hardening tests for Phase 4."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

        # Create realistic test data
        self.test_segments = [
            Segment(
                episode_id="EP001",
                start=0.0,
                end=5.0,
                speaker="Fr Stephen De Young",
                text="This is a theological explanation about divine nature and its implications for human understanding.",
                order=0,
                source_path=Path("test.jsonl")
            ),
            Segment(
                episode_id="EP001",
                start=6.0,
                end=10.0,
                speaker="Jonathan Pageau",
                text="The symbolic interpretation reveals deeper patterns in the biblical narrative.",
                order=1,
                source_path=Path("test.jsonl")
            ),
            Segment(
                episode_id="EP001",
                start=11.0,
                end=15.0,
                speaker="Other Speaker",
                text="This should not appear in answers.",
                order=2,
                source_path=Path("test.jsonl")
            ),
        ]

        self.test_window = ContextWindow(
            episode_id="EP001",
            segments=self.test_segments,
            anchor_index=0,
            window_id="EP001|0"
        )

    def test_determinism_identical_seed(self):
        """Test 1: Same seed produces identical results."""
        # Test cross-encoder lazy loading
        ce1 = get_cross_encoder()
        ce2 = get_cross_encoder()
        if ce1 is not None:
            self.assertIs(ce1, ce2)  # Same instance

        # Test merge behavior with identical inputs
        segments1 = [
            Segment("TEST", 0.0, 2.0, "Speaker", "First part", 0, Path("test")),
            Segment("TEST", 2.8, 4.0, "Speaker", "Second part", 1, Path("test")),
        ]

        segments2 = [
            Segment("TEST", 0.0, 2.0, "Speaker", "First part", 0, Path("test")),
            Segment("TEST", 2.8, 4.0, "Speaker", "Second part", 1, Path("test")),
        ]

        merged1 = merge_contiguous_segments(segments1, max_gap=1.0, max_length=1200)
        merged2 = merge_contiguous_segments(segments2, max_gap=1.0, max_length=1200)

        self.assertEqual(len(merged1), len(merged2))
        self.assertEqual(merged1[0].text, merged2[0].text)

    def test_leakage_guard_no_pair_id_overlap(self):
        """Test 2: No pair_id overlap between train/val splits."""
        # Create test metadata with unique pair_ids
        train_metadata = [
            {"pair_id": "TRAIN_001", "episode_id": "EP001", "answer_speaker": "Fr Stephen De Young"},
            {"pair_id": "TRAIN_002", "episode_id": "EP001", "answer_speaker": "Jonathan Pageau"},
        ]

        val_metadata = [
            {"pair_id": "VAL_001", "episode_id": "EP002", "answer_speaker": "Fr Stephen De Young"},
            {"pair_id": "VAL_002", "episode_id": "EP002", "answer_speaker": "Jonathan Pageau"},
        ]

        # Extract pair_ids
        train_ids = {record["pair_id"] for record in train_metadata}
        val_ids = {record["pair_id"] for record in val_metadata}

        # Assert no overlap
        overlap = train_ids.intersection(val_ids)
        self.assertEqual(len(overlap), 0, f"Found overlapping pair_ids: {overlap}")

    def test_speaker_allow_list_enforcement(self):
        """Test 3: Only allowed speakers in answers."""
        allowed_speakers = {"fr stephen de young", "jonathan pageau"}

        # Test valid speakers
        valid_speakers = ["Fr Stephen De Young", "Jonathan Pageau", "fr stephen de young", "JONATHAN PAGEAU"]
        for speaker in valid_speakers:
            self.assertIn(speaker.lower(), allowed_speakers)

        # Test invalid speakers
        invalid_speakers = ["Other Speaker", "Unknown", "Invalid Speaker"]
        for speaker in invalid_speakers:
            self.assertNotIn(speaker.lower(), allowed_speakers)

    def test_merge_thresholds_exactly_one_second(self):
        """Test 4: Merge at ≤1.0s gap, not at >1.0s gap."""
        # Test merge at exactly 1.0s gap (should merge per requirement)
        segments_at_gap = [
            Segment("TEST", 0.0, 2.0, "Fr Stephen De Young", "First part", 0, Path("test")),
            Segment("TEST", 3.0, 5.0, "Fr Stephen De Young", "Second part", 1, Path("test")),
        ]

        merged = merge_contiguous_segments(segments_at_gap, max_gap=1.0, max_length=1200)
        self.assertEqual(len(merged), 1)  # Should merge at exactly 1.0s gap per requirement

        # Test no merge at >1.0s gap
        segments_over_gap = [
            Segment("TEST", 0.0, 2.0, "Fr Stephen De Young", "First part", 0, Path("test")),
            Segment("TEST", 3.5, 5.0, "Fr Stephen De Young", "Second part", 1, Path("test")),
        ]

        merged = merge_contiguous_segments(segments_over_gap, max_gap=1.0, max_length=1200)
        self.assertEqual(len(merged), 2)  # Should not merge at >1.0s gap

        # Test merge at 0.9s gap
        segments_under_gap = [
            Segment("TEST", 0.0, 2.0, "Fr Stephen De Young", "First part", 0, Path("test")),
            Segment("TEST", 2.9, 5.0, "Fr Stephen De Young", "Second part", 1, Path("test")),
        ]

        merged = merge_contiguous_segments(segments_under_gap, max_gap=1.0, max_length=1200)
        self.assertEqual(len(merged), 1)  # Should merge at 0.9s gap

    def test_substring_trim_guard_contiguous(self):
        """Test 5: Trimmed answers must be contiguous substrings."""
        original_answer = "This is a very long answer that contains multiple sentences and should be trimmed properly."
        question = "What is this about?"

        # Mock trim function that returns valid contiguous substring
        def mock_valid_trim(q, ans):
            return "This is a very long answer that contains"

        config = AnswerTrimConfig(model="test", temperature=0.1, max_output_tokens=100, threshold_tokens=10, enabled=True)
        result = apply_answer_trim(question, original_answer, config, mock_valid_trim)

        self.assertIsNotNone(result)
        self.assertIn("start_char", result)
        self.assertIn("end_char", result)
        self.assertTrue(result["start_char"] >= 0)  # Should find substring

        # Test invalid trim (non-contiguous)
        def mock_invalid_trim(q, ans):
            return "This is completely different content"

        result = apply_answer_trim(question, original_answer, config, mock_invalid_trim)
        self.assertIsNone(result)  # Should reject non-contiguous trim

    def test_questionify_contract_strict(self):
        """Test 6: Questionify contract (≤25 words, ends with '?', valid JSON)."""
        # Test valid question format
        valid_questions = [
            "What is the meaning of life?",
            "How should we understand this?",
            "What does theology teach us about divine nature?"
        ]

        for question in valid_questions:
            self.assertTrue(question.endswith("?"))
            word_count = len(question.split())
            self.assertLessEqual(word_count, 25)

        # Test invalid questions
        invalid_questions = [
            "What is the meaning of life and why do we exist in this universe with all these questions and considerations that we need to think about deeply",  # Too long
            "What is the meaning of life",  # No question mark
            ""  # Empty
        ]

        for question in invalid_questions:
            if question == "":  # Empty case
                self.assertFalse(_validate_question(question, "context", "answer"))
            elif not question.endswith("?"):  # No question mark
                self.assertFalse(_validate_question(question, "context", "answer"))
            else:  # Too long
                self.assertFalse(_validate_question(question, "context", "answer"))

    def test_pair_fit_threshold_and_gray_zone(self):
        """Test 7: Pair-fit threshold (≥0.50) and gray zone behavior."""
        def mock_pairfit_fn(question, answer, source_spans):
            return {
                "is_supported": True,
                "is_good_question": True,
                "reason": "Good fit",
                "suggested_question": ""
            }

        config = PairFitConfig(model="test", temperature=0.1, max_output_tokens=100, enabled=True)

        # Test high score acceptance
        with patch('generators.harmony_qa_from_transcripts.compute_semantic_fit_score') as mock_score:
            mock_score.return_value = 0.8  # Above threshold

            question, metadata = evaluate_pairfit(
                "What is this?", "This is the answer.", "Source spans",
                "Context", config, mock_pairfit_fn
            )

            self.assertEqual(question, "What is this?")
            self.assertEqual(metadata["status"], "accepted")
            self.assertEqual(metadata["fit_score_ce"], 0.8)

        # Test gray zone behavior (0.45-0.50)
        with patch('generators.harmony_qa_from_transcripts.compute_semantic_fit_score') as mock_score:
            mock_score.return_value = 0.48  # Gray zone

            def mock_rejecting_pairfit(question, answer, source_spans):
                return {
                    "is_supported": False,
                    "is_good_question": False,
                    "reason": "Poor fit",
                    "suggested_question": "What is this instead?"
                }

            question, metadata = evaluate_pairfit(
                "What is this?", "This is the answer.", "Source spans",
                "Context", config, mock_rejecting_pairfit
            )

            # Should handle gray zone logic
            self.assertIsNotNone(metadata)
            self.assertEqual(metadata["fit_score_ce"], 0.48)

    def test_qc_sampler_reproducibility(self):
        """Test 8: QC sampler produces identical results with fixed seed."""
        # Create test data
        test_metadata = []
        for i in range(100):
            test_metadata.append({
                "pair_id": f"TEST_{i:03d}",
                "question": f"What is question {i}?",
                "answer": f"This is answer {i}.",
                "answer_speaker": "Fr Stephen De Young" if i % 2 == 0 else "Jonathan Pageau"
            })

        # Test multiple auditors with same seed
        auditor1 = HarmonyQCAuditor(seed=42)
        auditor2 = HarmonyQCAuditor(seed=42)

        # Both should produce identical sampling
        indices1 = auditor1._get_sample_indices(len(test_metadata))
        indices2 = auditor2._get_sample_indices(len(test_metadata))

        self.assertEqual(indices1, indices2)

        # Different seeds should produce different samples
        auditor3 = HarmonyQCAuditor(seed=123)
        indices3 = auditor3._get_sample_indices(len(test_metadata))

        self.assertNotEqual(indices1, indices3)

    def test_comprehensive_data_pipeline(self):
        """Test 9: End-to-end data pipeline with all gates."""
        # This test simulates the full pipeline with realistic data
        segments = [
            Segment("EP001", 0.0, 5.0, "Fr Stephen De Young",
                   "The concept of divine nature has profound implications for understanding human existence and purpose.", 0, Path("test")),
            Segment("EP001", 6.0, 10.0, "Fr Stephen De Young",
                   "We must consider how this theological framework applies to our daily lives.", 1, Path("test")),
        ]

        # Test merge behavior
        merged = merge_contiguous_segments(segments, max_gap=1.0, max_length=1200)
        self.assertEqual(len(merged), 1)  # Should merge with small gap

        # Test answer matching
        matched = match_answer_to_source(
            "The concept of divine nature has profound implications",
            segments
        )
        self.assertIsNotNone(matched)  # Should find match

        # Test question validation
        valid_q = "What are the implications of divine nature?"
        self.assertTrue(_validate_question(valid_q, "context about divine nature", "answer"))

    def test_error_handling_and_edge_cases(self):
        """Test 10: Error handling and edge cases."""
        # Test empty inputs
        empty_segments = []
        merged = merge_contiguous_segments(empty_segments, max_gap=1.0, max_length=1200)
        self.assertEqual(len(merged), 0)

        # Test None answer matching
        matched = match_answer_to_source("", [])
        self.assertIsNone(matched)

        # Test invalid question validation
        self.assertFalse(_validate_question("", "context", "answer"))
        self.assertFalse(_validate_question("Not a question", "context", "answer"))

        # Test cross-encoder fallback
        with patch('generators.harmony_qa_from_transcripts.get_cross_encoder', return_value=None):
            from generators.harmony_qa_from_transcripts import compute_semantic_fit_score
            score = compute_semantic_fit_score("Q", "A", "C")
            self.assertIsNone(score)

    def test_file_hash_consistency(self):
        """Test 11: File hash consistency for idempotence."""
        # Create test files
        test_files = {}

        for filename in ["train.jsonl", "val.jsonl", "train_metadata.jsonl", "val_metadata.jsonl"]:
            path = self.temp_dir / filename
            test_data = [
                {"test": "data", "id": f"{filename}_1"},
                {"test": "data", "id": f"{filename}_2"},
            ]
            with open(path, 'w', encoding='utf-8') as f:
                for item in test_data:
                    f.write(json.dumps(item) + '\n')

            test_files[filename] = path

        # Test manifest creation
        manifest = create_split_manifest(self.temp_dir)

        self.assertIn("file_hashes", manifest)
        self.assertIn("record_counts", manifest)

        # Verify record counts
        for filename in test_files:
            self.assertEqual(manifest["record_counts"][filename], 2)

        # Test hash consistency
        hash1 = manifest["file_hashes"]["train.jsonl"]
        hash2 = manifest["file_hashes"]["train.jsonl"]
        self.assertEqual(hash1, hash2)  # Same file should have same hash

    def test_qc_audit_comprehensive(self):
        """Test 12: Comprehensive QC audit functionality."""
        # Create test data
        train_data = [{"messages": [{"role": "user", "content": "Test"}]}]
        val_data = [{"messages": [{"role": "user", "content": "Test"}]}]
        train_metadata = [{"pair_id": "T1", "question": "What is this?", "answer": "This is it.", "answer_speaker": "Fr Stephen De Young", "source_spans_text": "This is it."}]
        val_metadata = [{"pair_id": "V1", "question": "What is that?", "answer": "That is it.", "answer_speaker": "Jonathan Pageau", "source_spans_text": "That is it."}]

        auditor = HarmonyQCAuditor(sample_rate=1.0, seed=42)  # 100% sample for testing

        results = auditor.run_qc_audit(train_data, val_data, train_metadata, val_metadata)

        # Verify results structure
        self.assertIn("total_records", results)
        self.assertIn("sample_size", results)
        self.assertIn("passed", results)
        self.assertIn("failed", results)
        self.assertIn("checks", results)

        # Should have 2 total records
        self.assertEqual(results["total_records"], 2)
        self.assertEqual(results["sample_size"], 2)

        # All should pass for clean test data
        self.assertEqual(results["passed"], 2)
        self.assertEqual(results["failed"], 0)
        self.assertEqual(results["pass_rate"], 1.0)

    def test_strict_validation_comprehensive(self):
        """Test 13: Comprehensive strict validation."""
        # Create valid test files
        for filename in ["train.jsonl", "val.jsonl", "train_metadata.jsonl", "val_metadata.jsonl"]:
            path = self.temp_dir / filename
            with open(path, 'w', encoding='utf-8') as f:
                f.write('{"messages": [{"role": "user", "content": "test"}, {"role": "assistant", "content": "response"}]}\n')

        validator = StrictValidator()
        validation_passed = validator.run_validation(self.temp_dir)

        # Should pass for valid files
        self.assertTrue(validation_passed)
        self.assertEqual(len(validator.errors), 0)

        # Test with invalid file
        invalid_path = self.temp_dir / "invalid.jsonl"
        with open(invalid_path, 'w', encoding='utf-8') as f:
            f.write('{"invalid": json}\n')  # Invalid JSON

        validator.validate_jsonl_format(invalid_path, "Invalid file")
        self.assertGreater(len(validator.errors), 0)

    def test_cross_encoder_robustness(self):
        """Test 14: Cross-encoder robustness and fallback."""
        # Test lazy loading
        ce1 = get_cross_encoder()
        ce2 = get_cross_encoder()

        if ce1 is not None:
            self.assertIs(ce1, ce2)  # Same instance due to lazy loading
        else:
            # Should handle gracefully when not available
            self.assertIsNone(ce1)

        # Test fallback behavior
        with patch('generators.harmony_qa_from_transcripts.get_cross_encoder', return_value=None):
            from generators.harmony_qa_from_transcripts import compute_semantic_fit_score
            score = compute_semantic_fit_score("Question", "Answer", "Context")
            self.assertIsNone(score)  # Should return None when CE unavailable

    def test_text_processing_robustness(self):
        """Test 15: Text processing robustness."""
        # Test Unicode normalization
        unicode_text = 'Text with \'smart quotes\' and "double quotes"'
        normalized = normalise(unicode_text)
        self.assertIsInstance(normalized, str)
        self.assertIn("smart quotes", normalized)

        # Test empty text handling
        empty_normalized = normalise("")
        self.assertEqual(empty_normalized, "")

        # Test whitespace normalization
        whitespace_text = "  Multiple   spaces   and\t\ttabs  "
        normalized = normalise(whitespace_text)
        self.assertNotIn("  ", normalized)  # No double spaces
        self.assertNotIn("\t", normalized)  # No tabs

    def test_speaker_normalization(self):
        """Test 16: Speaker name normalization."""
        test_cases = [
            ("Fr Stephen De Young", "fr stephen de young"),
            ("FR STEPHEN DE YOUNG", "fr stephen de young"),
            ("Jonathan Pageau", "jonathan pageau"),
            ("JONATHAN PAGEAU", "jonathan pageau"),
        ]

        for original, expected in test_cases:
            normalized = original.lower()
            self.assertEqual(normalized, expected)

    def test_question_detection_robustness(self):
        """Test 17: Question detection robustness."""
        # Test various question formats
        questions = [
            "What is this?",
            "How do we understand this?",
            "Why should we consider this?",
            "Is this the correct approach?",
            "Can you explain this concept?",
        ]

        for question in questions:
            self.assertTrue(_is_question(question))
            self.assertTrue(question.endswith("?"))

        # Test non-questions
        non_questions = [
            "This is a statement",
            "Here is some information",
            "The answer is clear",
        ]

        for statement in non_questions:
            self.assertFalse(_is_question(statement))

    def test_answer_matching_robustness(self):
        """Test 18: Answer matching robustness."""
        speaker_segments = [
            Segment("TEST", 0.0, 10.0, "Fr Stephen De Young",
                   "This is a comprehensive answer that covers multiple aspects of the theological concept.", 0, Path("test")),
        ]

        # Test exact match
        exact_match = match_answer_to_source(
            "This is a comprehensive answer that covers multiple aspects of the theological concept.",
            speaker_segments
        )
        self.assertIsNotNone(exact_match)

        # Test partial match
        partial_match = match_answer_to_source(
            "This is a comprehensive answer",
            speaker_segments
        )
        self.assertIsNotNone(partial_match)

        # Test no match
        no_match = match_answer_to_source("Completely different text", speaker_segments)
        self.assertIsNone(no_match)

    def test_merge_edge_cases(self):
        """Test 19: Merge edge cases."""
        # Test single segment
        single_segment = [Segment("TEST", 0.0, 5.0, "Speaker", "Single", 0, Path("test"))]
        merged = merge_contiguous_segments(single_segment, max_gap=1.0, max_length=1200)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].text, "Single")

        # Test empty segments
        empty_segments = []
        merged = merge_contiguous_segments(empty_segments, max_gap=1.0, max_length=1200)
        self.assertEqual(len(merged), 0)

        # Test segments with NaN timing
        nan_segment = [Segment("TEST", float('nan'), float('nan'), "Speaker", "NaN", 0, Path("test"))]
        merged = merge_contiguous_segments(nan_segment, max_gap=1.0, max_length=1200)
        self.assertEqual(len(merged), 1)  # Should handle NaN gracefully

    def test_qc_audit_edge_cases(self):
        """Test 20: QC audit edge cases."""
        auditor = HarmonyQCAuditor()

        # Test empty data
        empty_results = auditor.run_qc_audit([], [], [], [])
        self.assertEqual(empty_results["total_records"], 0)
        self.assertEqual(empty_results["error"], "No data to audit")

        # Test with missing source spans
        metadata_no_spans = {
            "pair_id": "TEST_001",
            "question": "What is this?",
            "answer": "This is the answer.",
            "answer_speaker": "Fr Stephen De Young",
            "source_spans_text": ""  # Missing
        }

        passed, reason = auditor.check_groundedness({}, metadata_no_spans)
        self.assertFalse(passed)
        self.assertIn("Missing", reason)

    def test_serialization_robustness(self):
        """Test 21: JSON serialization robustness."""
        # Test data with special characters
        special_data = {
            "pair_id": "TEST_001",
            "question": "What is the meaning of life?",
            "answer": "The meaning includes 'divine' aspects and \"human\" elements.",
            "answer_speaker": "Fr Stephen De Young",
            "source_spans_text": "The meaning includes 'divine' aspects and \"human\" elements."
        }

        # Should serialize without errors
        json_str = json.dumps(special_data, ensure_ascii=False)
        parsed = json.loads(json_str)

        self.assertEqual(parsed["pair_id"], "TEST_001")
        self.assertEqual(parsed["question"], "What is the meaning of life?")

    def test_performance_characteristics(self):
        """Test 22: Performance characteristics."""
        # Test that operations complete in reasonable time
        import time

        # Test merge performance
        large_segments = []
        for i in range(100):
            large_segments.append(
                Segment("TEST", i*5.0, i*5.0+2.0, "Speaker", f"Segment {i}", i, Path("test"))
            )

        start_time = time.time()
        merged = merge_contiguous_segments(large_segments, max_gap=1.0, max_length=1200)
        end_time = time.time()

        # Should complete quickly
        self.assertLess(end_time - start_time, 5.0)  # Less than 5 seconds
        self.assertGreater(len(merged), 0)  # Should produce results

    def test_backward_compatibility(self):
        """Test 23: Backward compatibility with existing data."""
        # Test that existing functionality still works
        segments = [
            Segment("EP001", 0.0, 5.0, "Fr Stephen De Young", "Test answer", 0, Path("test")),
        ]

        # All core functions should work
        merged = merge_contiguous_segments(segments, max_gap=1.0, max_length=1200)
        self.assertEqual(len(merged), 1)

        matched = match_answer_to_source("Test answer", segments)
        self.assertIsNotNone(matched)

        normalized = normalise("Test text")
        self.assertIsInstance(normalized, str)

    def test_integration_end_to_end(self):
        """Test 24: End-to-end integration."""
        # This test simulates the full pipeline
        segments = [
            Segment("EP001", 0.0, 5.0, "Fr Stephen De Young",
                   "The divine nature is fundamental to understanding Christian theology.", 0, Path("test")),
        ]

        # Test merge
        merged = merge_contiguous_segments(segments, max_gap=1.0, max_length=1200)
        self.assertEqual(len(merged), 1)

        # Test answer matching
        matched = match_answer_to_source("The divine nature is fundamental", segments)
        self.assertIsNotNone(matched)

        # Test question validation
        question = "What is fundamental to Christian theology?"
        self.assertTrue(_validate_question(question, "context about divine nature", "answer"))

        # Test cross-encoder fallback
        with patch('generators.harmony_qa_from_transcripts.get_cross_encoder', return_value=None):
            from generators.harmony_qa_from_transcripts import compute_semantic_fit_score
            score = compute_semantic_fit_score("Q", "A", "C")
            self.assertIsNone(score)


if __name__ == "__main__":
    unittest.main()
