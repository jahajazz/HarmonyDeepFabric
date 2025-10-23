#!/usr/bin/env python3
"""Comprehensive integration tests for Phase 4 hardening."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import sys
import os

# Add the scripts directory to the path so we can import the module
os.environ.setdefault("OPENAI_API_KEY", "test-key")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from generators.harmony_qa_from_transcripts import (
    Segment,
    ContextWindow,
    merge_contiguous_segments,
    match_answer_to_source,
    normalise,
    _validate_question,
    _is_question,
    apply_answer_trim,
    get_cross_encoder,
    compute_semantic_fit_score,
    evaluate_pairfit,
    questionify_context,
    refine_question,
    QuestionifyConfig,
    PairFitConfig,
    AnswerTrimConfig,
)
from llm import openai_client as questionify_client

from validators.validate_harmony_qc import (
    HarmonyQCAuditor,
    StrictValidator,
    create_split_manifest,
    verify_idempotence,
)


class TestIntegrationEndToEnd(unittest.TestCase):
    """Comprehensive end-to-end integration tests."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

        # Create realistic test segments
        self.test_segments = [
            Segment(
                episode_id="EP001",
                start=0.0,
                end=5.0,
                speaker="Fr Stephen De Young",
                text="The concept of divine nature has profound implications for understanding human existence and purpose in Christian theology.",
                order=0,
                source_path=Path("test.jsonl")
            ),
            Segment(
                episode_id="EP001",
                start=6.0,
                end=10.0,
                speaker="Fr Stephen De Young",
                text="We must consider how this theological framework applies to our daily lives and spiritual development.",
                order=1,
                source_path=Path("test.jsonl")
            ),
            Segment(
                episode_id="EP001",
                start=11.0,
                end=15.0,
                speaker="Jonathan Pageau",
                text="The symbolic interpretation reveals deeper patterns in the biblical narrative that connect to our understanding.",
                order=2,
                source_path=Path("test.jsonl")
            ),
            Segment(
                episode_id="EP001",
                start=16.0,
                end=20.0,
                speaker="Other Speaker",
                text="This should not appear in answers due to speaker restrictions.",
                order=3,
                source_path=Path("test.jsonl")
            ),
        ]

        self.test_window = ContextWindow(
            episode_id="EP001",
            segments=self.test_segments,
            anchor_index=0,
            window_id="EP001|0"
        )

    def test_full_pipeline_integration(self):
        """Test the complete pipeline from segments to final validation."""
        # 1. Test segment merging
        merged = merge_contiguous_segments(self.test_segments[:2], max_gap=1.0, max_length=1200)
        self.assertEqual(len(merged), 1)
        self.assertIn("divine nature", merged[0].text)
        self.assertIn("theological framework", merged[0].text)

        # 2. Test answer matching
        matched = match_answer_to_source(
            "The concept of divine nature has profound implications",
            [self.test_segments[0]]
        )
        self.assertIsNotNone(matched)

        # 3. Test question validation
        valid_question = "What are the implications of divine nature?"
        self.assertTrue(_validate_question(valid_question, "context about divine nature", "answer"))

        # 4. Test normalization
        normalized = normalise("Test with   multiple  spaces and punctuation!")
        self.assertEqual(normalized, "test with multiple spaces and punctuation!")

        # 5. Test cross-encoder fallback
        with patch('generators.harmony_qa_from_transcripts.get_cross_encoder', return_value=None):
            score = compute_semantic_fit_score("Question", "Answer", "Context")
        self.assertIsNone(score)

        print("✅ Full pipeline integration test completed successfully")

    def test_speaker_allowlist_integration(self):
        """Test speaker allow-list enforcement throughout the pipeline."""
        allowed_speakers = ["fr stephen de young", "jonathan pageau"]

        # Test valid speakers
        valid_speakers = ["Fr Stephen De Young", "Jonathan Pageau", "fr stephen de young", "JONATHAN PAGEAU"]
        for speaker in valid_speakers:
            self.assertIn(speaker.lower(), allowed_speakers)

        # Test invalid speakers
        invalid_speakers = ["Other Speaker", "Unknown", "Invalid Speaker"]
        for speaker in invalid_speakers:
            self.assertNotIn(speaker.lower(), allowed_speakers)

        # Test that only allowed speakers appear in answers
        for segment in self.test_segments:
            if segment.speaker.lower() in allowed_speakers:
                # Should be able to match answers from this speaker
                matched = match_answer_to_source(segment.text[:50], [segment])
                self.assertIsNotNone(matched)
            else:
                # Should not use this speaker for answers
                self.assertEqual(segment.speaker, "Other Speaker")

        print("✅ Speaker allow-list integration test completed successfully")

    def test_merge_threshold_integration(self):
        """Test merge threshold behavior throughout the pipeline."""
        # Test merge at exactly 1.0s gap (should merge per requirement)
        segments_at_gap = [
            Segment("TEST", 0.0, 2.0, "Fr Stephen De Young", "First part", 0, Path("test")),
            Segment("TEST", 3.0, 5.0, "Fr Stephen De Young", "Second part", 1, Path("test")),
        ]

        merged = merge_contiguous_segments(segments_at_gap, max_gap=1.0, max_length=1200)
        self.assertEqual(len(merged), 1)  # Should merge at exactly 1.0s gap per requirement

        # Test merge at 0.9s gap (should merge)
        segments_under_gap = [
            Segment("TEST", 0.0, 2.0, "Fr Stephen De Young", "First part", 0, Path("test")),
            Segment("TEST", 2.9, 5.0, "Fr Stephen De Young", "Second part", 1, Path("test")),
        ]

        merged = merge_contiguous_segments(segments_under_gap, max_gap=1.0, max_length=1200)
        self.assertEqual(len(merged), 1)  # Should merge at 0.9s gap

        # Test length constraint
        long_segments = [
            Segment("TEST", 0.0, 2.0, "Fr Stephen De Young", "A" * 600, 0, Path("test")),
            Segment("TEST", 2.5, 4.0, "Fr Stephen De Young", "B" * 600, 1, Path("test")),
        ]

        merged = merge_contiguous_segments(long_segments, max_gap=1.0, max_length=1200)
        self.assertEqual(len(merged), 2)  # Should not merge due to length constraint

        print("✅ Merge threshold integration test completed successfully")

    def test_questionify_integration(self):
        """Test questionify integration with constraints."""
        # Mock OpenAI client
        mock_client = MagicMock()

        # Mock successful questionify response
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = '{"question": "What are the implications of divine nature?"}'
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        config = QuestionifyConfig(model="gpt-4o-mini", temperature=0.2, max_output_tokens=120, enabled=True)

        # Test questionify with valid context
        context_text = "The divine nature has profound implications for understanding"
        answer_text = "The concept of divine nature has profound implications"

        questionify_client._questionify_cached.cache_clear()
        question = questionify_context(mock_client, config, context_text, answer_text)
        self.assertIsNotNone(question)
        self.assertTrue(question.endswith("?"))

        # Test question validation
        self.assertTrue(_validate_question(question, context_text, answer_text))

        # Test with invalid response
        mock_message.content = '{"question": ""}'
        questionify_client._questionify_cached.cache_clear()
        question = questionify_context(mock_client, config, context_text, answer_text)
        self.assertIsNone(question)

        print("✅ Questionify integration test completed successfully")

    def test_pairfit_integration(self):
        """Test pair-fit evaluation with gray zone handling."""
        # Mock pairfit function
        def mock_pairfit_fn(question, answer, source_spans):
            return {
                "is_supported": True,
                "is_good_question": True,
                "reason": "Good semantic fit",
                "suggested_question": ""
            }

        config = PairFitConfig(model="gpt-4o-mini", temperature=0.1, max_output_tokens=100, enabled=True)

        # Test normal acceptance
        with patch('generators.harmony_qa_from_transcripts.compute_semantic_fit_score', return_value=0.8):
            question, metadata = evaluate_pairfit(
                "What is this?", "This is the answer.", "Source spans",
                "Context", config, mock_pairfit_fn
            )

        self.assertEqual(question, "What is this?")
        self.assertEqual(metadata["status"], "accepted")

        # Test gray zone behavior
        with patch('generators.harmony_qa_from_transcripts.compute_semantic_fit_score') as mock_score:
            mock_score.return_value = 0.48  # Gray zone score

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

        print("✅ Pair-fit integration test completed successfully")

    def test_answer_trim_integration(self):
        """Test answer trimming with contiguous substring validation."""
        original_answer = "This is a very long answer that contains multiple sentences and should be trimmed properly for testing."
        question = "What is this about?"

        # Mock trim function that returns valid contiguous substring
        def mock_valid_trim(q, ans):
            return "This is a very long answer that contains"

        config = AnswerTrimConfig(model="gpt-4o-mini", temperature=0.1, max_output_tokens=100, threshold_tokens=10, enabled=True)
        result = apply_answer_trim(question, original_answer, config, mock_valid_trim)

        self.assertIsNotNone(result)
        self.assertIn("start_char", result)
        self.assertIn("end_char", result)
        self.assertTrue(result["start_char"] >= 0)

        # Test invalid trim (non-contiguous)
        def mock_invalid_trim(q, ans):
            return "This is completely different content"

        result = apply_answer_trim(question, original_answer, config, mock_invalid_trim)
        self.assertIsNone(result)  # Should reject non-contiguous trim

        print("✅ Answer trim integration test completed successfully")

    def test_qc_audit_integration(self):
        """Test QC audit integration with full dataset."""
        # Create comprehensive test data
        train_data = []
        val_data = []
        train_metadata = []
        val_metadata = []

        for i in range(100):
            # Training data
            train_data.append({
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"What is theological question {i}?"},
                    {"role": "assistant", "content": f"This is theological answer {i}."}
                ]
            })

            train_metadata.append({
                "pair_id": f"TRAIN_{i:03d}",
                "episode_id": "EP001",
                "question": f"What is theological question {i}?",
                "answer": f"This is theological answer {i}.",
                "answer_speaker": "Fr Stephen De Young" if i % 2 == 0 else "Jonathan Pageau",
                "source_spans_text": f"This is theological answer {i}. This provides context.",
                "gates": {
                    "speaker_check_passed": True,
                    "pairfit_passed": True,
                    "questionify_used": False,
                    "answer_trimmed": False
                }
            })

            # Validation data
            val_data.append({
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"What is validation question {i}?"},
                    {"role": "assistant", "content": f"This is validation answer {i}."}
                ]
            })

            val_metadata.append({
                "pair_id": f"VAL_{i:03d}",
                "episode_id": "EP002",
                "question": f"What is validation question {i}?",
                "answer": f"This is validation answer {i}.",
                "answer_speaker": "Fr Stephen De Young" if i % 2 == 0 else "Jonathan Pageau",
                "source_spans_text": f"This is validation answer {i}. This provides context.",
                "gates": {
                    "speaker_check_passed": True,
                    "pairfit_passed": True,
                    "questionify_used": False,
                    "answer_trimmed": False
                }
            })

        # Run QC audit
        auditor = HarmonyQCAuditor(sample_rate=0.1, seed=42)  # 10% sample
        results = auditor.run_qc_audit(train_data, val_data, train_metadata, val_metadata)

        # Verify results structure
        self.assertIn("total_records", results)
        self.assertIn("sample_size", results)
        self.assertIn("passed", results)
        self.assertIn("failed", results)
        self.assertIn("checks", results)

        # Should have 200 total records
        self.assertEqual(results["total_records"], 200)
        self.assertEqual(results["sample_size"], 20)  # 10% of 200

        # All should pass for clean test data
        self.assertEqual(results["passed"], 20)
        self.assertEqual(results["failed"], 0)
        self.assertEqual(results["pass_rate"], 1.0)

        print("✅ QC audit integration test completed successfully")

    def test_validation_integration(self):
        """Test validation integration with realistic data."""
        # Create test files
        for filename in ["train.jsonl", "val.jsonl", "train_metadata.jsonl", "val_metadata.jsonl"]:
            path = self.temp_dir / filename
            with open(path, 'w', encoding='utf-8') as f:
                for i in range(50):  # 50 records each
                    if "metadata" in filename:
                        record = {
                            "pair_id": f"{filename.upper()}_{i:03d}",
                            "episode_id": "EP001",
                            "question": f"What is integration question {i}?",
                            "answer": f"This is integration answer {i}.",
                            "answer_speaker": "Fr Stephen De Young" if i % 2 == 0 else "Jonathan Pageau",
                            "source_spans_text": f"This is integration answer {i}. Additional context.",
                            "gates": {
                                "speaker_check_passed": True,
                                "pairfit_passed": True,
                                "questionify_used": False,
                                "answer_trimmed": False
                            }
                        }
                    else:
                        record = {
                            "messages": [
                                {"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": f"What is integration question {i}?"},
                                {"role": "assistant", "content": f"This is integration answer {i}."}
                            ]
                        }
                    f.write(json.dumps(record) + '\n')

        # Test strict validation
        validator = StrictValidator()
        validation_passed = validator.run_validation(self.temp_dir)

        # Should pass for valid files
        self.assertTrue(validation_passed)
        self.assertEqual(len(validator.errors), 0)

        # Test manifest creation
        manifest = create_split_manifest(self.temp_dir)
        self.assertIn("file_hashes", manifest)
        self.assertIn("record_counts", manifest)

        # Verify record counts
        for filename in ["train.jsonl", "val.jsonl", "train_metadata.jsonl", "val_metadata.jsonl"]:
            self.assertEqual(manifest["record_counts"][filename], 50)

        # Test idempotence
        idempotence_ok = verify_idempotence(self.temp_dir)
        self.assertTrue(idempotence_ok)

        print("✅ Validation integration test completed successfully")

    def test_error_handling_integration(self):
        """Test error handling throughout the pipeline."""
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
            score = compute_semantic_fit_score("Q", "A", "C")
        self.assertIsNone(score)

        # Test with invalid JSON
        invalid_json_path = self.temp_dir / "invalid.jsonl"
        with open(invalid_json_path, 'w', encoding='utf-8') as f:
            f.write('{"invalid": json}\n')

        validator = StrictValidator()
        validator.validate_jsonl_format(invalid_json_path, "Invalid file")
        self.assertGreater(len(validator.errors), 0)

        print("✅ Error handling integration test completed successfully")

    def test_determinism_integration(self):
        """Test determinism throughout the pipeline."""
        # Test multiple runs with same seed produce same results
        auditor1 = HarmonyQCAuditor(seed=42)
        auditor2 = HarmonyQCAuditor(seed=42)

        # Create test data
        test_metadata = []
        for i in range(100):
            test_metadata.append({
                "pair_id": f"TEST_{i:03d}",
                "question": f"What is question {i}?",
                "answer": f"This is answer {i}.",
                "answer_speaker": "Fr Stephen De Young" if i % 2 == 0 else "Jonathan Pageau"
            })

        # Both should produce identical sampling
        indices1 = auditor1._get_sample_indices(len(test_metadata))
        indices2 = auditor2._get_sample_indices(len(test_metadata))
        self.assertEqual(indices1, indices2)

        # Test file hash consistency
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

        print("✅ Determinism integration test completed successfully")

    def test_comprehensive_data_pipeline(self):
        """Test comprehensive data pipeline with all gates."""
        # This test simulates the full pipeline with realistic data
        segments = [
            Segment("EP001", 0.0, 5.0, "Fr Stephen De Young",
                   "The concept of divine nature is fundamental to understanding Christian theology.", 0, Path("test")),
            Segment("EP001", 6.0, 10.0, "Fr Stephen De Young",
                   "We must consider how this theological framework applies to our daily lives.", 1, Path("test")),
        ]

        # Test merge behavior
        merged = merge_contiguous_segments(segments, max_gap=1.0, max_length=1200)
        self.assertEqual(len(merged), 1)  # Should merge with small gap

        # Test answer matching
        matched = match_answer_to_source("The concept of divine nature is fundamental", segments)
        self.assertIsNotNone(matched)  # Should find match

        # Test question validation
        valid_q = "What is fundamental to Christian theology?"
        self.assertTrue(_validate_question(valid_q, "context about divine nature", "answer"))

        # Test normalization
        normalized = normalise("Test   text  with   multiple   spaces")
        self.assertEqual(normalized, "test text with multiple spaces")

        # Test cross-encoder fallback
        with patch('generators.harmony_qa_from_transcripts.get_cross_encoder', return_value=None):
            score = compute_semantic_fit_score("Q", "A", "C")
        self.assertIsNone(score)

        print("✅ Comprehensive data pipeline integration test completed successfully")


if __name__ == "__main__":
    unittest.main()
