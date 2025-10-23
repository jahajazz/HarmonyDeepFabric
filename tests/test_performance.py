#!/usr/bin/env python3
"""Performance and load tests for Phase 4 hardening."""

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import sys
import os

# Add the scripts directory to the path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from generators.harmony_qa_from_transcripts import (
    Segment,
    ContextWindow,
    merge_contiguous_segments,
    match_answer_to_source,
    normalise,
    _validate_question,
    compute_semantic_fit_score,
    get_cross_encoder,
)

from validators.validate_harmony_qc import (
    HarmonyQCAuditor,
    StrictValidator,
    create_split_manifest,
)


class TestPerformanceAndLoad(unittest.TestCase):
    """Performance and load tests for Phase 4."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def test_merge_performance_large_dataset(self):
        """Test merge performance with large dataset."""
        # Create large dataset of segments
        large_segments = []
        for i in range(1000):
            large_segments.append(
                Segment(
                    episode_id="PERF_TEST",
                    start=i * 5.0,
                    end=i * 5.0 + 2.0,
                    speaker="Fr Stephen De Young",
                    text=f"This is segment number {i} with some content for testing performance.",
                    order=i,
                    source_path=Path("test.jsonl")
                )
            )

        # Test merge performance
        start_time = time.time()
        merged = merge_contiguous_segments(large_segments, max_gap=1.0, max_length=1200)
        end_time = time.time()

        # Should complete in reasonable time (< 5 seconds for 1000 segments)
        self.assertLess(end_time - start_time, 5.0)
        self.assertGreater(len(merged), 0)

        print(f"✅ Merged {len(large_segments)} segments in {end_time - start_time:.2f}s")

    def test_text_processing_performance(self):
        """Test text processing performance with large texts."""
        # Create large text for processing
        large_text = "This is a test sentence. " * 1000  # ~6000 words

        # Test normalization performance
        start_time = time.time()
        normalized = normalise(large_text)
        end_time = time.time()

        # Should complete quickly
        self.assertLess(end_time - start_time, 1.0)
        self.assertIsInstance(normalized, str)

        print(f"✅ Normalized {len(large_text)} chars in {end_time - start_time:.3f}s")

    def test_answer_matching_performance(self):
        """Test answer matching performance with large speaker segments."""
        # Create large set of speaker segments
        speaker_segments = []
        for i in range(100):
            speaker_segments.append(
                Segment(
                    episode_id="PERF_TEST",
                    start=i * 10.0,
                    end=i * 10.0 + 5.0,
                    speaker="Fr Stephen De Young",
                    text=f"This is a comprehensive answer segment number {i} that contains detailed theological content for testing performance.",
                    order=i,
                    source_path=Path("test.jsonl")
                )
            )

        # Test answer matching performance
        test_answer = "This is a comprehensive answer segment number 50 that contains detailed theological content"

        start_time = time.time()
        matched = match_answer_to_source(test_answer, speaker_segments)
        end_time = time.time()

        # Should complete quickly
        self.assertLess(end_time - start_time, 2.0)
        self.assertIsNotNone(matched)

        print(f"✅ Matched answer in {len(speaker_segments)} segments in {end_time - start_time:.3f}s")

    def test_cross_encoder_performance(self):
        """Test cross-encoder performance and fallback."""
        # Test cross-encoder initialization performance
        start_time = time.time()
        ce_model = get_cross_encoder()
        end_time = time.time()

        # Should initialize quickly (or return None quickly)
        self.assertLess(end_time - start_time, 10.0)  # Allow up to 10s for model loading

        if ce_model is not None:
            # Test scoring performance
            start_time = time.time()
            score = compute_semantic_fit_score(
                "What is the meaning of life?",
                "The meaning of life is a profound question.",
                "Context about philosophy and theology."
            )
            end_time = time.time()

            # Should complete reasonably quickly
            self.assertLess(end_time - start_time, 5.0)
            self.assertIsInstance(score, float)

            print(f"✅ Cross-encoder scoring completed in {end_time - start_time:.3f}s")
        else:
            print("✅ Cross-encoder not available, using fallback")

    def test_qc_audit_performance(self):
        """Test QC audit performance with large dataset."""
        # Create large test dataset
        large_train_data = []
        large_val_data = []
        large_train_metadata = []
        large_val_metadata = []

        for i in range(500):  # 500 records each
            large_train_data.append({
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"What is question {i}?"},
                    {"role": "assistant", "content": f"This is answer {i}."}
                ]
            })

            large_val_data.append({
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"What is validation question {i}?"},
                    {"role": "assistant", "content": f"This is validation answer {i}."}
                ]
            })

            large_train_metadata.append({
                "pair_id": f"TRAIN_{i:03d}",
                "episode_id": "EP001",
                "question": f"What is question {i}?",
                "answer": f"This is answer {i}.",
                "answer_speaker": "Fr Stephen De Young",
                "source_spans_text": f"This is answer {i}. This is additional context.",
                "gates": {
                    "speaker_check_passed": True,
                    "pairfit_passed": True,
                    "questionify_used": False,
                    "answer_trimmed": False
                }
            })

            large_val_metadata.append({
                "pair_id": f"VAL_{i:03d}",
                "episode_id": "EP001",
                "question": f"What is validation question {i}?",
                "answer": f"This is validation answer {i}.",
                "answer_speaker": "Jonathan Pageau",
                "source_spans_text": f"This is validation answer {i}. This is additional context.",
                "gates": {
                    "speaker_check_passed": True,
                    "pairfit_passed": True,
                    "questionify_used": False,
                    "answer_trimmed": False
                }
            })

        # Test QC audit performance with 3% sample
        auditor = HarmonyQCAuditor(sample_rate=0.03, seed=42)

        start_time = time.time()
        results = auditor.run_qc_audit(large_train_data, large_val_data, large_train_metadata, large_val_metadata)
        end_time = time.time()

        # Should complete in reasonable time
        self.assertLess(end_time - start_time, 10.0)
        self.assertIn("total_records", results)
        self.assertEqual(results["total_records"], 1000)  # 500 train + 500 val

        print(f"✅ QC audit of {results['total_records']} records completed in {end_time - start_time:.2f}s")

    def test_validation_performance(self):
        """Test validation performance with large files."""
        # Create large test files
        for filename in ["train.jsonl", "val.jsonl", "train_metadata.jsonl", "val_metadata.jsonl"]:
            path = self.temp_dir / filename
            with open(path, 'w', encoding='utf-8') as f:
                for i in range(1000):  # 1000 records each
                    if "metadata" in filename:
                        record = {
                            "pair_id": f"{filename.upper()}_{i:03d}",
                            "episode_id": "EP001",
                            "question": f"What is question {i}?",
                            "answer": f"This is answer {i}.",
                            "answer_speaker": "Fr Stephen De Young" if i % 2 == 0 else "Jonathan Pageau",
                            "source_spans_text": f"This is answer {i}. Additional context.",
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
                                {"role": "user", "content": f"What is question {i}?"},
                                {"role": "assistant", "content": f"This is answer {i}."}
                            ]
                        }
                    f.write(json.dumps(record) + '\n')

        # Test validation performance
        validator = StrictValidator()

        start_time = time.time()
        validation_passed = validator.run_validation(self.temp_dir)
        end_time = time.time()

        # Should complete in reasonable time
        self.assertLess(end_time - start_time, 15.0)

        print(f"✅ Validation of {4000} total records completed in {end_time - start_time:.2f}s")

    def test_manifest_creation_performance(self):
        """Test manifest creation performance with large files."""
        # Create large test files
        for filename in ["train.jsonl", "val.jsonl", "train_metadata.jsonl", "val_metadata.jsonl"]:
            path = self.temp_dir / filename
            with open(path, 'w', encoding='utf-8') as f:
                for i in range(500):  # 500 records each
                    record = {
                        "pair_id": f"{filename.upper()}_{i:03d}",
                        "question": f"What is question {i}?",
                        "answer": f"This is answer {i}.",
                        "answer_speaker": "Fr Stephen De Young" if i % 2 == 0 else "Jonathan Pageau",
                        "source_spans_text": f"This is answer {i}. Additional context.",
                    }
                    f.write(json.dumps(record) + '\n')

        # Test manifest creation performance
        start_time = time.time()
        manifest = create_split_manifest(self.temp_dir)
        end_time = time.time()

        # Should complete quickly
        self.assertLess(end_time - start_time, 5.0)
        self.assertIn("file_hashes", manifest)
        self.assertIn("record_counts", manifest)

        print(f"✅ Manifest creation for {2000} records completed in {end_time - start_time:.2f}s")

    def test_memory_usage_bounds(self):
        """Test that memory usage stays within reasonable bounds."""
        import psutil
        import os

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create large dataset for testing
        large_segments = []
        for i in range(5000):
            large_segments.append(
                Segment(
                    episode_id="MEMORY_TEST",
                    start=i * 1.0,
                    end=i * 1.0 + 0.5,
                    speaker="Fr Stephen De Young",
                    text=f"This is segment {i} for memory testing.",
                    order=i,
                    source_path=Path("test.jsonl")
                )
            )

        # Test merge with large dataset
        merged = merge_contiguous_segments(large_segments, max_gap=1.0, max_length=1200)

        # Check memory usage hasn't grown excessively
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (< 500MB for this test)
        self.assertLess(memory_increase, 500)

        print(f"✅ Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB (+{memory_increase:.1f}MB)")

    def test_concurrent_operations(self):
        """Test concurrent operations don't interfere."""
        import threading
        import queue

        results = queue.Queue()

        def worker(worker_id: int):
            # Each worker processes a subset of segments
            segments = []
            for i in range(100):
                segments.append(
                    Segment(
                        episode_id=f"CONCURRENT_{worker_id}",
                        start=i * 2.0,
                        end=i * 2.0 + 1.0,
                        speaker="Fr Stephen De Young",
                        text=f"Worker {worker_id} segment {i}",
                        order=i,
                        source_path=Path("test.jsonl")
                    )
                )

            # Test merge operation
            merged = merge_contiguous_segments(segments, max_gap=1.0, max_length=1200)
            results.put(f"Worker {worker_id}: {len(merged)} segments")

        # Start multiple workers
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all workers to complete
        for thread in threads:
            thread.join()

        # Collect results
        worker_results = []
        while not results.empty():
            worker_results.append(results.get())

        # All workers should complete successfully
        self.assertEqual(len(worker_results), 5)
        for result in worker_results:
            self.assertIn("Worker", result)
            self.assertIn("segments", result)

        print(f"✅ Concurrent operations completed: {worker_results}")

    def test_scalability_with_data_size(self):
        """Test scalability as data size increases."""
        # Test with different data sizes
        sizes = [100, 500, 1000]

        for size in sizes:
            segments = []
            for i in range(size):
                segments.append(
                    Segment(
                        episode_id=f"SCALE_{size}",
                        start=i * 3.0,
                        end=i * 3.0 + 1.0,
                        speaker="Fr Stephen De Young",
                        text=f"Scale test segment {i}",
                        order=i,
                        source_path=Path("test.jsonl")
                    )
                )

            start_time = time.time()
            merged = merge_contiguous_segments(segments, max_gap=1.0, max_length=1200)
            end_time = time.time()

            # Time should scale roughly linearly
            processing_time = end_time - start_time

            print(f"✅ Size {size}: {processing_time:.3f}s for {len(merged)} merged segments")

            # Basic sanity checks
            self.assertGreater(len(merged), 0)
            self.assertLessEqual(processing_time, size * 0.01)  # Rough upper bound


if __name__ == "__main__":
    unittest.main()
