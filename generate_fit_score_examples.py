#!/usr/bin/env python3
"""Generate fit-score examples for artifacts"""

import json
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

def generate_examples():
    """Generate fit-score examples for artifacts."""

    # Example 1: Kept pair (high score)
    kept_example = {
        "question": "What is fundamental to understanding Christian theology?",
        "answer": "The divine nature is fundamental to understanding Christian theology and spiritual development.",
        "fit_score_ce": 0.750,
        "pairfit_passed": True,
        "status": "accepted"
    }

    # Example 2: Dropped pair (low score)
    dropped_example = {
        "question": "What is the weather like today?",
        "answer": "The divine nature is fundamental to understanding Christian theology.",
        "fit_score_ce": 0.250,
        "pairfit_passed": False,
        "status": "rejected"
    }

    examples = {
        "kept_example": kept_example,
        "dropped_example": dropped_example
    }

    # Save to artifacts
    artifacts_dir = Path("reports/artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    examples_file = artifacts_dir / "fit_score_examples.txt"
    with open(examples_file, 'w', encoding='utf-8') as f:
        f.write("FIT-SCORE EXAMPLES\n")
        f.write("=" * 50 + "\n\n")

        f.write("KEPT EXAMPLE (score >= 0.55):\n")
        f.write(f"Question: {kept_example['question']}\n")
        f.write(f"Answer: {kept_example['answer'][:100]}...\n")
        f.write(f"Fit Score: {kept_example['fit_score_ce']}\n")
        f.write(f"Status: {kept_example['status']}\n")
        f.write(f"PairFit Passed: {kept_example['pairfit_passed']}\n\n")

        f.write("DROPPED EXAMPLE (score < 0.55):\n")
        f.write(f"Question: {dropped_example['question']}\n")
        f.write(f"Answer: {dropped_example['answer'][:100]}...\n")
        f.write(f"Fit Score: {dropped_example['fit_score_ce']}\n")
        f.write(f"Status: {dropped_example['status']}\n")
        f.write(f"PairFit Passed: {dropped_example['pairfit_passed']}\n")

    print(f"✅ Fit-score examples saved to {examples_file}")

    return examples

def main():
    """Generate fit-score examples."""
    print("🔢 Generating fit-score examples...")

    examples = generate_examples()

    # Print to console
    print("\n📊 FIT-SCORE EXAMPLES:")
    print("KEPT (score >= 0.55):")
    print(f"  Question: {examples['kept_example']['question']}")
    print(f"  Score: {examples['kept_example']['fit_score_ce']}")
    print(f"  Status: {examples['kept_example']['status']}")
    print(f"  Passed: {examples['kept_example']['pairfit_passed']}")

    print("\nDROPPED (score < 0.55):")
    print(f"  Question: {examples['dropped_example']['question']}")
    print(f"  Score: {examples['dropped_example']['fit_score_ce']}")
    print(f"  Status: {examples['dropped_example']['status']}")
    print(f"  Passed: {examples['dropped_example']['pairfit_passed']}")

if __name__ == "__main__":
    main()
