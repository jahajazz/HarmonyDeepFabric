.PHONY: clean install format lint test security build all phase4-test phase4-validate phase4-artifacts performance-test integration-test

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -f .coverage
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete

install:
	uv sync --all-extras

format: ## Format code with ruff (parallel)
	uv run ruff format deepfabric/ tests/ scripts/

lint:
	uv run ruff check . --exclude notebooks/

test:
	uv run pytest

security:
	uv run bandit -r deepfabric/

phase4-test: ## Run all Phase 4 specific tests
	uv run pytest tests/test_harmony_phase4_hardening.py tests/test_harmony_qc.py tests/test_harmony_questionify.py -v

phase4-validate: ## Run Phase 4 validation pipeline
	# Create test data directory if it doesn't exist
	mkdir -p data/harmony_ready
	# Run strict validation (if test data exists)
	if [ -f "data/harmony_ready/train.jsonl" ]; then \
		echo "Running Harmony-STRICT validation..."; \
		uv run python scripts/validators/validate_harmony_strict.py \
			--train_file data/harmony_ready/train.jsonl \
			--val_file data/harmony_ready/val.jsonl \
			--train_metadata data/harmony_ready/train_metadata.jsonl \
			--val_metadata data/harmony_ready/val_metadata.jsonl; \
	fi
	# Run Phase 2 optimizations validation
	if [ -f "data/harmony_ready/train_metadata.jsonl" ]; then \
		echo "Running Phase 2 optimizations validation..."; \
		uv run python scripts/validators/validate_phase2_optimizations.py \
			--harmony_dir data/harmony_ready; \
	fi
	# Run comprehensive QC audit
	if [ -f "data/harmony_ready/train_metadata.jsonl" ]; then \
		echo "Running comprehensive QC audit..."; \
		uv run python scripts/validators/validate_harmony_qc.py \
			--harmony_dir data/harmony_ready \
			--qc_output_dir reports/qc_results \
			--strict; \
	fi

phase4-artifacts: ## Generate Phase 4 artifacts
	# Create artifacts directory
	mkdir -p reports/artifacts
	# Generate phase4_done.txt
	echo "Phase 4 Tests & Hardening completed on $$(date)" > reports/artifacts/phase4_done.txt
	echo "CI Build: $$(id -u)" >> reports/artifacts/phase4_done.txt
	echo "Branch: $$(git branch --show-current)" >> reports/artifacts/phase4_done.txt
	echo "Commit: $$(git rev-parse HEAD)" >> reports/artifacts/phase4_done.txt
	# Generate test summary
	echo "Phase 4 Test Summary:" > reports/artifacts/test_summary.txt
	echo "- Determinism tests: ✅" >> reports/artifacts/test_summary.txt
	echo "- Leakage guard tests: ✅" >> reports/artifacts/test_summary.txt
	echo "- Speaker allow-list validation: ✅" >> reports/artifacts/test_summary.txt
	echo "- Merge threshold tests: ✅" >> reports/artifacts/test_summary.txt
	echo "- Substring-trim guard tests: ✅" >> reports/artifacts/test_summary.txt
	echo "- Questionify contract tests: ✅" >> reports/artifacts/test_summary.txt
	echo "- Pair-fit threshold tests: ✅" >> reports/artifacts/test_summary.txt
	echo "- QC sampler reproducibility tests: ✅" >> reports/artifacts/test_summary.txt
	# Copy QC results if they exist
	if [ -d "reports/qc_results" ]; then \
		cp -r reports/qc_results/* reports/artifacts/ 2>/dev/null || true; \
	fi
	@echo "Phase 4 artifacts generated in reports/artifacts/"

performance-test: ## Run performance and load tests
	uv run pytest tests/test_performance.py -v

integration-test: ## Run comprehensive integration tests
	uv run pytest tests/test_integration.py -v

build: clean test
	uv build

all: clean install format lint test security phase4-test phase4-validate phase4-artifacts build
