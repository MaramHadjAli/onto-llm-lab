.PHONY: install test lint format reproduce

UV ?= uv

install:
	$(UV) sync --all-groups --frozen

lint:
	$(UV) run ruff check src tests
	$(UV) run ruff format --check src tests

format:
	$(UV) run ruff format src tests
	$(UV) run ruff check --fix src tests

test: lint
	$(UV) run pytest

# Phase 0 has no experiment cache yet. Later phases make this replay cached LLM calls.
reproduce: test
	@echo "Phase 0 reproduce: offline test suite only. Experiment replay starts in a later phase."
