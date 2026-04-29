# Makefile — common project commands

.PHONY: install test lint clean notebook help

## Install the project in editable mode
install:
	pip install -e .
	pip install -r requirements.txt

## Run all unit tests
test:
	pytest tests/ -v --tb=short

## Run tests with coverage report
test-cov:
	pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

## Lint source code
lint:
	flake8 src/ --max-line-length=100 --ignore=E501,W503

## Start Jupyter Lab
notebook:
	jupyter lab notebooks/

## Remove Python cache files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

## Show available commands
help:
	@grep -E '^##' Makefile | sed 's/## //'