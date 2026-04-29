# Makefile — Healthcare RAG-Powered Medical Q&A Assistant

.PHONY: install setup test lint clean notebook api dashboard help

## Install dependencies
install:
	pip install -r requirements.txt

## Full setup (install + download data)
setup:
	pip install -r requirements.txt
	python download_data.py

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

## Start FastAPI server
api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

## Start Streamlit dashboard
dashboard:
	streamlit run dashboard/app.py

## Download data from HuggingFace
download-data:
	python download_data.py

## Remove Python cache files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true

## Show available commands
help:
	@echo "Available commands:"
	@echo "  make setup          - Install deps + download data (fresh clone)"
	@echo "  make install        - Install dependencies only"
	@echo "  make download-data  - Download data from HuggingFace"
	@echo "  make test           - Run unit tests"
	@echo "  make api            - Start FastAPI server"
	@echo "  make dashboard      - Start Streamlit dashboard"
	@echo "  make notebook       - Open Jupyter Lab"
	@echo "  make clean          - Remove cache files"