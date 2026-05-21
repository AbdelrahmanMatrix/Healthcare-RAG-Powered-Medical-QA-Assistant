APP_NAME  := healthcare-rag
API_PORT  := 8000
ACR_NAME  := healthcareragacr
.DEFAULT_GOAL := help

.PHONY: install test test-api run lint docker-build docker-run docker-push latency-test mlflow dashboard download-classifier clean help

install:       ## Install all dependencies
	pip install --upgrade pip && pip install -r requirements.txt && pip install -e .

download-classifier: ## Download BioBERT classifier weights from HuggingFace
	python scripts/download_classifier.py

test:          ## Run all pytest tests
	pytest tests/ -v --tb=short

test-api:      ## Run API tests only (fast, no models needed)
	pytest tests/test_api.py -v

run:           ## Start FastAPI with hot-reload
	uvicorn api.main:app --reload --host 0.0.0.0 --port $(API_PORT)

lint:          ## Run flake8 style check
	flake8 src/ api/ scripts/ mlops/ --max-line-length 120 --exclude __pycache__

docker-build:  ## Build Docker image
	docker build -f docker/Dockerfile -t $(APP_NAME):latest .

docker-run:    ## Run Docker container locally
	docker run -p $(API_PORT):$(API_PORT) --env-file .env --rm $(APP_NAME):latest

docker-push:   ## Push image to Azure Container Registry
	docker tag $(APP_NAME):latest $(ACR_NAME).azurecr.io/$(APP_NAME):v1
	docker push $(ACR_NAME).azurecr.io/$(APP_NAME):v1

latency-test:  ## Run 20-query latency test against live Azure API
	python scripts/latency_test.py

mlflow:        ## Start MLflow UI
	mlflow ui --port 5000

dashboard:     ## Start Streamlit KPI dashboard
	streamlit run dashboard/app.py

clean:         ## Remove Python cache files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

help:          ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'
