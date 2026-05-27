APP_NAME  := healthcare-rag
API_PORT  := 8000
ACR_NAME  := healthcareragacr
.DEFAULT_GOAL := help

.PHONY: install test test-api test-ci test-e2e test-workflow test-coverage validate run lint docker-build docker-build-no-cache docker-smoke-test docker-run docker-push docker-pull docker-login docker-dev docker-prod docker-clean docker-stats docker-logs docker-exec docker-restart docker-ps docker-top docker-test docker-save docker-load latency-test mlflow dashboard download download-classifier clean help

install:       ## Install all dependencies
	pip install --upgrade pip && pip install -r requirements.txt && pip install -e .

download-classifier: ## Download BioBERT classifier weights from HuggingFace
	python scripts/download_classifier.py

download:       ## Download data artifacts (FAISS index, CSVs) from HuggingFace
	python download.py

test-workflow:  ## Run GitHub Actions workflow YAML tests
	pytest tests/test_workflow_yml.py -v --tb=short

test:          ## Run all pytest tests
	pytest tests/ -v --tb=short

test-api:      ## Run API tests only (fast, no models needed)
	pytest tests/test_api.py -v

test-ci:       ## Run unit tests matching CI scope (excludes integration-heavy tests)
	pytest tests/ --ignore=tests/test_rag_pipeline.py --tb=short -q

test-e2e:      ## Run end-to-end integration tests (requires real models + FAISS index)
	pytest tests/test_integration_full_pipeline.py -v --tb=long -m integration

test-coverage: ## Run unit tests with coverage gate (100% min, matches CI)
	pytest tests/ --ignore=tests/test_rag_pipeline.py --cov=src --cov=api --cov-report=term-missing --cov-fail-under=100

run:           ## Start FastAPI with hot-reload
	uvicorn api.main:app --reload --host 0.0.0.0 --port $(API_PORT)

lint:          ## Run flake8 style check (matches CI scope)
	flake8 src/ api/ config/ scripts/ mlops/ tests/ dashboard/ --max-line-length 120 --exclude __pycache__,*.pyc

docker-build:  ## Build Docker image
	docker build -f docker/Dockerfile -t $(APP_NAME):latest .

docker-smoke-test:  ## Build Docker image as smoke test (matches CI docker-build job)
	docker build -f docker/Dockerfile -t $(APP_NAME):ci-smoke .

docker-run:    ## Run Docker container locally
	docker run -p $(API_PORT):$(API_PORT) --env-file .env --rm $(APP_NAME):latest

docker-build-no-cache:  ## Build Docker image from scratch (no cache), use when dependencies change
	docker build --no-cache -f docker/Dockerfile -t $(APP_NAME):latest .

docker-push:   ## Push image to Azure Container Registry
	docker tag $(APP_NAME):latest $(ACR_NAME).azurecr.io/$(APP_NAME):v1
	docker push $(ACR_NAME).azurecr.io/$(APP_NAME):v1

docker-pull:   ## Pull latest stack images from Azure Container Registry
	docker pull $(ACR_NAME).azurecr.io/$(APP_NAME):v1
	docker tag $(ACR_NAME).azurecr.io/$(APP_NAME):v1 $(APP_NAME):latest
	docker compose -f docker/docker-compose.yml pull

docker-login:   ## Authenticate Docker to Azure Container Registry (requires az CLI)
	az acr login --name $(ACR_NAME)

docker-dev:    ## Start full stack with dev hot-reloading (auto-loads docker-compose.override.yml)
	docker compose -f docker/docker-compose.yml up --build -d

docker-prod:   ## Start full stack with production overrides
	docker compose -f docker/docker-compose.yml -f docker/docker-compose.prod.yml up --build -d

docker-clean:  ## Stop stack, remove containers, volumes, and dangling images
	docker compose -f docker/docker-compose.yml down -v --remove-orphans
	docker image prune -f 2>/dev/null || true

docker-stats:  ## Show live resource usage of running stack containers (CPU, memory, net, block IO)
	docker stats --filter "name=healthcare-rag"

docker-logs:   ## Tail logs from all stack services simultaneously
	docker compose -f docker/docker-compose.yml logs -f

docker-exec:   ## Open interactive shell in the API container
	docker compose -f docker/docker-compose.yml exec healthcare-rag sh

docker-restart:  ## Restart all stack services
	docker compose -f docker/docker-compose.yml restart

docker-ps:     ## List all stack containers with their status and ports
	docker compose -f docker/docker-compose.yml ps

docker-top:    ## Show running processes inside each stack container
	docker compose -f docker/docker-compose.yml top

docker-test:   ## Run test suite inside a disposable container (pass extra args via ARGS="-k test_api")
	docker compose -f docker/docker-compose.yml run --rm healthcare-rag pytest tests/ -v --tb=short $(ARGS)

docker-save:   ## Export all stack images to tar archives in docker/images/ (for air-gapped deployment)
	mkdir -p docker/images
	docker save $(APP_NAME):latest -o docker/images/$(APP_NAME).tar
	docker save python:3.10-slim -o docker/images/mlflow-base.tar
	@echo ""
	@echo "Images saved to docker/images/. Transfer these .tar files to the air-gapped machine."

docker-load:   ## Load all stack images from tar archives in docker/images/ (for air-gapped deployment)
	docker load -i docker/images/$(APP_NAME).tar
	docker load -i docker/images/mlflow-base.tar
	@echo ""
	@echo "Images loaded. Run 'make docker-dev' or 'make docker-prod' to start the stack."

latency-test:  ## Run 20-query latency test against live Azure API
	python scripts/latency_test.py

mlflow:        ## Start MLflow UI
	mlflow ui --port 5000

dashboard:     ## Open the dashboard (requires API running at /dashboard)
	@echo "Dashboard is served by the FastAPI app at http://localhost:$(API_PORT)/dashboard"
	@echo "Start the API with:  make run"

validate:      ## Run CI-equivalent checks locally (lint + test-workflow + test-ci + test-coverage)
	$(MAKE) lint
	$(MAKE) test-workflow
	$(MAKE) test-ci
	$(MAKE) test-coverage

clean:         ## Remove Python cache files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

help:          ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'
