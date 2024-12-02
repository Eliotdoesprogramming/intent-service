.PHONY: dockerdev test test-short test-long test-cov lint lint-fix clean build-model-image

# Image name for consistency
IMAGE_NAME = test-intent

# Python paths
PYTHON_PATHS = src

# Test settings
TEST_PATHS = tests/
PYTEST_ARGS = -v

# Ruff settings
RUFF_FORMAT_ARGS = --select I --fix
RUFF_LINT_ARGS = --fix

# MLflow model settings
MODEL_URI ?= runs:/latest/model
MODEL_IMAGE_NAME ?= mlflow-model

dockerdev:
	docker build -t $(IMAGE_NAME) .
	docker run --rm -it -p 8000:8000 -v $(PWD)/mlruns:/app/mlruns $(IMAGE_NAME)

test:
	uv run pytest $(TEST_PATHS) $(PYTEST_ARGS)

test-short:
	uv run pytest $(TEST_PATHS) $(PYTEST_ARGS) -m "not long"

test-long:
	uv run pytest $(TEST_PATHS) $(PYTEST_ARGS) -m long

test-cov:
	uv run pytest $(TEST_PATHS) $(PYTEST_ARGS) --cov=$(PYTHON_PATHS) --cov-report=term-missing

lint:
	uv run ruff check $(PYTHON_PATHS)
	uv run ruff format --check $(PYTHON_PATHS)

lint-fix:
	uv run ruff format src
	uv run ruff check src --fix

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf model_cache
	rm -rf mlruns
	- docker rmi $(IMAGE_NAME)

install:
	uv sync

build-model-image:
	uv run mlflow models build-docker -m $(MODEL_URI) -n $(MODEL_IMAGE_NAME) --enable-mlserver
