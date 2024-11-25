.PHONY: dockerdev test lint lint-fix clean

# Image name for consistency
IMAGE_NAME = test-intent

# Python paths
PYTHON_PATHS = src

# Test settings
TEST_PATHS = src/
PYTEST_ARGS = -v

# Ruff settings
RUFF_FORMAT_ARGS = --select I --fix
RUFF_LINT_ARGS = --fix

dockerdev:
	docker build -t $(IMAGE_NAME) .
	docker run --rm -it -p 8000:8000 $(IMAGE_NAME) 

test:
	uv run pytest $(TEST_PATHS) $(PYTEST_ARGS)

test-cov:
	uv run pytest $(TEST_PATHS) $(PYTEST_ARGS) --cov=$(PYTHON_PATHS) --cov-report=term-missing

lint:
	uv run ruff check $(PYTHON_PATHS)
	uv run ruff format --check $(PYTHON_PATHS)

lint-fix:
	uv run ruff check $(PYTHON_PATHS) $(RUFF_LINT_ARGS)
	uv run ruff format $(PYTHON_PATHS)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	- docker rmi $(IMAGE_NAME)