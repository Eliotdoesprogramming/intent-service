.PHONY: dockerdev

# Image name for consistency
IMAGE_NAME = test-intent

dockerdev:
	docker build -t $(IMAGE_NAME) .
	docker run --rm -it -p 8000:8000 $(IMAGE_NAME) 