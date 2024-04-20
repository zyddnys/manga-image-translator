.PHONY: default

default:
	@echo Please use other targets

conda-env:
	conda env update --prune --file conda.yaml

run-worker:
	conda run -n mit-py311 --no-capture-output celery --app moeflow_worker worker --queues mit --loglevel=debug --concurrency=1

prepare-models:
	conda run -n mit-py311 --no-capture-output python3 docker_prepare.py

build-image:
	docker rmi manga-image-translator || true
	docker build . --tag=manga-image-translator

run-web-server:
	docker run --gpus all -p 5003:5003 --ipc=host --rm zyddnys/manga-image-translator:main \
		--target-lang=ENG \
		--manga2eng \
		--verbose \
		--mode=web \
		--use-gpu \ 
		--host=0.0.0.0 \
		--port=5003
