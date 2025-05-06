default:
	@echo Please use other targets

build-image:
	docker rmi manga-image-translator || true
	docker build . --tag=manga-image-translator

run-web-server:
	docker run --gpus all -p 5003:5003 --ipc=host --rm manga-image-translator \
		--verbose \
		--use-gpu \
		--host=0.0.0.0 \
		--port=5003 \
		--entrypoint python \
		-v /demo/doc/../../result:/app/result \
		-v /demo/doc/../../server/main.py:/app/server/main.py \
		-v /demo/doc/../../server/instance.py:/app/server/instance.py \	
		zyddnys/manga-image-translator:main \
		server/main.py --verbose --start-instance --host=0.0.0.0 --port=5003 --use-gpu

CONDA_ENV = mit-py311

# for shell completion. they got overridden in included files
venv:

deps:

conda-venv:



include Makefile.shared.mk Makefile.moeflow.mk
