.PHONY: default

CONDA_ENV = mit-py311
CONDA_YML ?= conda.yaml

default:
	@echo Please use other targets

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

install-venv-deps:
	venv/bin/pip install -r requirements-moeflow.txt

conda-venv: .conda_env_created # alt to `venv/.venv_created` target, but uses conda python to create venv
	micromamba run --attach '' -n $(CONDA_ENV) python3 -mvenv --system-site-packages  ./venv
	touch venv/.venv_created

.conda_env_created: $(CONDA_YML)
	# setup conda environment AND env-wise deps
	micromamba env create -n $(CONDA_ENV) --yes -f $(CONDA_YML)
	touch $@
