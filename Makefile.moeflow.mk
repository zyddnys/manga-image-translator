CONDA_ENV = mit-py311

run-gradio:
	venv/bin/gradio gradio-main.py

run-gradio-mit:
	venv/bin/gradio gradio-mit.py --demo-name=mit_workflow_block

prepare-models:
	venv/bin/$(PYTHON_BIN) docker_prepare.py

