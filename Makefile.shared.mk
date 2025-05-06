PYTHON_BIN ?= python3
CONDA_YML ?= conda.yaml
# CONDA_ENV = OVERRIDE_ME

deps: venv venv/.deps_installed

venv/.deps_installed: venv requirements-moeflow.txt
	venv/bin/pip install -r requirements-moeflow.txt --editable .
	@echo "deps installed"
	@touch $@

upgrade-deps:
	venv/bin/pur -r requirements.txt

test:
	venv/bin/pytest

format:
	venv/bin/ruff format src notebooks

venv: venv/.venv_created

# default:
venv/.venv_created:
	$(PYTHON_BIN) -mvenv venv
	@touch $@

# alter: `make conda-venv` to uses conda python
conda-venv: .conda_env_created
	micromamba run --attach '' -n $(CONDA_ENV) $(PYTHON_BIN) -mvenv --system-site-packages  ./venv
	@touch venv/.venv_created

.conda_env_created: $(CONDA_YML)
	# setup conda environment AND env-wise deps
	micromamba env create -n $(CONDA_ENV) --yes -f $(CONDA_YML)
	@touch $@

.PHONY:
