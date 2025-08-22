###
### SECTION dev scripts
###

format-bean:
	find . -name '*.bean' -print0 | xargs -0 --max-procs=8 -I % venv/bin/bean-format --output=% %

format-py:
	venv/bin/ruff format moeflow_companion

test: deps
	venv/bin/pytest lib

test-watch: deps
	. venv/bin/activate && exec pytest-watcher lib

###
### SECTION deps
###

PYTHON_VER ?= 3.11

# comma separated packages
FREEZE_PY_REQ = elasticsearch,fava-dashboard,opentelemetry-api,opentelemetry-sdk

REQUIREMENTS = -r requirements-moeflow.txt

deps: venv/.deps_installed # .PHONY

venv/.deps_installed: venv requirements-moeflow.txt Makefile
	@# the most useful feature of uv
	UV_PYTHON=venv UV_LINK_MODE=symlink uv pip install $(REQUIREMENTS)
	@echo "deps installed"
	@touch $@

upgrade-deps:
	venv/bin/pur -r requirements.txt --force --skip=$(FREEZE_PY_REQ)

venv: venv/.venv_created

venv/.venv_created: Makefile
	@# the 2nd most useful feature of uv
	uv venv --clear --python=$(PYTHON_VER) venv
	@touch $@
