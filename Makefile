.EXPORT_ALL_VARIABLES:
DEV=True

venv:
	# Install https://github.com/astral-sh/uv on macOS and Linux:
	# $ curl -LsSf https://astral.sh/uv/install.sh | sh
	# Other recommended libraries, add with `uv add <library>`:
	# tenacity, joblib, jupyterlab, litellm, datasets, pytorch, fastapi, uvicorn, rich
	uv sync

which-python:
	uv run which python | pbcopy
	uv run which python

clean:
	rm -rf .venv

mlflow:
	mlflow ui --host 127.0.0.1 --port 5001 --backend-store-uri sqlite:///mlruns.db

run-example:
	PYTHONPATH='./src' uv run src/clients/example/aime.py --max-metric-calls 1

run-gnaf-implicit-default-adapter:
	# Case 1: No adapter specified, implicity uses builtin GEPA default adapter
	PYTHONPATH='./src' uv run src/clients/example/gnaf_implicit_default_adapter.py --max-metric-calls 2

run-gnaf-explicit-default-adapter:
	# Case 2: Explicitly use builtin default GEPA adapter
	PYTHONPATH='./src' uv run src/clients/example/gnaf_explicit_default_adapter.py --max-metric-calls 2

run-gnaf-copy-default-adapter:
	# Case 3: Copy of default GEPA adapter classes
	# PYTHONPATH='./src' uv run src/clients/example/gnaf_copy_default_adapter.py --max-metric-calls 100 --split-counts 20  # ~15 cents
	# PYTHONPATH='./src' uv run src/clients/example/gnaf_copy_default_adapter.py --max-metric-calls 200 --split-counts 40
	PYTHONPATH='./src' uv run src/clients/example/gnaf_copy_default_adapter.py --max-metric-calls 2 --split-counts 2

test:
	PYTHONPATH='./src' uv run pytest -vv --capture=no tests

manual-checks:
	uv run ruff format .
	uv run ruff check . --fix
	uv run pyright

precommit-install:
	# One time: Install git hook to run pre-commit automatically on git commit
	# Uninstall with: uv run pre-commit uninstall
	uv run pre-commit install

precommit:
	uv run pre-commit run --all-files

.DEFAULT_GOAL := help
.PHONY: help
help:
	@LC_ALL=C $(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$'
