VENV := .venv
PYTHON := python3
PIP := $(VENV)/bin/pip
PYBIN := $(VENV)/bin/python

.PHONY: setup test

setup:
	$(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate; $(PIP) install -U pip
	. $(VENV)/bin/activate; $(PIP) install -r requirements.txt

test:
	. $(VENV)/bin/activate; PYTHONPATH=. $(VENV)/bin/pytest -q
