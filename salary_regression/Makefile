SRC?=$(shell pwd)

# Test resources dir
TEST_RESOURCES_DIR?=salary_regression/resources


help:
	@cat Makefile

###########################################################################
# Dev targets
###########################################################################
setup-quality-env:
	pip install ruff coverage[toml] -U

setup-dev-env: setup-quality-env
	pip install -r requirements.txt
	
run-test: setup-dev-env
	PYTHONPATH=$(SRC):$(SRC)/test:$$PYTHONPATH TEST_RESOURCES_DIR=$(TEST_RESOURCES_DIR) coverage run -m unittest discover -s test -p 'test_*.py' -v
	coverage xml

check_dirs := . test

quality-check: setup-quality-env
	ruff check $(check_dirs)

format: setup-quality-env
	ruff format $(check_dirs)
	ruff check $(check_dirs) --fix

.PHONY: run

run: #setup-dev-env
	PYTHONPATH=$(SRC) python3 polynomial_salary_regressor.py