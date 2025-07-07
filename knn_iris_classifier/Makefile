SRC?=$(shell pwd)

# Test resources dir
TEST_RESOURCES_DIR?=knn_iris_classifier/resources


help:
	@cat Makefile

###########################################################################
# Dev targets
###########################################################################
setup-quality-env:
	pip install ruff coverage[toml] -U

setup-dev-env: setup-quality-env
	ln -nfs $(PWD) ~/.knn_iris_classifier
	pip install -r requirements.txt
	
run-test: setup-dev-env
	PYTHONPATH=$(SRC):$(SRC)/test:$$PYTHONPATH TEST_RESOURCES_DIR=$(TEST_RESOURCES_DIR) coverage run -m unittest discover -s test -p 'test_*.py' -v
	coverage xml

check_dirs := . knn_iris_classifier test

quality-check: setup-quality-env
	ruff check $(check_dirs)

format: setup-quality-env
	ruff format $(check_dirs)
	ruff check $(check_dirs) --fix

.PHONY: run

run:
	PYTHONPATH=$(SRC) python knn_iris_classifier/iris_knn.py