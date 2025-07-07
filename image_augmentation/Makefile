SRC?=$(shell pwd)

# Resources dir
RESOURCES_DIR?=image_augmentation/resources

PYTHON?=python3.10

help:
	@cat Makefile

###########################################################################
# Dev targets
###########################################################################
setup-quality-env:
	$(PYTHON) -m pip install ruff coverage[toml] -U

setup-dev-env: setup-quality-env
	ln -nfs $(PWD) ~/.image_augmentation
	$(PYTHON) -m pip install -r requirements.txt

run-test: setup-dev-env
	PYTHONPATH=$(SRC):$(SRC)/test:$$PYTHONPATH RESOURCES_DIR=$(RESOURCES_DIR) coverage run -m unittest discover -s test -p 'test_*.py' -v
	coverage xml

check_dirs := . image_augmentation test

quality-check: setup-quality-env
	ruff check $(check_dirs)

format: setup-quality-env
	ruff format $(check_dirs)
	ruff check $(check_dirs) --fix

.PHONY: run

run:
	PYTHONPATH=$(SRC) $(PYTHON) image_augmentation/image_augmentor.py