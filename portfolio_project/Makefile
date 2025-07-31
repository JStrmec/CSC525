SRC?=$(shell pwd)

# Resources dir
RESOURCES_DIR?=mental_health_retrieval_chatbot/resources

PYTHON?=python3.10

help:
	@cat Makefile

########################################################
# Dev targets
########################################################
setup-quality-env:
	$(PYTHON) -m pip install ruff coverage[toml] -U

setup-dev-env: setup-quality-env
	ln -nfs $(PWD) ~/.mental_health_retrieval_chatbot
	$(PYTHON) -m pip install -r requirements.txt

run-test: setup-dev-env
	PYTHONPATH=$(SRC):$(SRC)/test:$$PYTHONPATH RESOURCES_DIR=$(RESOURCES_DIR) coverage run -m unittest discover -s test -p 'test_*.py' -v
	coverage xml

check_dirs := . test

quality-check: setup-quality-env
	ruff check $(check_dirs)

format: setup-quality-env
	ruff format $(check_dirs)
	ruff check $(check_dirs) --fix

.PHONY: run

run: setup-dev-env
	PYTHONPATH=$(SRC) $(PYTHON) mental_health_retrieval_chatbot/ui.py
	
run-tuning: setup-dev-env
	PYTHONPATH=$(SRC) $(PYTHON) mental_health_retrieval_chatbot/chatbot_model_tuning.py