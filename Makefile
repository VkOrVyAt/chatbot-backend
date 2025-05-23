SHELL := /bin/bash

run:
	pipenv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

shell:
	pipenv shell

install:
	pipenv install

freeze:
	pipenv lock --requirements > requirements.txt

test:
	pipenv run pytest

format:
	pipenv run black .

lint:
	pipenv run pylint app

env:
	pipenv run python -m dotenv.cli get

ai:
	pipenv run python tests/test_model.py
