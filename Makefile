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

migrate:
	@if [ -z "$(msg)" ]; then echo "Please provide a message: make migrate msg='your message'"; exit 1; fi
	alembic revision --autogenerate -m "$(msg)"

# Применить все миграции
upgrade:
	alembic upgrade head

# Откатить последнюю миграцию (на одну версию назад)
downgrade:
	alembic downgrade -1

# Показать текущую версию миграции
show:
	alembic current