SHELL := powershell.exe

run:
	$$env:PIPENV_PIPFILE = "app/Pipfile"; $$env:PYTHONPATH = "."; pipenv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

shell:
	cd app; $$env:PIPENV_PIPFILE = "Pipfile"; pipenv shell

install:
	cd app; $$env:PIPENV_PIPFILE = "Pipfile"; pipenv install

sync:
	cd app; pipenv sync

freeze:
	cd app; pipenv lock --requirements > requirements.txt

test:
	cd app; pipenv run pytest

format:
	cd app; pipenv run black .

lint:
	cd app; pipenv run pylint app

env:
	cd app; pipenv run python -m dotenv.cli get

ai:
	pipenv run python tests/test_model.py

migrate:
	@if ("$(msg)" -eq "") { Write-Host "Please provide a message: make migrate msg='your message'"; exit 1 }
	cd app; pipenv run alembic --config ../alembic.ini revision --autogenerate -m "$(msg)"

upgrade:
	cd app; pipenv run alembic --config ../alembic.ini upgrade head

downgrade:
	cd app; pipenv run alembic --config ../alembic.ini downgrade -1

show:
	cd app; pipenv run alembic --config ../alembic.ini current

redis-install:
	docker run -d -p 6379:6379 --name redis redis

redis-ping:
	rdcli ping

redis-start:
	docker start redis

redis-stop:
	docker stop redis

redis-remove:
	docker rm redis

task-kill:
	taskkill /PID 13332 /F;
	taskkill /PID 14360 /F