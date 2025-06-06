SHELL := powershell.exe

REDIS_CONF_PATH := D:/UrFU-chatbot/redis.conf
REDIS_CONTAINER_NAME := redis-chatbot

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
	docker run -d -p 6379:6379 --name $(REDIS_CONTAINER_NAME) -v "$(REDIS_CONF_PATH):/usr/local/etc/redis/redis.conf" redis:latest redis-server /usr/local/etc/redis/redis.conf

# Проверка подключения к Redis
redis-ping:
	docker exec $(REDIS_CONTAINER_NAME) redis-cli ping

# Запуск существующего контейнера
redis-start:
	docker start $(REDIS_CONTAINER_NAME)

# Остановка контейнера
redis-stop:
	docker stop $(REDIS_CONTAINER_NAME)

# Удаление контейнера
redis-remove:
	docker rm $(REDIS_CONTAINER_NAME)

# Полная переустановка Redis
redis-reinstall: redis-stop redis-remove redis-install

# Просмотр логов Redis
redis-logs:
	docker logs $(REDIS_CONTAINER_NAME)

# Подключение к Redis CLI
redis-cli:
	docker exec -it $(REDIS_CONTAINER_NAME) redis-cli

# Остановка и удаление (для очистки)
redis-clean: redis-stop redis-remove

task-kill:
	taskkill /PID 13332 /F;
	taskkill /PID 14360 /F

