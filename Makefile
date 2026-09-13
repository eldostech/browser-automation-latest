# ---------------------------------------------------------------------------
# Browser Agent -- developer entry points.
# Everything assumes you run `make` from the repository root.
# ---------------------------------------------------------------------------
SHELL := /bin/bash
PY          ?= python
VENV        ?= .venv

ifeq ($(OS),Windows_NT)
  VENV_BIN := $(VENV)/Scripts
else
  VENV_BIN := $(VENV)/bin
endif

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

.PHONY: setup
setup: setup-backend setup-frontend browsers ## Full first-time setup

.PHONY: setup-backend
setup-backend: ## Create the venv and install Python dependencies
	$(PY) -m venv $(VENV)
	$(VENV_BIN)/python -m pip install --upgrade pip
	$(VENV_BIN)/pip install -r backend/requirements.txt

.PHONY: setup-frontend
setup-frontend: ## Install frontend dependencies
	cd frontend && npm install

.PHONY: browsers
browsers: playwright-browser ## Download the Chromium the engine and recorder drive

.PHONY: playwright-browser
playwright-browser: ## Chromium for the replay engine and the recorder
	# One browser for both. Recording shells out to this same Playwright, so
	# there is no second install and no version to keep in step.
	$(VENV_BIN)/python -m playwright install chromium

# --- database --------------------------------------------------------------

.PHONY: db-upgrade
db-upgrade: ## Apply migrations (creates the schema on first run)
	cd backend && ../$(VENV_BIN)/python -m alembic upgrade head

.PHONY: db-revision
db-revision: ## Autogenerate a migration from model changes: make db-revision m="what changed"
	@test -n "$(m)" || { echo 'usage: make db-revision m="what changed"'; exit 1; }
	cd backend && ../$(VENV_BIN)/python -m alembic revision --autogenerate -m "$(m)"

.PHONY: db-check
db-check: ## Fail if the models and the migrations disagree
	cd backend && ../$(VENV_BIN)/python -m alembic check

.PHONY: db-history
db-history: ## Show the migration history and where this database sits
	cd backend && ../$(VENV_BIN)/python -m alembic history --indicate-current

.PHONY: backend
backend: db-upgrade ## Run the FastAPI backend (http://localhost:8000)
	cd backend && HOST=0.0.0.0 ../$(VENV_BIN)/python serve.py

.PHONY: worker
worker: ## Run a batch worker on its own (set WORKER_ENABLED=false on the API)
	cd backend && ../$(VENV_BIN)/python -m worker

.PHONY: frontend
frontend: ## Run the Vite dev server (http://localhost:5173)
	cd frontend && npm run dev

.PHONY: test
test: ## Run the backend test suite (needs Postgres; uses its own schema)
	cd backend && ../$(VENV_BIN)/python -m pytest -q

.PHONY: test-e2e
test-e2e: ## Run the real-browser tests (record -> replay against a local site)
	cd backend && RUN_E2E=1 ../$(VENV_BIN)/python -m pytest -q -m e2e

.PHONY: lint
lint: ## Type-check the frontend and byte-compile the backend
	cd frontend && npm run typecheck
	$(VENV_BIN)/python -m compileall -q backend

.PHONY: lock
lock: ## Freeze the currently installed Python deps into requirements.lock.txt
	$(VENV_BIN)/pip freeze > backend/requirements.lock.txt

.PHONY: clean
clean: ## Remove local runtime files (screenshots, checkpoints). Leaves Postgres alone.
	rm -rf data artifacts
	mkdir -p data artifacts

.PHONY: db-drop
db-drop: ## DESTRUCTIVE: drop the application schema and everything in it
	@read -p "Drop schema '$${DB_SCHEMA:-browser}' and all its data? [y/N] " ok; 	  [ "$$ok" = "y" ] || { echo "cancelled"; exit 1; }
	cd backend && ../$(VENV_BIN)/python -c "import sys; sys.path.insert(0,'.'); 	  from config import get_settings; from db.engine import sync_database_url; 	  from sqlalchemy import create_engine, text; s=get_settings(); 	  e=create_engine(sync_database_url(s)); c=e.connect(); 	  c.execute(text('DROP SCHEMA IF EXISTS \"'+s.db_schema+'\" CASCADE')); c.commit(); 	  print('dropped', s.db_schema)"

.PHONY: up
up: ## Start everything with Docker Compose
	docker compose up --build

.PHONY: down
down: ## Stop Docker Compose
	docker compose down
