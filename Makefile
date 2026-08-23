# ---------------------------------------------------------------------------
# Browser Agent -- developer entry points.
# Everything assumes you run `make` from the repository root.
# ---------------------------------------------------------------------------
SHELL := /bin/bash
PY          ?= python
VENV        ?= .venv
MCP_PACKAGE ?= @playwright/mcp@latest

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
browsers: ## Download the Chromium build the MCP server drives
	# The browser revision must match the Playwright version bundled *inside*
	# @playwright/mcp, which is not necessarily `playwright@latest`. Installing
	# the MCP package first makes `npx playwright` resolve to that exact
	# version, so it downloads the revision the server will actually look for.
	@mkdir -p .tools
	# `npm init -y` cannot name a dot-directory, and without a package.json
	# `npm install` silently no-ops -- so write the manifest directly.
	cd .tools && echo '{ "name": "playwright-tools", "private": true }' > package.json
	cd .tools && npm install --no-audit --no-fund $(MCP_PACKAGE)
	cd .tools && npx playwright install chromium

.PHONY: backend
backend: ## Run the FastAPI backend (http://localhost:8000)
	cd backend && ../$(VENV_BIN)/uvicorn main:app --reload --host 0.0.0.0 --port 8000

.PHONY: frontend
frontend: ## Run the Vite dev server (http://localhost:5173)
	cd frontend && npm run dev

.PHONY: mcp-server
mcp-server: ## Run Playwright MCP standalone over HTTP (for MCP_TRANSPORT=http)
	npx -y @playwright/mcp@latest --port 8931 --headless --isolated

.PHONY: test
test: ## Run the backend test suite
	cd backend && ../$(VENV_BIN)/python -m pytest -q

.PHONY: test-e2e
test-e2e: ## Run tests including the real-browser end-to-end test
	cd backend && RUN_E2E=1 ../$(VENV_BIN)/python -m pytest -q -m e2e

.PHONY: lint
lint: ## Type-check the frontend and byte-compile the backend
	cd frontend && npm run typecheck
	$(VENV_BIN)/python -m compileall -q backend

.PHONY: lock
lock: ## Freeze the currently installed Python deps into requirements.lock.txt
	$(VENV_BIN)/pip freeze > backend/requirements.lock.txt

.PHONY: clean
clean: ## Remove runtime data (database + screenshots)
	rm -rf data artifacts
	mkdir -p data artifacts

.PHONY: up
up: ## Start everything with Docker Compose
	docker compose up --build

.PHONY: down
down: ## Stop Docker Compose
	docker compose down
