# ---------------------------------------------------------------------------
# The backend needs Python and a Chromium with its system libraries. The
# official Playwright image carries the browsers and the OS packages, so we
# start there and add Python rather than the other way round.
#
# The tag must match the `playwright` pin in requirements.txt: the Python
# package looks for a specific browser revision, and an image built from a
# different one fails at launch with a message about a missing executable.
# ---------------------------------------------------------------------------
FROM mcr.microsoft.com/playwright/python:v1.48.0-noble

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Recording opens a headed browser and needs a display. A container has
    # none, so it is off here by configuration rather than by a spawn failing
    # with an X11 error nobody can act on.
    RECORDER_ENABLED=false

WORKDIR /app

COPY backend/requirements.txt /app/backend/requirements.txt
# The image already has a Python with Playwright's browsers registered to it,
# so install into it rather than building a venv the browsers do not know about.
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY backend /app/backend

# The Playwright image already ships a non-root `pwuser`, which also owns the
# browser cache -- so reuse it rather than creating another account that would
# then need the cache re-chowned. Running the browser and the app as root is
# the default this image invites and the one worth declining: a page that
# escapes the sandbox should not land on a root shell.
RUN mkdir -p /app/data /app/artifacts     && chown -R pwuser:pwuser /app

USER pwuser

WORKDIR /app/backend
EXPOSE 8000

# Liveness only: is the process answering? Readiness is a separate question
# (can it reach Postgres and Bedrock) and lives at /healthz, which returns
# 503 when a dependency is down. Conflating them
# makes an orchestrator kill a container whose only problem is that the
# database is briefly unavailable -- which restarting will not fix.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3     CMD python -c "import urllib.request,sys;         sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz', timeout=5).status in (200,503) else 1)"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
