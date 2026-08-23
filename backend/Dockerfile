# ---------------------------------------------------------------------------
# The backend needs Python (the app), Node (to run `npx @playwright/mcp`), and
# the Chromium build plus its system libraries. The official Playwright image
# already carries Node, the browsers and the OS packages, so we start there and
# add Python rather than the other way round.
#
# Bump this tag when you bump the Playwright MCP version -- the browser binaries
# baked into the image must be new enough for the MCP server that drives them.
# ---------------------------------------------------------------------------
FROM mcr.microsoft.com/playwright:v1.49.0-noble

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    NODE_ENV=production

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pre-warm the npm cache so the first run does not pay the download cost while
# the MCP handshake timeout is ticking.
RUN npm install -g @playwright/mcp@latest

COPY backend/requirements.txt /app/backend/requirements.txt
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r /app/backend/requirements.txt
ENV PATH="/opt/venv/bin:$PATH"

COPY backend /app/backend

RUN mkdir -p /app/data /app/artifacts

WORKDIR /app/backend
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
