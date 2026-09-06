"""HTTP routers, one module per resource.

``main.py`` used to hold all thirty endpoints in 1,280 lines. Splitting them
buys three things beyond readability: a router can declare an authorization
dependency once for every route it carries, the shared 404 lookups live in
``deps`` instead of being written out twenty-one times, and the business logic
underneath can be tested without a ``TestClient``.
"""

from routers import (
    admin,
    agent_sessions,
    auth,
    credentials,
    datasets,
    executions,
    health,
    memory,
    targets,
    recordings,
    runs,
    usecases,
)

#: Registration order matters only for documentation grouping; FastAPI matches
#: on path, so there is no shadowing between these.
ALL_ROUTERS = (
    health.router,
    auth.router,
    admin.router,
    runs.router,
    usecases.router,
    credentials.router,
    datasets.router,
    executions.router,
    recordings.router,
    agent_sessions.router,
    memory.router,
    targets.router,
)

__all__ = [
    "ALL_ROUTERS",
    "admin",
    "auth",
    "credentials",
    "datasets",
    "executions",
    "health",
    "memory",
    "agent_sessions",
    "recordings",
    "runs",
    "usecases",
]
