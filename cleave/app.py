"""Cleave web app: upload → background job → routed, explainable knowledge units.

The composition root, and nothing else. This file was 540 lines holding the job
model, disk persistence, the whole ingest→chunk→enrich orchestration and every
route; those live in ``cleave.pipeline`` and ``cleave.web`` now. What stays here
is what an entry point is for: build the app, mount the static files, register
the routers, and start the server.

Still deliberately small underneath: a module-level job dict and FastAPI
``BackgroundTasks``. Jobs run in seconds and the artifacts land on disk as JSON;
nothing here needs a queue, a database, or cancellation.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .logging_setup import configure_logging
from .pipeline import run_job
from .web.jobs import DATA, JOBS, Job, rehydrate_jobs
from .web.routes_api import router as api_router
from .web.routes_pages import router as pages_router
from .web.security import SecurityMiddleware
from .web.templating import ROOT, STATIC_DIR

log = logging.getLogger(__name__)

#: Re-exported so ``cleave.app.JOBS`` / ``cleave.app.run_job`` keep working for
#: anything that imported them before the split, tests included.
__all__ = ["DATA", "JOBS", "ROOT", "Job", "app", "run_job"]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Configured here rather than under __main__, so `uvicorn cleave.app:app`
    # gets application logs too — it used to silently discard every one.
    configure_logging()
    rehydrate_jobs()
    yield


app = FastAPI(title="Cleave", docs_url="/docs", lifespan=lifespan)
app.add_middleware(SecurityMiddleware)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.include_router(pages_router)
app.include_router(api_router)


@app.get("/favicon.svg", include_in_schema=False)
def favicon():
    return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")


if __name__ == "__main__":
    import uvicorn

    configure_logging()
    uvicorn.run(app, host="127.0.0.1", port=8321, reload=False)  # `cleave serve` for options
