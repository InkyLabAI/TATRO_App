"""Thread-safe, per-browser progress reporting for the local Dash server.

Workers publish small events; a separate polling callback reads them while model
callbacks are still running. No model state or credentials are stored here.
"""
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from datetime import datetime, timezone
import re
from threading import RLock
import time
import uuid

_current = ContextVar("progress_job", default=None)
_lock = RLock()
_sessions = {}
MAX_EVENTS = 500


def _prune():
    cutoff = time.time() - 24 * 3600
    for key, state in list(_sessions.items()):
        if state["updated"] < cutoff and not any(j["status"] == "running" for j in state["jobs"].values()):
            del _sessions[key]


@contextmanager
def task(session_id, title):
    job_id = uuid.uuid4().hex
    with _lock:
        _prune()
        state = _sessions.setdefault(session_id, {"jobs": {}, "events": deque(maxlen=MAX_EVENTS), "updated": time.time()})
        state["jobs"][job_id] = {"id": job_id, "title": title, "status": "running", "step": "Starting",
                                 "current": None, "total": None, "started": time.time(), "ended": None}
        finished = [key for key, job in state["jobs"].items() if job["status"] != "running"]
        for key in finished[:-7]:
            del state["jobs"][key]
    token = _current.set((session_id, job_id))
    report("Starting task")
    try:
        yield
    except BaseException as error:
        fail(error)
        raise
    else:
        with _lock:
            job = _sessions[session_id]["jobs"][job_id]
            if job["status"] == "running":
                report("Completed")
                job.update(status="completed", ended=time.time())
    finally:
        _current.reset(token)


def report(message, current=None, total=None):
    identity = _current.get()
    if identity is None:
        return  # CLI callers do not require a browser progress session.
    if total is not None and (total < 1 or current is None or not 0 <= current <= total):
        raise ValueError("Invalid progress count")
    message = re.sub(r"hf_[A-Za-z0-9]+", "[redacted]", str(message))
    with _lock:
        state = _sessions[identity[0]]
        job = state["jobs"][identity[1]]
        job.update(step=message, current=current, total=total)
        state["updated"] = time.time()
        count = f" ({current}/{total})" if total is not None else ""
        state["events"].append({"time": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
                                "task": job["title"], "message": message + count})


def fail(error):
    if _current.get() is None:
        return
    report(f"Failed: {error}")
    with _lock:
        session_id, job_id = _current.get()
        _sessions[session_id]["jobs"][job_id].update(status="failed", ended=time.time())


def snapshot(session_id):
    with _lock:
        state = _sessions.get(session_id)
        if state is None:
            return {"jobs": [], "events": []}
        return {"jobs": deepcopy(list(state["jobs"].values())), "events": list(state["events"])}
