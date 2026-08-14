"""Background work for the desktop UI.

Tkinter is single-threaded: anything slow on the main thread freezes the
window. Training runs for minutes and evaluation for seconds, so both run on a
worker thread and report back through a queue that the UI drains on a timer.

A thread (rather than a process) is deliberate. The trainer already fans work
out to a ``multiprocessing.Pool``, so the thread itself is mostly idle
coordination, and sharing the trainer object makes cooperative stop and
progress reporting straightforward.
"""

from __future__ import annotations

import queue
import threading
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


@dataclass
class JobEvent:
    """A message from a background job to the UI."""
    kind: str                      # "progress" | "log" | "done" | "error"
    payload: Any = None
    job: str = ""


class JobRunner:
    """Runs one background job at a time and queues its events.

    The UI polls :meth:`drain` from the Tk event loop; nothing here touches
    widgets, which keeps all widget access on the main thread.
    """

    def __init__(self) -> None:
        self._events: "queue.Queue[JobEvent]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._cancel = threading.Event()
        self._current: str = ""

    # -- state -------------------------------------------------------------

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def current_job(self) -> str:
        return self._current if self.busy else ""

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    # -- control -----------------------------------------------------------

    def start(self, name: str, target: Callable[["JobContext"], Any]) -> bool:
        """Start ``target`` on a worker thread.

        Args:
            name: Job label, echoed on every event.
            target: Callable taking a :class:`JobContext`. Its return value is
                delivered as the payload of the final ``done`` event.

        Returns:
            False if a job is already running, True otherwise.
        """
        if self.busy:
            return False

        self._cancel.clear()
        self._current = name
        context = JobContext(name=name, runner=self)

        def run() -> None:
            try:
                result = target(context)
                self._events.put(JobEvent("done", result, name))
            except Exception as exc:  # surfaced in the UI, not swallowed
                self._events.put(JobEvent(
                    "error",
                    {"message": str(exc), "traceback": traceback.format_exc()},
                    name,
                ))

        self._thread = threading.Thread(target=run, name=f"crp-{name}",
                                        daemon=True)
        self._thread.start()
        return True

    def cancel(self) -> None:
        """Ask the running job to stop.

        Cooperative: the job decides where it is safe to stop. Training checks
        between generations, so a stop takes effect once the current
        generation finishes rather than mid-tournament.
        """
        self._cancel.set()

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    # -- events ------------------------------------------------------------

    def post(self, event: JobEvent) -> None:
        self._events.put(event)

    def drain(self, limit: int = 200):
        """Pop up to ``limit`` pending events. Called from the Tk event loop.

        Bounded so a burst of log lines cannot stall the UI thread.
        """
        events = []
        for _ in range(limit):
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                break
        return events


@dataclass
class JobContext:
    """Handle given to a background job for reporting and cancellation."""
    name: str
    runner: JobRunner
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def cancelled(self) -> bool:
        return self.runner.cancelled

    def progress(self, payload: Any) -> None:
        self.runner.post(JobEvent("progress", payload, self.name))

    def log(self, message: str) -> None:
        self.runner.post(JobEvent("log", message, self.name))
