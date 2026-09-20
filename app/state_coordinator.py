"""Data-state coordination: a process-wide barrier that makes managed-state
snapshots coherent across files.

The backup pipeline needs an instant at which **every** managed file (JSON
stores, the device manifest, marketplace state, plugin data, SQLite databases)
belongs to the same logical moment. Walking ``data/`` file by file does not
provide that: the scheduler thread, the push pipeline, marketplace installs and
settings POSTs keep mutating independent files between visits, so a zip can
combine a new ``pages.json`` with an older device / deck file.

The coordinator closes that window with a **persistent read barrier**:

1. ``barrier()`` (backup) / ``maintenance()`` (restore) flips a paused flag.
2. New managed-write sessions started after that point either wait in a queue
   (background loops) or fail fast with :class:`DataStatePaused` (HTTP layer
   turns that into a 503).
3. The barrier drains every writer already in flight (session counters + registered
   component locks such as the push pipeline lock) and flushes every registered
   JSON store under its own lock + fsync.
4. The caller copies the managed tree into a staging generation (and runs the
   SQLite online backups) **while the barrier is held**; hashing and zip
   compression happen after release, so writer downtime is bounded by the copy.

Stores register a flusher (``register_store`` knows the ``_lock`` +
atomic-rename convention used across :mod:`app.state`) and long-lived writer
components register their lock (``register_lock``) so the barrier can drain them.
Writers whose critical section isn't naturally lock-shaped (the scheduler tick)
wrap their work in :meth:`DataStateCoordinator.write_session`.

mypy --strict applies to this module, see pyproject.toml.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Max time a barrier waits for in-flight writers (a full frame push, the
# longest managed operation, can run ~105 s with renderer timeouts) before it
# refuses to produce a snapshot. A timed-out backup fails loudly instead of
# producing an incoherent manifest.
DEFAULT_DRAIN_TIMEOUT_S = 180.0


class DataStatePaused(RuntimeError):
    """Raised by a non-blocking :meth:`write_session` while a barrier or
    maintenance window is held. HTTP callers translate this into a 503."""


class BarrierTimeout(TimeoutError):
    """A barrier could not drain in-flight writers / acquire component locks
    within the configured timeout."""


Flusher = Callable[[], None]
LockLike = Any  # threading.Lock / RLock - only acquire/release is needed


def _fsync_path(path: Path) -> None:
    """fsync a file and (best effort) its parent directory so an atomic
    tmp+rename write is durable before we snapshot its directory. Missing files
    are tolerated (a store may simply never have saved)."""
    if path.exists():
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    with contextlib.suppress(OSError):
        parent = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)


class _Component:
    """A long-lived writer serialised by an existing lock (e.g. the push
    pipeline). ``provider`` resolves the (possibly rebuilt) lock at barrier
    time; ``drain`` adds custom quiesce logic on top of the lock."""

    __slots__ = ("name", "lock", "provider", "drain")

    def __init__(
        self,
        name: str,
        *,
        lock: LockLike | None,
        provider: Callable[[], LockLike | None] | None,
        drain: Callable[[], None] | None,
    ) -> None:
        self.name = name
        self.lock = lock
        self.provider = provider
        self.drain = drain

    def resolve_lock(self) -> LockLike | None:
        if self.provider is not None:
            try:
                return self.provider()
            except Exception:
                logger.exception("coordinator: lock provider for %r failed", self.name)
                return None
        return self.lock


class DataStateCoordinator:
    """Per-data-root coordinator. One instance per live ``data/`` tree,
    obtained by :func:`coordinator_for`; standalone callers that never got one
    registered fall back to a bare instance (barrier still serialises file
    copies, there are just no live writers to drain)."""

    def __init__(self, data_root: Path | str) -> None:
        self.data_root = Path(data_root)
        self._cond = threading.Condition()
        self._paused = False
        self._maintenance = False
        self._reason = ""
        self._active: dict[str, int] = {}
        self._flushers: dict[str, Flusher] = {}
        self._components: dict[str, _Component] = {}

    # -- registration -----------------------------------------------------

    def register_flusher(self, name: str, flusher: Flusher) -> None:
        """Register ``flusher()`` to run under the barrier so the file it
        owns is durable on disk before the copy. Must be idempotent and
        non-blocking apart from the store's own lock."""
        self._flushers[name] = flusher

    def register_store(self, name: str, store: Any, path: Path | str) -> None:
        """Register a JSON store following the house convention (``_lock`` plus
        atomic tmp+rename writes). All such stores are write-through, so the
        generated flusher only needs to take the store lock and fsync the
        file, which guarantees no rename is mid-flight and durability of the
        bytes we copy."""
        target = Path(path)
        lock = getattr(store, "_lock", None)

        def _flush_store() -> None:
            if lock is not None:
                lock.acquire()
                try:
                    _fsync_path(target)
                finally:
                    lock.release()
            else:
                _fsync_path(target)

        self.register_flusher(name, _flush_store)

    def register_lock(
        self,
        name: str,
        lock: LockLike | None = None,
        *,
        provider: Callable[[], LockLike | None] | None = None,
        drain: Callable[[], None] | None = None,
    ) -> None:
        """Register a writer component whose in-flight work is serialised by
        ``lock`` (or the lock ``provider`` returns at barrier time, for
        components that get rebuilt, like the push manager). The barrier holds
        the lock for its duration, which both drains the writer in flight and
        blocks its next run."""
        self._components[name] = _Component(name, lock=lock, provider=provider, drain=drain)

    # -- writer side ------------------------------------------------------

    @contextlib.contextmanager
    def write_session(
        self,
        component: str,
        *,
        block: bool = True,
        timeout: float | None = None,
    ) -> Iterator[None]:
        """Gate one managed-write critical section.

        * ``block=True`` (background loops): wait out the barrier, then run.
        * ``block=False`` (request threads): raise
          :class:`DataStatePaused` immediately so the caller can 503/skip.

        The component is counted as "in flight" so an entering barrier
        waits for the section to finish before snapshotting."""
        with self._cond:
            if self._paused or self._maintenance:
                if not block:
                    raise DataStatePaused(self._reason or "data state paused")
                if not self._cond.wait_for(
                    lambda: not self._paused and not self._maintenance, timeout=timeout
                ):
                    raise DataStatePaused(f"still paused after {timeout}s: {self._reason}")
            self._active[component] = self._active.get(component, 0) + 1
        try:
            yield
        finally:
            with self._cond:
                count = self._active.get(component, 0) - 1
                if count <= 0:
                    self._active.pop(component, None)
                else:
                    self._active[component] = count
                self._cond.notify_all()

    def is_blocking(self) -> bool:
        """Whether new managed writes should currently be rejected. The HTTP
        layer polls this per mutating request."""
        with self._cond:
            return self._paused or self._maintenance

    # -- barrier side -----------------------------------------------------

    def _drain_sessions(self, timeout_s: float) -> None:
        import time

        deadline = time.monotonic() + timeout_s
        with self._cond:
            while self._active:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise BarrierTimeout(
                        f"managed writers still in flight after {timeout_s:.0f}s: "
                        + ", ".join(sorted(self._active))
                    )
                self._cond.wait(timeout=min(1.0, remaining))

    def _acquire_component_locks(self, timeout_s: float) -> list[tuple[str, LockLike]]:
        import time

        held: list[tuple[str, LockLike]] = []
        try:
            for name, component in list(self._components.items()):
                deadline = time.monotonic() + timeout_s
                while True:
                    lock = component.resolve_lock()
                    if lock is None:
                        break  # component currently absent, nothing to drain
                    if lock.acquire(blocking=True, timeout=max(0.0, deadline - time.monotonic())):
                        held.append((name, lock))
                        break
                    raise BarrierTimeout(f"component {name!r} did not quiesce in time")
                if component.drain is not None:
                    component.drain()
        except BaseException:
            for _, lock in reversed(held):
                with contextlib.suppress(RuntimeError):
                    lock.release()
            raise
        return held

    def _flush_all(self) -> None:
        for name, flusher in list(self._flushers.items()):
            try:
                flusher()
            except Exception:
                logger.exception("coordinator: flusher %r failed", name)
                raise

    @contextlib.contextmanager
    def _barrier(self, *, maintenance: bool, reason: str, timeout_s: float) -> Iterator[None]:
        import time

        with self._cond:
            if self._paused or self._maintenance:
                raise DataStatePaused(f"another barrier is active ({self._reason})")
            self._paused = True
            if maintenance:
                self._maintenance = True
            self._reason = reason
            self._cond.notify_all()
        held: list[tuple[str, LockLike]] = []
        start = time.monotonic()
        try:
            self._drain_sessions(timeout_s)
            budget = max(1.0, timeout_s - (time.monotonic() - start))
            held = self._acquire_component_locks(budget)
            # Stores last: a flusher taking a store lock must land after any
            # writer session that held that lock has drained above.
            self._flush_all()
            yield
        finally:
            for _, lock in reversed(held):
                with contextlib.suppress(RuntimeError):
                    lock.release()
            with self._cond:
                self._paused = False
                self._maintenance = False
                self._reason = ""
                self._cond.notify_all()

    @contextlib.contextmanager
    def barrier(
        self,
        *,
        reason: str,
        timeout_s: float = DEFAULT_DRAIN_TIMEOUT_S,
    ) -> Iterator[None]:
        """Short read barrier for taking a snapshot. Managed writers are paused,
        drained and flushed for the duration; everything resumes on exit even if
        the staged copy raised."""
        with self._barrier(maintenance=False, reason=reason, timeout_s=timeout_s):
            yield

    @contextlib.contextmanager
    def maintenance(
        self,
        *,
        reason: str,
        timeout_s: float = DEFAULT_DRAIN_TIMEOUT_S,
    ) -> Iterator[None]:
        """Persistent barrier for a restore. Same guarantees as
        :meth:`barrier`; kept as a distinct name (and surfaced through
        :meth:`is_blocking`) so metrics / HTTP layers can advertise the
        maintenance window."""
        with self._barrier(maintenance=True, reason=reason, timeout_s=timeout_s):
            yield


# -- per-data-root registry ----------------------------------------------

_registry: dict[str, DataStateCoordinator] = {}
_registry_lock = threading.Lock()


def coordinator_for(data_root: Path | str, *, create: bool = True) -> DataStateCoordinator | None:
    """The process-wide coordinator for ``data_root`` (resolved, so symlinks /
    ``.`` spellings map to one instance). Creates a bare coordinator when
    absent; pass ``create=False`` to only look one up."""
    key = str(Path(data_root).resolve())
    with _registry_lock:
        coord = _registry.get(key)
        if coord is None and create:
            coord = DataStateCoordinator(key)
            _registry[key] = coord
        return coord


def register_coordinator(coordinator: DataStateCoordinator) -> None:
    with _registry_lock:
        _registry[str(Path(coordinator.data_root).resolve())] = coordinator
