"""Coherent snapshots + transactional restores of the runtime ``data/`` tree.

Used by the maintenance UI (Settings → System → Backups) and by the
updater to snapshot state before applying an update.

Snapshot protocol (manifest v3)
------------------------------
A snapshot is a single ``.zip`` under ``data/core/backups/``. Every byte of
its payload is copied during ONE held read barrier
(:mod:`app.state_coordinator`): managed writers (scheduler ticks, the push
pipeline, marketplace installs, settings POSTs) are paused and drained, JSON
stores are flushed under their locks, and managed files are copied into an
isolated staging generation while SQLite databases go through the online backup
API. The barrier is released as soon as the staging copy exists; SHA-256
hashing and zip compression run on a worker thread afterwards, so writer
downtime is bounded by the copy.

The embedded versioned manifest (``.tesserae-backup.json``) records, for
every file, ``sha256`` / ``size`` / ``mode`` / ``kind`` / ``status`` and
the exclusion rules in force; ``coherent: true`` certifies the payload was
staged inside a barrier. The updater only touches the code tree when that flag
is present.

Restore protocol
-------------
Restore enters persistent maintenance mode and performs *all* validation before
touching live state: zip-slip-safe member paths, manifest hash/size/mode
verification, state-version compatibility, free space, and
``PRAGMA integrity_check`` on every SQLite member. The validated payload lands
in a new generation directory sibling of ``data/`` (persistent volume root
 permitting; otherwise ``data/.generations/``), together with hardlinked
carry-overs of the backups directory and excluded subpaths. A journaled,
rename-atomic per-entry exchange then commits the generation, and the previous
generation is retained on disk for automatic rollback if the next boot fails
to come up (see :func:`recover_pending` / :func:`rollback_unverified`,
called from ``app.main``).

**Backups contain real secrets** (API tokens, MQTT passwords, OAuth
tokens). Treat the ``.zip`` with the same care as ``data/`` itself.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from app import state_coordinator

logger = logging.getLogger(__name__)

BACKUPS_SUBDIR = "core/backups"  # relative to data_root
STAGE_SUBDIR = "core/backups/.stage"  # barrier-held staging generations
META_NAME = ".tesserae-backup.json"

# Manifest schema version. v1 = unversioned originals, v2 added
# ``excluded_subpaths``, v3 adds the per-file manifest, ``coherent`` and
# ``state_version``.
META_VERSION = 3
# Version of the *managed on-disk state layout* (independent of the manifest
# schema). Restores refuse payloads produced by newer state.
STATE_VERSION = 1
SUPPORTED_STATE_VERSIONS: tuple[int, ...] = (1,)

# Journal for generation swaps / boot-time recovery.
RESTORE_STATE_NAME = ".tesserae-restore.json"
GENERATIONS_DIRNAME = ".generations"  # inside-data fallback location
# Extra headroom required before extracting a generation (on top of payload size).
FREE_SPACE_MARGIN_BYTES = 64 * 1024 * 1024

LABEL_MANUAL = "manual"
LABEL_PRE_UPDATE = "pre-update"

# Per-file manifest status / kind codes.
STATUS_INCLUDED = "included"
STATUS_EXCLUDED = "excluded"
KIND_SQLITE = "sqlite"
KIND_JSON = "json"
KIND_FILE = "file"

# Subpaths under data_root whose **regular files are excluded** from a
# snapshot, dotfiles (config like ``.folders.json``) inside them are still
# included so plugin metadata survives a restore. See the v2 module history for
# the rationale (gallery photos, regenerable push cache, Companion personal
# data). Restoring a backup that excluded a subpath does NOT wipe live files
# there: they are carried into the new generation before the exchange.
DEFAULT_EXCLUDED_SUBPATHS: tuple[str, ...] = (
    "plugins/picture_gallery",
    "core/renders",
    "core/companion_personal_data.json",
)

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tesserae-backup")
_jobs: dict[str, BackupJob] = {}
_jobs_lock = threading.Lock()


@dataclass(frozen=True)
class Backup:
    id: str  # filename stem, e.g. "20260530-082145-manual"
    path: Path
    bytes: int
    created_at: float  # unix seconds
    label: str  # "manual" / "pre-update" / user-set
    note: str  # optional context (e.g. SHA going from/to)
    coherent: bool = True
    finalizing: bool = False  # True while the async compression is still running


@dataclass(frozen=True)
class RestoreReport:
    backup_id: str
    generation: str
    entries: tuple[str, ...]
    verified_files: int
    carried_paths: tuple[str, ...] = field(default_factory=tuple)


class BackupError(RuntimeError):
    """Snapshot creation failed (barrier drain, staging I/O)."""


class RestoreError(ValueError):
    """Restore validation or commit failed. Raised *before* the generation
    exchange whenever validation failed, so live state is untouched."""


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def _backups_dir(data_root: Path) -> Path:
    d = data_root / BACKUPS_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _stage_dir(data_root: Path, bid: str) -> Path:
    d = data_root / STAGE_SUBDIR / bid
    d.mkdir(parents=True, exist_ok=True)
    return d


def _snapshot_sqlite(src: Path, dst: Path) -> bool:
    """Copy a SQLite file via the online backup API so a live writer
    can't tear the snapshot. Returns False (caller should fall back to a
    byte copy) when ``src`` isn't a SQLite database."""
    try:
        src_db = sqlite3.connect(str(src))
        try:
            dst_db = sqlite3.connect(str(dst))
            try:
                with dst_db:
                    src_db.backup(dst_db)
            finally:
                dst_db.close()
        finally:
            src_db.close()
        return True
    except sqlite3.DatabaseError:
        return False


def _is_excluded(rel: Path, excluded_subpaths: tuple[str, ...]) -> bool:
    """Whether ``rel`` (relative to data_root) is a regular file inside any
    excluded subpath. Dotfiles within an excluded subpath are kept so
    plugin config / metadata still rides along."""
    if rel.name.startswith("."):
        return False
    rel_posix = rel.as_posix()
    return any(
        rel_posix == prefix or rel_posix.startswith(prefix + "/") for prefix in excluded_subpaths
    )


def _kind_of(rel: Path) -> str:
    suffix = rel.suffix.lower()
    if suffix == ".db":
        return KIND_SQLITE
    if suffix == ".json":
        return KIND_JSON
    return KIND_FILE


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _fsync_dir(path: Path) -> None:
    with contextlib.suppress(OSError):
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    _fsync_dir(tmp.parent)
    tmp.replace(path)
    _fsync_dir(path.parent)


def _safe_member_path(member: str) -> bool:
    """Zip-slip / absolute-path guard for one archive member name."""
    parts = PurePosixPath(member).parts
    if not parts:
        return False
    if any(p in ("..", "") for p in parts):
        return False
    if member.startswith(("/", "\\")):
        return False
    if parts[0].endswith(":"):  # Windows drive prefix, e.g. C:foo
        return False
    return True


# ---------------------------------------------------------------------
# create
# ---------------------------------------------------------------------


class BackupJob:
    """A snapshot whose staging copy is already coherent and sealed behind the
    barrier, with zip finalization running on a worker thread. ``wait()`` for
    the finished :class:`Backup`."""

    def __init__(
        self,
        *,
        bid: str,
        final_path: Path,
        tmp_path: Path,
        stage: Path,
        manifest: dict[str, Any],
        created_at: float,
        label: str,
        note: str,
    ) -> None:
        self.id = bid
        self._final_path = final_path
        self._tmp_path = tmp_path
        self._stage = stage
        self._manifest = manifest
        self._created_at = created_at
        self._label = label
        self._note = note
        self._done = threading.Event()
        self._backup: Backup | None = None
        self._error: BaseException | None = None

    @property
    def ready(self) -> bool:
        return self._done.is_set()

    @property
    def error(self) -> BaseException | None:
        return self._error

    def wait(self, timeout: float | None = None) -> Backup:
        if not self._done.wait(timeout=timeout):
            raise TimeoutError(f"backup {self.id} still finalizing after {timeout}s")
        if self._error is not None:
            raise self._error
        assert self._backup is not None
        return self._backup

    def _finish(self) -> None:
        try:
            with zipfile.ZipFile(self._tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for src in sorted(self._stage.rglob("*")):
                    if src.is_dir():
                        continue
                    zf.write(src, src.relative_to(self._stage).as_posix())
                zf.writestr(META_NAME, json.dumps(self._manifest, indent=2))
            _fsync_dir(self._tmp_path.parent)
            self._tmp_path.replace(self._final_path)
            _fsync_dir(self._final_path.parent)
            self._backup = Backup(
                id=self.id,
                path=self._final_path,
                bytes=self._final_path.stat().st_size,
                created_at=self._created_at,
                label=self._label,
                note=self._note,
                coherent=bool(self._manifest.get("coherent", False)),
            )
        except BaseException as exc:  # finalization failure surfaced by wait()
            with contextlib.suppress(OSError):
                self._tmp_path.unlink()
            self._error = exc
        finally:
            with contextlib.suppress(OSError):
                shutil.rmtree(self._stage, ignore_errors=True)
            with _jobs_lock:
                _jobs.pop(self.id, None)
            self._done.set()


def create_async(
    data_root: Path | str,
    *,
    label: str = LABEL_MANUAL,
    note: str = "",
    excluded_subpaths: tuple[str, ...] = DEFAULT_EXCLUDED_SUBPATHS,
    coordinator: state_coordinator.DataStateCoordinator | None = None,
) -> BackupJob:
    """Take a coherent snapshot and return immediately; zip finalization continues
    on a worker thread.

    The expensive writer-visible work (pause + drain + flush + copy) happens
    synchronously inside the barrier; hashing + compression happen after the barrier
    has been released. Raises :class:`BackupError` if the barrier can't drain
    writers (no incoherent archive is ever produced)."""
    root = Path(data_root)
    out_dir = _backups_dir(root)
    backups_resolved = out_dir.resolve()
    ts = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    safe_label = "".join(c if c.isalnum() else "-" for c in (label or "manual")).strip("-")
    bid = f"{ts}-{safe_label or 'manual'}"
    final = out_dir / f"{bid}.zip"
    tmp = out_dir / f"{bid}.zip.part"
    created_at = time.time()

    coord = coordinator or state_coordinator.coordinator_for(root)
    assert coord is not None
    stage = _stage_dir(root, bid)
    included: list[dict[str, Any]] = []
    excluded_records: list[dict[str, Any]] = []
    barrier_started = time.monotonic()
    try:
        with coord.barrier(reason=f"backup:{bid}"):
            for src in root.rglob("*"):
                if src.is_dir() or not src.exists():
                    continue
                # Skip backups + staging entirely (no recursive nesting).
                try:
                    src.resolve().relative_to(backups_resolved)
                    continue
                except ValueError:
                    pass
                rel = src.relative_to(root)
                # Restore journal + in-data generation fallback live outside the
                # managed payload even when they can't sit sibling of data/.
                if len(rel.parts) == 1 and rel.name in (
                    RESTORE_STATE_NAME,
                    GENERATIONS_DIRNAME,
                ):
                    continue
                if _is_excluded(rel, excluded_subpaths):
                    with contextlib.suppress(OSError):
                        excluded_records.append(
                            {
                                "path": rel.as_posix(),
                                "size": src.stat().st_size,
                                "status": STATUS_EXCLUDED,
                            }
                        )
                    continue
                staged = stage / rel
                staged.parent.mkdir(parents=True, exist_ok=True)
                kind = _kind_of(rel)
                if kind == KIND_SQLITE and _snapshot_sqlite(src, staged):
                    pass
                else:
                    if kind == KIND_SQLITE:
                        # Not really a SQLite file / unreadable header: byte copy.
                        kind = KIND_FILE
                    shutil.copy2(src, staged)
                included.append({"path": rel.as_posix(), "kind": kind})
    except BaseException as exc:
        with contextlib.suppress(OSError):
            shutil.rmtree(stage, ignore_errors=True)
        raise BackupError(f"could not stage a coherent snapshot: {exc}") from exc

    # Barrier released: hash + finalize without holding writers paused.
    for record in included:
        staged_file = stage / record["path"]
        st = staged_file.stat()
        record["sha256"] = _sha256_file(staged_file)
        record["size"] = st.st_size
        record["mode"] = f"{st.st_mode & 0o7777:04o}"
        record["status"] = STATUS_INCLUDED
    excluded_records.sort(key=lambda r: r["path"])
    included.sort(key=lambda r: r["path"])
    manifest: dict[str, Any] = {
        "tool": "tesserae",
        "version": META_VERSION,
        "manifest_version": META_VERSION,
        "state_version": STATE_VERSION,
        "created_at": created_at,
        "label": label,
        "note": note,
        "coherent": True,
        "barrier": {
            "held_s": round(time.monotonic() - barrier_started, 3),
            "files": len(included),
        },
        "excluded_subpaths": list(excluded_subpaths),
        "files": included,
        "excluded": excluded_records,
    }
    job = BackupJob(
        bid=bid,
        final_path=final,
        tmp_path=tmp,
        stage=stage,
        manifest=manifest,
        created_at=created_at,
        label=label,
        note=note,
    )
    with _jobs_lock:
        _jobs[bid] = job
    _executor.submit(job._finish)
    return job


def create(
    data_root: Path | str,
    *,
    label: str = LABEL_MANUAL,
    note: str = "",
    excluded_subpaths: tuple[str, ...] = DEFAULT_EXCLUDED_SUBPATHS,
    coordinator: state_coordinator.DataStateCoordinator | None = None,
) -> Backup:
    """Snapshot ``data/`` into a ``.zip`` under
    ``data/core/backups/`` and block until it has been finalized. Writes to
    a ``.part`` first and renames on success so a crash never leaves a
    partial."""
    return create_async(
        data_root,
        label=label,
        note=note,
        excluded_subpaths=excluded_subpaths,
        coordinator=coordinator,
    ).wait()


# ---------------------------------------------------------------------
# list / get / delete
# ---------------------------------------------------------------------


def _read_meta(path: Path) -> dict[str, Any] | None:
    try:
        with zipfile.ZipFile(path) as zf:
            if META_NAME not in zf.namelist():
                return None
            data = json.loads(zf.read(META_NAME))
    except (zipfile.BadZipFile, OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def list_all(data_root: Path | str) -> list[Backup]:
    """Newest first. Finalizing jobs are surfaced as ``finalizing`` entries."""
    root = Path(data_root)
    out_dir = root / BACKUPS_SUBDIR
    items: list[Backup] = []
    if out_dir.is_dir():
        for f in sorted(out_dir.glob("*.zip"), reverse=True):
            meta = _read_meta(f) or {}
            items.append(
                Backup(
                    id=f.stem,
                    path=f,
                    bytes=f.stat().st_size,
                    created_at=float(meta.get("created_at") or f.stat().st_mtime),
                    label=str(meta.get("label") or "?"),
                    note=str(meta.get("note") or ""),
                    coherent=bool(meta.get("coherent", True)),
                )
            )
    with _jobs_lock:
        jobs = list(_jobs.values())
    for job in jobs:
        if any(b.id == job.id for b in items):
            continue
        items.append(
            Backup(
                id=job.id,
                path=job._final_path,
                bytes=0,
                created_at=job._created_at,
                label=job._label,
                note=job._note,
                coherent=True,
                finalizing=True,
            )
        )
    items.sort(key=lambda b: b.id, reverse=True)
    return items


def get(data_root: Path | str, backup_id: str) -> Backup | None:
    return next((b for b in list_all(Path(data_root)) if b.id == backup_id), None)


def delete(data_root: Path | str, backup_id: str) -> bool:
    backup = get(Path(data_root), backup_id)
    if backup is None:
        return False
    if backup.finalizing:
        # Let finalization land first; the zip appears/disappears consistently.
        with _jobs_lock:
            job = _jobs.get(backup_id)
        if job is not None:
            job.wait(timeout=300)
        backup = get(Path(data_root), backup_id)
        if backup is None:
            return False
    with contextlib.suppress(OSError):
        backup.path.unlink()
    return not backup.path.exists()


# ---------------------------------------------------------------------
# restore: validation
# ---------------------------------------------------------------------


def _open_manifest(zf: zipfile.ZipFile) -> tuple[dict[str, Any], list[str]]:
    names = zf.namelist()
    for name in names:
        if not _safe_member_path(name):
            raise RestoreError(f"unsafe zip member path: {name!r}")
    if META_NAME not in names:
        raise RestoreError("not a Tesserae backup (no manifest)")
    try:
        meta = json.loads(zf.read(META_NAME))
    except (json.JSONDecodeError, OSError) as exc:
        raise RestoreError(f"manifest is not valid JSON: {exc}") from exc
    if not isinstance(meta, dict):
        raise RestoreError("manifest has the wrong shape")
    payload = [n for n in names if n != META_NAME and not n.endswith("/")]
    return meta, payload


def _validate_manifest(meta: dict[str, Any], payload_names: list[str]) -> list[dict[str, Any]]:
    """Check compatibility rules that need no extracted bytes; return the
    per-file records to verify against."""
    version = meta.get("manifest_version", meta.get("version", 1))
    try:
        version_int = int(version)
    except (TypeError, ValueError) as exc:
        raise RestoreError(f"manifest version not readable: {version!r}") from exc
    if version_int > META_VERSION:
        raise RestoreError(
            f"backup manifest v{version_int} is newer than this build supports (v{META_VERSION})"
        )
    state_version = meta.get("state_version")
    if state_version is not None:
        try:
            sv = int(state_version)
        except (TypeError, ValueError) as exc:
            raise RestoreError(f"state version not readable: {state_version!r}") from exc
        if sv not in SUPPORTED_STATE_VERSIONS:
            raise RestoreError(f"backup state version v{sv} is not supported by this build")
    records = meta.get("files")
    if records is None:
        # v1/v2 backup: no per-file manifest, verify by payload presence only.
        return [{"path": n, "status": STATUS_INCLUDED} for n in payload_names]
    if not isinstance(records, list):
        raise RestoreError("manifest files section has the wrong shape")
    paths = [str(r.get("path")) for r in records if isinstance(r, dict)]
    if set(paths) != set(payload_names):
        raise RestoreError("manifest file list does not match zip payload")
    if payload_names and not bool(meta.get("coherent", True)):
        raise RestoreError("backup manifest is not marked coherent; refusing to restore")
    return [r for r in records if isinstance(r, dict)]


def _check_free_space(target_dir: Path, needed_bytes: int) -> None:
    usage = shutil.disk_usage(target_dir)
    required = needed_bytes + FREE_SPACE_MARGIN_BYTES
    if usage.free < required:
        raise RestoreError(
            f"not enough free space to restore: need ~{required // 1024 // 1024} MiB, "
            f"{usage.free // 1024 // 1024} MiB free"
        )


def _verify_sqlite_integrity(path: Path) -> None:
    try:
        conn = sqlite3.connect(str(path))
        try:
            rows = conn.execute("PRAGMA integrity_check").fetchall()
        finally:
            conn.close()
    except sqlite3.DatabaseError as exc:
        raise RestoreError(f"SQLite database unreadable ({path.name}): {exc}") from exc
    ok = rows and all(str(r[0]).lower() == "ok" for r in rows)
    if not ok:
        detail = "; ".join(str(r[0]) for r in rows[:3])
        raise RestoreError(f"SQLite integrity check failed for {path.name}: {detail}")


def _extract_and_verify(
    zf: zipfile.ZipFile,
    records: list[dict[str, Any]],
    *,
    manifest_version: int,
    gen_dir: Path,
) -> int:
    """Extract every payload member into ``gen_dir`` and verify everything the
    manifest asserts. Returns the number of verified files."""
    total = 0
    for record in sorted(records, key=lambda r: str(r.get("path"))):
        rel_posix = str(record["path"])
        rel = Path(*PurePosixPath(rel_posix).parts)
        target = gen_dir / rel
        if not _safe_member_path(rel_posix) or not str(target.resolve()).startswith(
            str(gen_dir.resolve()) + os.sep
        ):
            raise RestoreError(f"unsafe extraction target for {rel_posix!r}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(rel_posix) as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
        st = target.stat()
        expected_size = record.get("size")
        if expected_size is not None and st.st_size != int(expected_size):
            raise RestoreError(
                f"size mismatch for {rel_posix}: manifest {expected_size} != disk {st.st_size}"
            )
        mode = record.get("mode")
        if isinstance(mode, str):
            with contextlib.suppress(ValueError, OSError):
                target.chmod(int(mode, 8))
        expected_hash = record.get("sha256")
        if isinstance(expected_hash, str) and expected_hash:
            actual = _sha256_file(target)
            if actual != expected_hash:
                raise RestoreError(f"sha256 mismatch for {rel_posix}")
        kind = str(record.get("kind") or _kind_of(rel))
        if kind == KIND_SQLITE or rel.suffix.lower() == ".db":
            _verify_sqlite_integrity(target)
        total += 1
    if manifest_version >= 3 and total != len(records):
        raise RestoreError("manifest verification count mismatch")
    return total


# ---------------------------------------------------------------------
# restore: generation directories + carry-over
# ---------------------------------------------------------------------


def _generations_base(root: Path) -> tuple[Path, bool]:
    """Generation tree location: a hidden directory sibling of ``data`` so the
    renames stay atomic on the same filesystem. Falls back to
    ``data/.generations`` when the sibling location isn't writable (read-only
    image layer around a mounted volume, e.g. some container layouts)."""
    sibling = root.parent / f".{root.name}{GENERATIONS_DIRNAME}"
    try:
        sibling.mkdir(parents=True, exist_ok=True)
        (sibling / ".write-probe").write_bytes(b"t")
        (sibling / ".write-probe").unlink()
        return sibling, False
    except OSError:
        inside = root / GENERATIONS_DIRNAME
        inside.mkdir(parents=True, exist_ok=True)
        return inside, True


def _generation_paths(root: Path, gen_id: str) -> tuple[Path, Path, bool]:
    base, inside = _generations_base(root)
    return base / f"gen-{gen_id}-new", base / f"gen-{gen_id}-old", inside


def _link_or_copy(src: Path, dst: Path) -> None:
    """Carry a preserved file into the new generation. Hardlinks (same fs,
    zero extra bytes) first, byte copy fallback (cross-device bind mounts)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _carry_over(root: Path, gen_dir: Path, excluded_subpaths: tuple[str, ...]) -> list[str]:
    """Populate the new generation with files the snapshot deliberately omitted:
    the backups directory (restoring must never delete backups) and the current
    files inside excluded subpaths (gallery photos, render cache). Returns the
    carried relative paths."""
    carried: list[str] = []
    backups = root / BACKUPS_SUBDIR
    stage_resolved = (root / STAGE_SUBDIR).resolve()
    if backups.is_dir():
        for f in backups.rglob("*"):
            if not f.is_file():
                continue
            # Never carry stale staging generations from crashed backups.
            with contextlib.suppress(ValueError):
                f.resolve().relative_to(stage_resolved)
                continue
            rel = f.relative_to(root)
            _link_or_copy(f, gen_dir / rel)
            carried.append(rel.as_posix())
    for prefix in excluded_subpaths:
        src_path = root / prefix
        if src_path.is_file():
            _link_or_copy(src_path, gen_dir / prefix)
            carried.append(prefix)
        elif src_path.is_dir():
            for f in src_path.rglob("*"):
                if f.is_file():
                    rel = f.relative_to(root)
                    if _is_excluded(rel, excluded_subpaths):
                        _link_or_copy(f, gen_dir / rel)
                        carried.append(rel.as_posix())
    carried.sort()
    return carried


# ---------------------------------------------------------------------
# restore: journaled atomic exchange + boot recovery
# ---------------------------------------------------------------------


def _state_path(root: Path) -> Path:
    return root / RESTORE_STATE_NAME


def _write_restore_state(root: Path, state: dict[str, Any]) -> None:
    _atomic_write_text(_state_path(root), json.dumps(state, indent=2))


def _entry_names(gen_dir: Path) -> list[str]:
    return sorted(p.name for p in gen_dir.iterdir() if p.exists())


def _commit_generation(
    root: Path,
    *,
    new_dir: Path,
    retain_dir: Path,
    gen_id: str,
    backup_id: str,
    excluded_subpaths: tuple[str, ...],
    entries: list[str],
) -> RestoreReport:
    """Rename-atomic per-entry exchange, journaled through
    ``RESTORE_STATE_NAME`` so a crash between renames can be completed or
    rolled back at the next boot. The old generation is kept under
    ``retain_dir``."""
    state: dict[str, Any] = {
        "phase": "swapping",
        "gen_id": gen_id,
        "backup_id": backup_id,
        "new_dir": str(new_dir),
        "retain_dir": str(retain_dir),
        "created_at": time.time(),
        "entries": [{"name": name, "phase": "pending"} for name in entries],
    }
    _write_restore_state(root, state)
    for rec in state["entries"]:
        name = str(rec["name"])
        live = root / name
        new = new_dir / name
        retained = retain_dir / name
        if not new.exists():  # pragma: no cover - journaled defensively
            raise RestoreError(f"generation payload missing {name!r}")
        if live.exists():
            retained.parent.mkdir(parents=True, exist_ok=True)
            os.replace(live, retained)
            rec["phase"] = "retained"
            _write_restore_state(root, state)
        os.replace(new, live)
        rec["phase"] = "live"
        _write_restore_state(root, state)
    state["phase"] = "committed"
    state["verified"] = False
    state["excluded_subpaths"] = list(excluded_subpaths)
    _write_restore_state(root, state)
    _fsync_dir(root)
    return RestoreReport(
        backup_id=backup_id,
        generation=gen_id,
        entries=tuple(entries),
        verified_files=len(entries),
    )


def recover_pending(root: Path | str) -> dict[str, Any] | None:
    """Complete or roll back an interrupted generation exchange after a crash.

    Run BEFORE constructing the Flask app on boot. Returns the committed
    (but unverified) state so the caller can decide whether to verify or roll
    back; returns None when there is nothing pending."""
    root = Path(root)
    sp = _state_path(root)
    if not sp.exists():
        return None
    try:
        state = json.loads(sp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.exception("restore journal unreadable; leaving state untouched")
        return None
    if not isinstance(state, dict) or state.get("phase") != "swapping":
        return state if state.get("phase") == "committed" else None

    try:
        new_dir = Path(str(state["new_dir"]))
        retain_dir = Path(str(state["retain_dir"]))
        entries = list(state.get("entries", []))
    except KeyError:
        logger.exception("restore journal missing directory fields; leaving state untouched")
        return None
    rolled_back = False
    for rec in entries:
        name = str(rec.get("name") or "")
        if not name:
            continue
        live = root / name
        new = new_dir / name
        retained = retain_dir / name
        if live.exists():
            continue  # entry landed
        if new.exists():
            # Crashed between old→retained and new→live: finish commit.
            os.replace(new, live)
            rec["phase"] = "live"
            _write_restore_state(root, state)
        elif retained.exists():
            # Crashed with the old generation parked: bring it back.
            os.replace(retained, live)
            rec["phase"] = "rolled_back"
            rolled_back = True
            _write_restore_state(root, state)
        else:  # pragma: no cover - both sides gone, nothing safe to do
            raise RestoreError(f"cannot recover entry {name!r}: neither generation exists")
    if rolled_back:
        state["phase"] = "aborted"
        _write_restore_state(root, state)
        logger.error("restore interrupted mid-exchange; rolled the old generation back")
        return state
    state["phase"] = "committed"
    state["verified"] = False
    _write_restore_state(root, state)
    return state


def rollback_unverified(root: Path | str) -> str | None:
    """Return live state to the retained old generation after a failed first
    boot. Returns the generation id that was rolled back, or None when there
    was no unverified committed restore."""
    root = Path(root)
    sp = _state_path(root)
    if not sp.exists():
        return None
    try:
        state = json.loads(sp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if state.get("phase") != "committed" or state.get("verified"):
        return None
    new_dir = Path(str(state["new_dir"]))
    retain_dir = Path(str(state["retain_dir"]))
    state["phase"] = "rolling_back"
    _write_restore_state(root, state)
    for rec in reversed(list(state.get("entries", []))):
        name = str(rec.get("name") or "")
        if not name:
            continue
        live = root / name
        new = new_dir / name
        retained = retain_dir / name
        if live.exists():
            new.parent.mkdir(parents=True, exist_ok=True)
            os.replace(live, new)
        if retained.exists():
            os.replace(retained, live)
    state["phase"] = "aborted"
    state["verified"] = False
    _write_restore_state(root, state)
    gen_id = str(state.get("gen_id") or "?")
    logger.error("new generation failed verification; rolled back to generation %s", gen_id)
    return gen_id


def mark_verified(root: Path | str) -> None:
    """Mark the committed generation as booted successfully. The retained old
    generation stays on disk as a rollback artifact."""
    root = Path(root)
    sp = _state_path(root)
    if not sp.exists():
        return
    try:
        state = json.loads(sp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if state.get("phase") == "committed":
        state["verified"] = True
        _write_restore_state(root, state)


# ---------------------------------------------------------------------
# restore: public entry point
# ---------------------------------------------------------------------


def restore(
    data_root: Path | str,
    backup_id: str,
    *,
    coordinator: state_coordinator.DataStateCoordinator | None = None,
) -> RestoreReport:
    """Validate the snapshot, build a new generation sibling of ``data/``,
    and atomically exchange it for the current managed state.

    Nothing live is deleted or replaced until every check (member paths,
    manifest hashes, state version, free space, SQLite integrity) has passed,
    so a failed validation raises :class:`RestoreError` with the current state
    untouched. The caller restarts the process afterwards; the next boot
    verifies the generation and auto-rolls back if the app fails to
    construct."""
    root = Path(data_root)
    backup = get(root, backup_id)
    if backup is None:
        raise FileNotFoundError(backup_id)
    if backup.finalizing:
        raise RestoreError("backup is still finalizing, try again in a moment")

    coord = coordinator or state_coordinator.coordinator_for(root)
    assert coord is not None
    with coord.maintenance(reason=f"restore:{backup_id}"):
        with zipfile.ZipFile(backup.path) as zf:
            if zf.testzip() is not None:
                raise RestoreError("zip archive is corrupt (CRC failure)")
            meta, payload_names = _open_manifest(zf)
            records = _validate_manifest(meta, payload_names)
            excluded = tuple(meta.get("excluded_subpaths") or ())
            manifest_version = int(meta.get("manifest_version", meta.get("version", 1)))

            gen_id = f"{backup_id}-{int(time.time())}"
            new_dir, retain_dir, _inside = _generation_paths(root, gen_id)
            if new_dir.exists() or retain_dir.exists():
                raise RestoreError("generation directory already exists; refusing to overwrite")
            total_bytes = sum(int(r.get("size", 0)) for r in records)
            _check_free_space(new_dir.parent, total_bytes)
            new_dir.mkdir(parents=True)
            try:
                verified = _extract_and_verify(
                    zf, records, manifest_version=manifest_version, gen_dir=new_dir
                )
                carried = _carry_over(root, new_dir, excluded)
                entries = _entry_names(new_dir)
                report = _commit_generation(
                    root,
                    new_dir=new_dir,
                    retain_dir=retain_dir,
                    gen_id=gen_id,
                    backup_id=backup_id,
                    excluded_subpaths=excluded,
                    entries=entries,
                )
            except BaseException:
                with contextlib.suppress(OSError):
                    shutil.rmtree(new_dir, ignore_errors=True)
                raise
        report = RestoreReport(
            backup_id=report.backup_id,
            generation=report.generation,
            entries=report.entries,
            verified_files=verified,
            carried_paths=tuple(carried),
        )
        logger.info(
            "restore: committed generation %s (%d files verified, %d carried paths)",
            report.generation,
            verified,
            len(carried),
        )
        return report
