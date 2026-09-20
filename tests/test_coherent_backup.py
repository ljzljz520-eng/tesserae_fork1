"""Coherent snapshot + transactional restore: coordinator barriers,
versioned manifests, pre-touch validation rejection, generation
swap journal + boot-time rollback/recovery."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import zipfile
from pathlib import Path

import pytest

from app import backup as bk
from app.state_coordinator import (
    BarrierTimeout,
    DataStateCoordinator,
    DataStatePaused,
)


def _seed(root: Path) -> None:
    (root / "core").mkdir(parents=True, exist_ok=True)
    (root / "core" / "settings.json").write_text('{"app":{"x":1}}')
    (root / "core" / "pages.json").write_text('{"pages":[]}')
    db = sqlite3.connect(root / "core" / "events.db")
    db.execute("CREATE TABLE t (k TEXT, v INTEGER)")
    db.executemany("INSERT INTO t VALUES (?,?)", [("a", 1), ("b", 2)])
    db.commit()
    db.close()


# -- coordinator ----------------------------------------------------------


def test_barrier_pauses_writers_flushes_and_drains(tmp_path: Path) -> None:
    coord = DataStateCoordinator(tmp_path)
    flushed: list[int] = []
    coord.register_flusher("f", lambda: flushed.append(1))

    entered = threading.Event()
    release = threading.Event()

    def worker() -> None:
        with coord.write_session("writer", block=True):
            entered.set()
            release.wait(5.0)

    thread = threading.Thread(target=worker)
    thread.start()
    assert entered.wait(5.0)

    held = threading.Event()

    def hold_barrier() -> None:
        with coord.barrier(reason="test"):
            held.set()
            release.wait(5.0)

    barrier_thread = threading.Thread(target=hold_barrier)
    barrier_thread.start()
    # The barrier must wait for the in-flight writer instead of snapshotting
    # under it.
    assert not held.wait(0.3)
    # Non-blocking writers fail fast while it drains / holds.
    with pytest.raises(DataStatePaused):
        with coord.write_session("http", block=False):
            pass
    assert coord.is_blocking()
    release.set()
    barrier_thread.join(5.0)
    thread.join(5.0)
    # Flusher ran exactly once, during the barrier.
    assert flushed == [1]
    assert not coord.is_blocking()
    # Writers flow again afterwards.
    with coord.write_session("writer", block=True, timeout=2.0):
        pass


def test_barrier_times_out_when_a_writer_never_drains(tmp_path: Path) -> None:
    coord = DataStateCoordinator(tmp_path)
    stuck = threading.Event()

    def worker() -> None:
        with coord.write_session("stuck", block=True):
            stuck.set()
            threading.Event().wait(30.0)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    assert stuck.wait(5.0)
    with pytest.raises(BarrierTimeout):
        with coord.barrier(reason="test", timeout_s=0.2):
            pass


def test_two_barriers_cannot_overlap(tmp_path: Path) -> None:
    coord = DataStateCoordinator(tmp_path)
    with coord.barrier(reason="a"):
        with pytest.raises(DataStatePaused):
            with coord.barrier(reason="b"):
                pass


# -- versioned manifest ---------------------------------------------------


def test_manifest_is_v3_coherent_and_per_file_hashed(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _seed(root)
    backup = bk.create(root)

    assert backup.coherent is True
    with zipfile.ZipFile(backup.path) as zf:
        meta = json.loads(zf.read(bk.META_NAME))
        payload = set(zf.namelist()) - {bk.META_NAME}

    assert meta["coherent"] is True
    assert meta["manifest_version"] == 3
    assert meta["version"] == 3
    assert meta["state_version"] == 1
    assert meta["barrier"]["files"] == len(payload)
    assert isinstance(meta["barrier"]["held_s"], (int, float))

    files = {str(f["path"]): f for f in meta["files"]}
    assert set(files) == payload
    for record in files.values():
        assert record["status"] == bk.STATUS_INCLUDED
        assert len(str(record["sha256"])) == 64
        assert int(record["size"]) >= 0
        assert int(str(record["mode"]), 8) > 0
        assert record["kind"] in (bk.KIND_SQLITE, bk.KIND_JSON, bk.KIND_FILE)
    assert files["core/events.db"]["kind"] == bk.KIND_SQLITE
    assert files["core/settings.json"]["kind"] == bk.KIND_JSON
    excluded_paths = {e["path"] for e in meta["excluded"]}
    assert "plugins/picture_gallery" not in excluded_paths  # empty here
    assert "core/companion_personal_data.json" in meta["excluded_subpaths"]


# -- restore validation must refuse before touching live state ------------


def _tampered_copy(backup_path: Path, *, build) -> None:
    """Rewrite the zip through ``build(members)`` where members maps
    member name -> bytes, then atomically replace the archive."""
    with zipfile.ZipFile(backup_path) as zin:
        members = {info.filename: zin.read(info.filename) for info in zin.infolist()}
    members = build(members)
    tmp = backup_path.with_suffix(".zip.alt")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for name, data in members.items():
            zout.writestr(name, data)
    tmp.replace(backup_path)


def test_restore_rejects_hash_mismatch_and_keeps_live_state(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _seed(root)
    backup = bk.create(root)
    original = (root / "core" / "settings.json").read_text()

    _tampered_copy(
        backup.path,
        build=lambda members: {
            name: (b'{"app":{"x":42}}' if name == "core/settings.json" else data)
            for name, data in members.items()
        },
    )

    with pytest.raises(bk.RestoreError, match="sha256|size"):
        bk.restore(root, backup.id)
    assert (root / "core" / "settings.json").read_text() == original


def test_restore_rejects_corrupt_sqlite_and_keeps_live_state(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _seed(root)
    backup = bk.create(root)

    def _corrupt(members: dict[str, bytes]) -> dict[str, bytes]:
        # Keep the valid 100-byte header + original length (so size
        # verification passes) but zero the pages, then re-record the
        # payload hash in the manifest so hash verification can't mask
        # the failure: the independent PRAGMA integrity_check must still
        # reject the image.
        db = members["core/events.db"]
        bad = db[:100] + b"\x00" * (len(db) - 100)
        members["core/events.db"] = bad
        meta = json.loads(members[bk.META_NAME])
        for record in meta["files"]:
            if record["path"] == "core/events.db":
                record["sha256"] = hashlib.sha256(bad).hexdigest()
        members[bk.META_NAME] = json.dumps(meta).encode()
        return members

    _tampered_copy(backup.path, build=_corrupt)

    with pytest.raises(bk.RestoreError, match="integrity|malformed"):
        bk.restore(root, backup.id)
    conn = sqlite3.connect(root / "core" / "events.db")
    try:
        assert conn.execute("SELECT count(*) FROM t").fetchone()[0] == 2
    finally:
        conn.close()


def test_restore_rejects_zip_slip_member(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _seed(root)
    backup = bk.create(root)

    def inject(members: dict[str, bytes]) -> dict[str, bytes]:
        members["../evil.txt"] = b"pwn"
        return members

    _tampered_copy(backup.path, build=inject)
    with pytest.raises(bk.RestoreError, match="unsafe"):
        bk.restore(root, backup.id)
    assert not (root.parent / "evil.txt").exists()


def test_restore_rejects_manifest_marked_incoherent(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _seed(root)
    backup = bk.create(root)
    original = (root / "core" / "settings.json").read_text()

    def flip(members: dict[str, bytes]) -> dict[str, bytes]:
        meta = json.loads(members[bk.META_NAME])
        meta["coherent"] = False
        members[bk.META_NAME] = json.dumps(meta).encode()
        return members

    _tampered_copy(backup.path, build=flip)
    with pytest.raises(bk.RestoreError, match="coherent"):
        bk.restore(root, backup.id)
    assert (root / "core" / "settings.json").read_text() == original


def test_restore_rejects_unsupported_state_version(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _seed(root)
    backup = bk.create(root)

    def bump(members: dict[str, bytes]) -> dict[str, bytes]:
        meta = json.loads(members[bk.META_NAME])
        meta["state_version"] = 999
        members[bk.META_NAME] = json.dumps(meta).encode()
        return members

    _tampered_copy(backup.path, build=bump)
    with pytest.raises(bk.RestoreError, match="state version"):
        bk.restore(root, backup.id)


# -- generation journal: rollback + recovery ------------------------------


def test_committed_unverified_generation_rolls_back_then_verifies(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _seed(root)
    backup = bk.create(root)

    # Post-backup mutation, parked in the retained old generation by restore.
    (root / "core" / "settings.json").write_text('{"app":{"x":999}}')
    bk.restore(root, backup.id)
    assert (root / "core" / "settings.json").read_text() == '{"app":{"x":1}}'

    state_path = root / bk.RESTORE_STATE_NAME
    state = json.loads(state_path.read_text())
    assert state["phase"] == "committed"
    assert state["verified"] is False
    top_entries = {e["name"] for e in state["entries"]}
    assert "core" in top_entries

    # First boot failed: automatic rollback to the retained generation.
    gen_id = bk.rollback_unverified(root)
    assert gen_id == state["gen_id"]
    assert (root / "core" / "settings.json").read_text() == '{"app":{"x":999}}'
    assert json.loads(state_path.read_text())["phase"] == "aborted"

    # Nothing left to roll back afterwards.
    assert bk.rollback_unverified(root) is None


def test_verified_generation_is_kept(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _seed(root)
    backup = bk.create(root)
    (root / "core" / "settings.json").write_text('{"app":{"x":7}}')
    bk.restore(root, backup.id)

    bk.mark_verified(root)
    state = json.loads((root / bk.RESTORE_STATE_NAME).read_text())
    assert state["phase"] == "committed"
    assert state["verified"] is True
    # A healthy boot must not auto-roll back.
    assert bk.rollback_unverified(root) is None
    assert (root / "core" / "settings.json").read_text() == '{"app":{"x":1}}'


def test_recover_pending_completes_an_interrupted_exchange(tmp_path: Path) -> None:
    """Crash window: the live entry was parked into retain/ and the new
    generation had not yet been renamed into place. Recovery completes
    the commit (lands the candidate), re-marking it unverified."""
    root = tmp_path / "data"
    _seed(root)
    backup = bk.create(root)
    (root / "core" / "settings.json").write_text('{"app":{"x":999}}')
    bk.restore(root, backup.id)

    state = json.loads((root / bk.RESTORE_STATE_NAME).read_text())
    new_dir = Path(str(state["new_dir"]))
    retain_dir = Path(str(state["retain_dir"]))
    core_retained = retain_dir / "core"
    core_live = root / "core"

    # Rewind the filesystem + journal to the crash instant: live "core"
    # (the restored A generation) back in new/, old B parked in retain/.
    core_live.rename(new_dir / "core")
    assert core_retained.exists()
    state["phase"] = "swapping"
    for rec in state["entries"]:
        rec["phase"] = "live" if rec["name"] != "core" else "pending"
    (root / bk.RESTORE_STATE_NAME).write_text(json.dumps(state, indent=2))

    recovered = bk.recover_pending(root)
    assert recovered is not None
    assert recovered["phase"] == "committed"
    assert recovered["verified"] is False
    # The candidate generation landed.
    assert (root / "core" / "settings.json").read_text() == '{"app":{"x":1}}'


def test_recover_pending_rolls_back_when_only_retained_exists(tmp_path: Path) -> None:
    """Defensive branch: live and new are both gone, only the retained
    old generation survived - bring it back and abort the restore."""
    root = tmp_path / "data"
    _seed(root)
    backup = bk.create(root)
    (root / "core" / "settings.json").write_text('{"app":{"x":999}}')
    bk.restore(root, backup.id)

    state = json.loads((root / bk.RESTORE_STATE_NAME).read_text())

    # Live "core" vanished and the candidate was never placed either;
    # the retained B generation is all recovery has to work with.
    (root / "core").rename(root / "_core_gone")
    state["phase"] = "swapping"
    for rec in state["entries"]:
        rec["phase"] = "live" if rec["name"] != "core" else "pending"
    (root / bk.RESTORE_STATE_NAME).write_text(json.dumps(state, indent=2))

    recovered = bk.recover_pending(root)
    assert recovered is not None
    assert recovered["phase"] == "aborted"
    assert (root / "core" / "settings.json").read_text() == '{"app":{"x":999}}'
