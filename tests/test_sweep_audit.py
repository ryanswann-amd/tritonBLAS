"""
Unit tests for tools/sweep_audit.py.

These tests don't require CUDA / triton — they exercise the pure-python
classification logic by building synthetic sweep workspaces in tmp_path
and asserting the bucket each run lands in.
"""

import importlib.util
import json
import os
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "tools" / "sweep_audit.py"


@pytest.fixture(scope="module")
def sweep_audit():
    spec = importlib.util.spec_from_file_location("sweep_audit", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _set_age(p: Path, hours_ago: float) -> None:
    """Set mtime of ``p`` (and every file under it) to ``hours_ago`` ago."""
    target = time.time() - hours_ago * 3600
    for child in [p, *p.rglob("*")]:
        os.utime(child, (target, target))


def _make_run(parent: Path, name: str, *,
              shape=None, manifest=False, log=False) -> Path:
    run = parent / name
    run.mkdir()
    if shape is not None:
        m, n, k = shape
        (run / "config.json").write_text(json.dumps(
            {"M": m, "N": n, "K": k, "dtype": "fp16"}
        ))
    if manifest:
        (run / "manifest.json").write_text("[]")
    if log:
        (run / "run.log").write_text("hello")
    return run


def test_active_run_under_2h(tmp_path, sweep_audit):
    run = _make_run(tmp_path, "fresh", shape=(64, 64, 64), log=True)
    row = sweep_audit.classify(run)
    assert row["bucket"] == "ACTIVE"


def test_complete_run_with_manifest(tmp_path, sweep_audit):
    run = _make_run(tmp_path, "done", shape=(64, 64, 64), manifest=True)
    _set_age(run, hours_ago=10)
    row = sweep_audit.classify(run)
    assert row["bucket"] == "COMPLETE"
    assert row["manifest"] is True


def test_stale_run_24h_no_manifest(tmp_path, sweep_audit):
    run = _make_run(tmp_path, "stale", shape=(64, 64, 64), log=True)
    _set_age(run, hours_ago=30)
    row = sweep_audit.classify(run)
    assert row["bucket"] == "STALE"


def test_zombie_run_72h_no_recent_log(tmp_path, sweep_audit):
    run = _make_run(tmp_path, "dead", shape=(64, 64, 64), log=True)
    _set_age(run, hours_ago=100)
    row = sweep_audit.classify(run)
    assert row["bucket"] == "ZOMBIE"


def test_duplicate_promotion_against_complete(tmp_path, sweep_audit):
    done = _make_run(tmp_path, "done", shape=(128, 128, 256), manifest=True)
    _set_age(done, hours_ago=4)
    dup = _make_run(tmp_path, "dup", shape=(128, 128, 256), log=True)
    _set_age(dup, hours_ago=30)

    rows = [sweep_audit.classify(p) for p in (done, dup)]
    sweep_audit.detect_duplicates(rows)
    by_path = {r["path"]: r for r in rows}
    assert by_path[str(done)]["bucket"] == "COMPLETE"
    assert by_path[str(dup)]["bucket"] == "DUPLICATE"


def test_duplicate_does_not_demote_active(tmp_path, sweep_audit):
    done = _make_run(tmp_path, "done", shape=(128, 128, 256), manifest=True)
    _set_age(done, hours_ago=4)
    live = _make_run(tmp_path, "live", shape=(128, 128, 256), log=True)
    # live is ACTIVE (no age bump)

    rows = [sweep_audit.classify(p) for p in (done, live)]
    sweep_audit.detect_duplicates(rows)
    by_path = {r["path"]: r for r in rows}
    assert by_path[str(live)]["bucket"] == "ACTIVE"  # never DUPLICATE


def test_run_without_config_skipped_for_duplicate(tmp_path, sweep_audit):
    done = _make_run(tmp_path, "done", shape=(128, 128, 256), manifest=True)
    _set_age(done, hours_ago=4)
    no_cfg = _make_run(tmp_path, "no_cfg", shape=None, log=True)
    _set_age(no_cfg, hours_ago=30)

    rows = [sweep_audit.classify(p) for p in (done, no_cfg)]
    sweep_audit.detect_duplicates(rows)
    by_path = {r["path"]: r for r in rows}
    # no shape -> stays STALE, never DUPLICATE
    assert by_path[str(no_cfg)]["bucket"] == "STALE"


def test_discover_runs_single_run(tmp_path, sweep_audit):
    """If --root *is* a single run, it should be returned by itself."""
    run = _make_run(tmp_path, "only", shape=(8, 8, 8), manifest=True)
    discovered = sweep_audit.discover_runs(run)
    assert discovered == [run]


def test_discover_runs_workspace(tmp_path, sweep_audit):
    """If --root contains run subdirs, they should be discovered."""
    a = _make_run(tmp_path, "a", shape=(8, 8, 8), log=True)
    b = _make_run(tmp_path, "b", shape=(16, 16, 16), log=True)
    discovered = sweep_audit.discover_runs(tmp_path)
    assert set(discovered) == {a, b}
