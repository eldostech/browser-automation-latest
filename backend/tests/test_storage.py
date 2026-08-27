"""Artifact storage: one flag, two backends, and rows that outlive the choice."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from storage import (
    LocalStorage,
    S3_SCHEME,
    S3Storage,
    StorageError,
    artifact_key,
    build_storage,
    storage_for,
)
from tests.conftest import test_settings


# --- local -----------------------------------------------------------------


async def test_local_round_trip(tmp_path: Path):
    storage = LocalStorage(tmp_path)
    stored = await storage.put(
        artifact_key("run1", "abc", ".png"), b"\x89PNG-bytes", content_type="image/png"
    )

    assert stored.bytes == len(b"\x89PNG-bytes")
    assert Path(stored.locator).read_bytes() == b"\x89PNG-bytes"
    assert await storage.get(stored.locator) == b"\x89PNG-bytes"
    # Nothing to redirect to: this process serves the file itself.
    assert storage.presigned_url(stored.locator) is None


async def test_a_key_cannot_escape_the_artifacts_directory(tmp_path: Path):
    """The keys are ours, but treating them as untrusted costs nothing."""
    storage = LocalStorage(tmp_path / "artifacts")
    with pytest.raises(StorageError, match="outside the artifacts directory"):
        await storage.put("../../etc/passwd", b"x", content_type="text/plain")


async def test_reading_a_missing_file_raises_storage_error(tmp_path: Path):
    storage = LocalStorage(tmp_path)
    with pytest.raises(StorageError, match="could not read"):
        await storage.get(str(tmp_path / "not-here.png"))


# --- selection -------------------------------------------------------------


def test_the_flag_chooses_the_backend(tmp_path: Path):
    local = build_storage(test_settings(artifacts_dir=str(tmp_path)))
    assert local.name == "local"

    s3 = build_storage(
        test_settings(storage_backend="s3", s3_bucket="shots", artifacts_dir=str(tmp_path))
    )
    assert s3.name == "s3"


def test_s3_without_a_bucket_is_refused_at_construction(tmp_path: Path):
    """Fail at startup, not on the first screenshot of the first run."""
    with pytest.raises(StorageError, match="S3_BUCKET is not set"):
        build_storage(test_settings(storage_backend="s3", artifacts_dir=str(tmp_path)))


def test_an_old_local_row_still_reads_after_switching_to_s3(tmp_path: Path):
    """A deployment that switches backends must not orphan what it has.

    Reading follows the row's locator rather than the current setting, so
    yesterday's screenshots keep working after today's change.
    """
    settings = test_settings(
        storage_backend="s3", s3_bucket="shots", artifacts_dir=str(tmp_path)
    )
    configured = build_storage(settings)

    for_old_row = storage_for(str(tmp_path / "run1" / "a.png"), configured, settings)
    assert for_old_row.name == "local"

    for_new_row = storage_for(f"{S3_SCHEME}shots/run1/a.png", configured, settings)
    assert for_new_row.name == "s3"


def test_keys_are_identical_on_both_backends():
    """The key is the backend-independent part; only the locator differs."""
    assert artifact_key("run1", "abc", ".png") == "run1/abc.png"


# --- s3, without a bucket --------------------------------------------------


def test_s3_locators_are_prefixed_and_parsed_symmetrically(tmp_path: Path):
    storage = S3Storage(
        test_settings(storage_backend="s3", s3_bucket="shots", s3_prefix="runs")
    )
    locator = storage._locator("run1/a.png")  # noqa: SLF001 - the format is the contract

    assert locator == f"{S3_SCHEME}shots/runs/run1/a.png"
    assert storage._split(locator) == ("shots", "runs/run1/a.png")  # noqa: SLF001


# --- logging ---------------------------------------------------------------


def test_logs_are_written_to_a_file_as_json(tmp_path: Path):
    """The log directory was never created because nothing wrote to it."""
    from logging_setup import configure_logging

    try:
        configure_logging("INFO", log_dir=tmp_path / "logs", file_name="test.log")
        logging.getLogger("test").info("hello", extra={"run_id": "r1"})
        for handler in logging.getLogger().handlers:
            handler.flush()

        written = (tmp_path / "logs" / "test.log").read_text(encoding="utf-8").strip()
        record = json.loads(written.splitlines()[-1])
        assert record["message"] == "hello"
        assert record["run_id"] == "r1"
        # Both destinations carry the same JSON, so a line from a file and a
        # line scraped from stdout parse identically.
        assert record["level"] == "INFO"
    finally:
        # Release the file handle; Windows will not delete tmp_path otherwise.
        configure_logging("INFO")


def test_an_unwritable_log_directory_does_not_stop_the_backend(tmp_path: Path):
    """Losing the log is bad. Refusing to start is worse."""
    from logging_setup import configure_logging

    blocker = tmp_path / "logs"
    blocker.write_text("I am a file, not a directory", encoding="utf-8")

    try:
        configure_logging("INFO", log_dir=blocker, file_name="test.log")
        logging.getLogger("test").info("still logging")
        # stdout handler survives on its own.
        assert logging.getLogger().handlers
    finally:
        configure_logging("INFO")
