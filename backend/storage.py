"""Where screenshots and other run artifacts are kept.

One flag, ``STORAGE_BACKEND``, chooses between the local filesystem and S3.
Everything above this module works in terms of an opaque *key* and never learns
which was chosen.

Why a key rather than a path
----------------------------
The database column has always held a filesystem path, which quietly made the
filesystem part of the schema: a row written by one deployment could not be
read by another, and moving to object storage would have meant rewriting every
existing row. A key (``<run_id>/<artifact_id>.png``) is the same string on both
backends. The *locator* stored alongside it says which backend wrote it, so a
deployment that switches can still serve everything it wrote before.

Serving
-------
Local artifacts are streamed from disk. S3 artifacts are served by redirecting
to a presigned URL, so the bytes go straight from S3 to the browser rather than
through this process. Those URLs are short-lived and unguessable, which is what
makes it acceptable for content that is otherwise access-controlled -- the
trade is deliberate, and the expiry is configurable.

Failure
-------
A screenshot that cannot be stored must never fail the run it belongs to. It is
a diagnostic aid; the run is the work. Every method here either succeeds or
raises :class:`StorageError`, and the one caller that matters catches it and
carries on.
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from config import Settings

log = logging.getLogger(__name__)

#: Prefix marking a key as living in object storage. A row without it is a
#: local file, which is what every row written before this module existed is.
S3_SCHEME = "s3://"


class StorageError(RuntimeError):
    """The artifact could not be written or read."""


@dataclass(slots=True)
class StoredArtifact:
    """Where an artifact ended up.

    ``locator`` goes in the database. ``key`` is the backend-independent path
    within whichever store wrote it.
    """

    key: str
    locator: str
    bytes: int


class ArtifactStorage(Protocol):
    """The surface everything above this module sees."""

    name: str

    async def put(self, key: str, data: bytes, *, content_type: str) -> StoredArtifact: ...

    async def get(self, locator: str) -> bytes: ...

    def presigned_url(self, locator: str) -> str | None:
        """A URL the browser can fetch directly, or None if there is no such thing."""


# ---------------------------------------------------------------------------
# Local filesystem
# ---------------------------------------------------------------------------


class LocalStorage:
    """Files under ``artifacts_dir``. The default, and what tests use."""

    name = "local"

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # A key arrives from our own code, but treating it as untrusted costs
        # nothing and stops a future caller turning "../" into an escape.
        candidate = (self.root / key).resolve()
        if not candidate.is_relative_to(self.root.resolve()):
            raise StorageError(f"{key!r} points outside the artifacts directory.")
        return candidate

    async def put(self, key: str, data: bytes, *, content_type: str) -> StoredArtifact:
        path = self._path(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Off the event loop: a large screenshot write would otherwise
            # stall every other request for the duration.
            await asyncio.to_thread(path.write_bytes, data)
        except OSError as exc:
            raise StorageError(f"could not write {key}: {exc}") from exc
        return StoredArtifact(key=key, locator=str(path), bytes=len(data))

    async def get(self, locator: str) -> bytes:
        path = Path(locator)
        try:
            return await asyncio.to_thread(path.read_bytes)
        except OSError as exc:
            raise StorageError(f"could not read {locator}: {exc}") from exc

    def presigned_url(self, locator: str) -> str | None:
        return None  # a local file is served by this process


# ---------------------------------------------------------------------------
# S3
# ---------------------------------------------------------------------------


class S3Storage:
    """Objects in a bucket, addressed ``s3://bucket/key``.

    boto3 is synchronous, so every call runs in a worker thread. A client is
    built once and reused: constructing one costs a config and credential
    resolution pass that has no business happening per screenshot.
    """

    name = "s3"

    def __init__(self, settings: Settings) -> None:
        if not settings.s3_bucket:
            raise StorageError(
                "STORAGE_BACKEND is 's3' but S3_BUCKET is not set, so there is nowhere "
                "to put anything."
            )
        self.bucket = settings.s3_bucket
        self.prefix = settings.s3_prefix.strip("/")
        self.expiry = settings.s3_url_expiry_seconds
        self._settings = settings
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import boto3

            session = boto3.session.Session(
                region_name=self._settings.s3_region or self._settings.aws_region or None,
                profile_name=self._settings.aws_profile or None,
            )
            # endpoint_url lets this point at MinIO or any S3-compatible store,
            # which is also how it gets tested without a real bucket.
            self._client = session.client(
                "s3", endpoint_url=self._settings.s3_endpoint_url or None
            )
        return self._client

    def _object_key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def _locator(self, key: str) -> str:
        return f"{S3_SCHEME}{self.bucket}/{self._object_key(key)}"

    @staticmethod
    def _split(locator: str) -> tuple[str, str]:
        bucket, _, key = locator[len(S3_SCHEME) :].partition("/")
        return bucket, key

    async def put(self, key: str, data: bytes, *, content_type: str) -> StoredArtifact:
        object_key = self._object_key(key)
        try:
            await asyncio.to_thread(
                self.client.put_object,
                Bucket=self.bucket,
                Key=object_key,
                Body=data,
                ContentType=content_type,
            )
        except Exception as exc:  # noqa: BLE001 - botocore raises a wide family
            raise StorageError(f"could not upload {object_key}: {exc}") from exc
        return StoredArtifact(key=key, locator=self._locator(key), bytes=len(data))

    async def get(self, locator: str) -> bytes:
        bucket, key = self._split(locator)
        try:
            response = await asyncio.to_thread(
                self.client.get_object, Bucket=bucket, Key=key
            )
            return await asyncio.to_thread(response["Body"].read)
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"could not read {locator}: {exc}") from exc

    def presigned_url(self, locator: str) -> str | None:
        bucket, key = self._split(locator)
        try:
            return self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=self.expiry,
            )
        except Exception:  # noqa: BLE001 - fall back to streaming through us
            log.exception("could not presign an artifact URL", extra={"locator": locator})
            return None


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def build_storage(settings: Settings) -> ArtifactStorage:
    """The backend this deployment is configured for."""
    if settings.storage_backend == "s3":
        return S3Storage(settings)
    return LocalStorage(settings.artifacts_path)


def storage_for(locator: str, configured: ArtifactStorage, settings: Settings) -> ArtifactStorage:
    """The backend that can read a *specific* artifact.

    A deployment that switches from local to S3 still has rows pointing at
    files on disk. Reading follows the row rather than the current setting, so
    yesterday's screenshots keep working after today's change.
    """
    if locator.startswith(S3_SCHEME):
        if configured.name == "s3":
            return configured
        return S3Storage(settings)
    if configured.name == "local":
        return configured
    return LocalStorage(settings.artifacts_path)


def artifact_key(run_id: str, artifact_id: str, suffix: str) -> str:
    """``<run>/<artifact>.<ext>`` -- identical on both backends."""
    return f"{run_id}/{artifact_id}{suffix}"


def guess_content_type(suffix: str, fallback: str = "application/octet-stream") -> str:
    return mimetypes.types_map.get(suffix, fallback)


__all__ = [
    "ArtifactStorage",
    "LocalStorage",
    "S3Storage",
    "S3_SCHEME",
    "StorageError",
    "StoredArtifact",
    "artifact_key",
    "build_storage",
    "guess_content_type",
    "storage_for",
]
