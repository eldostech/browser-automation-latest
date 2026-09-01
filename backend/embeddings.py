"""Turning a failed page into a vector, so a past fix can be found again.

One seam, one implementation, and a deterministic stand-in for tests -- the
same shape ``llm.py`` uses for the same reason: nothing in the suite should
need AWS to run.

**Why embeddings at all.** A locator breaks because a page changed. The useful
question when it breaks again is "have we seen this page in this state before,
and what did we do about it?" -- and that is a similarity question over text
that will never match exactly. Two renderings of the same broken checkout page
differ in a session id and an order number; a string comparison says they are
unrelated and a vector says they are the same thing.

**Failure is not an error here.** Retrieval is an optimisation. If the model is
unreachable, or the account lacks the embedding model, healing carries on with
no prior fixes in the prompt -- which is exactly what it did before this module
existed. So every failure path returns ``None`` and logs at debug.
"""

from __future__ import annotations

import hashlib
import logging
import math
import struct
from typing import Any, Protocol

log = logging.getLogger(__name__)

#: Titan Text Embeddings V2, which is what Bedrock offers alongside Claude and
#: what an account with Claude access most likely already has. 1024 is its
#: default output size and the one the schema is built for -- changing it means
#: a migration, because a vector column has a fixed width.
DEFAULT_MODEL = "amazon.titan-embed-text-v2:0"
DIMENSIONS = 1024

#: Enough of a page to identify it. The whole accessibility tree of a large
#: application is mostly chrome that is identical on every page, and it costs
#: tokens to embed.
MAX_CHARS = 4_000


class Embedder(Protocol):
    async def embed(self, text: str) -> list[float] | None: ...


class BedrockEmbedder:
    """Titan on Bedrock, through the same credential chain as everything else."""

    def __init__(self, settings: Any) -> None:
        self._settings = settings
        self._client = None

    def _bedrock(self):
        if self._client is None:
            import boto3

            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self._settings.aws_region or None,
            )
        return self._client

    async def embed(self, text: str) -> list[float] | None:
        import asyncio
        import json

        body = json.dumps({"inputText": text[:MAX_CHARS], "dimensions": DIMENSIONS})
        model = getattr(self._settings, "embedding_model", DEFAULT_MODEL)

        def call() -> list[float] | None:
            response = self._bedrock().invoke_model(modelId=model, body=body)
            payload = json.loads(response["body"].read())
            vector = payload.get("embedding")
            return [float(v) for v in vector] if vector else None

        try:
            # boto3 is synchronous; a worker thread keeps the loop free.
            return await asyncio.to_thread(call)
        except Exception:  # noqa: BLE001 - retrieval is an optimisation
            log.debug("could not embed the page", exc_info=True)
            return None


class HashEmbedder:
    """A deterministic stand-in, for tests and for running without Bedrock.

    Not a real embedding: it hashes token trigrams into a fixed number of
    buckets. That is enough to make *similar text score higher than unrelated
    text*, which is the only property the tests assert and the only one the
    retrieval path depends on. It is not enough for production, which is why
    the real one is the default and this is opt-in.
    """

    def __init__(self, dimensions: int = DIMENSIONS) -> None:
        self.dimensions = dimensions

    async def embed(self, text: str) -> list[float] | None:
        if not text:
            return None
        vector = [0.0] * self.dimensions
        tokens = text.lower().split()
        for index in range(len(tokens)):
            for size in (1, 2):
                if index + size > len(tokens):
                    continue
                gram = " ".join(tokens[index : index + size])
                digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
                bucket = struct.unpack("<Q", digest)[0] % self.dimensions
                vector[bucket] += 1.0

        norm = math.sqrt(sum(value * value for value in vector))
        if not norm:
            return None
        return [value / norm for value in vector]


def build_embedder(settings: Any) -> Embedder:
    """The embedder this deployment should use."""
    if getattr(settings, "embedding_backend", "bedrock") == "hash":
        return HashEmbedder()
    return BedrockEmbedder(settings)


def page_signature(step_summary: str, wanted: str, page_url: str, page: str) -> str:
    """What gets embedded: the step, what it looked for, and the page.

    All three, because none identifies a failure on its own. The same page
    breaks different steps, and the same step breaks on different pages.
    """
    return "\n".join(
        [
            f"step: {step_summary}",
            f"looking for: {wanted}",
            f"page: {page_url}",
            "",
            page[:MAX_CHARS],
        ]
    )


__all__ = [
    "DEFAULT_MODEL",
    "DIMENSIONS",
    "BedrockEmbedder",
    "Embedder",
    "HashEmbedder",
    "build_embedder",
    "page_signature",
]
