"""Internal (non-security) digest helpers with a FIPS-safe fallback.

hashlib.md5() and hashlib.sha1() raise ValueError on FIPS-enabled systems
(OpenSSL 3.0+ with the FIPS provider) unless usedforsecurity=False is passed.
Some hardened crypto-policy configurations reject the legacy algorithms even
with that flag, so these helpers fall back to SHA-256 when the runtime
refuses to construct the requested hasher.

Only for internal values (cache keys, content dedup, change detection,
generated identifiers). Protocol-defined digests — gateway upload checksums,
request signing — must keep calling hashlib directly so their output matches
what the external API expects.

This module intentionally imports nothing beyond the stdlib so it is safe to
use from both agent/ and tools/.
"""

import hashlib
from typing import Optional


def internal_hasher(*, legacy_algorithm: str = "md5"):
    """Return an incremental hasher for internal, non-security digests.

    Uses the requested legacy algorithm with usedforsecurity=False where the
    runtime permits it (preserving existing digest values), and falls back to
    SHA-256 where the algorithm is rejected outright.
    """
    try:
        # TypeError: Python built against an OpenSSL without the
        # usedforsecurity kwarg; ValueError: algorithm disabled by the
        # runtime's crypto policy even for non-security use.
        return hashlib.new(legacy_algorithm, usedforsecurity=False)
    except (TypeError, ValueError):
        return hashlib.sha256()


def internal_digest(
    data: bytes,
    *,
    legacy_algorithm: str = "md5",
    length: Optional[int] = None,
) -> str:
    """Hex digest of ``data`` for internal, non-security use.

    Preserves the legacy digest where the runtime permits it and falls back
    to SHA-256 otherwise. ``length`` truncates the hex digest, matching the
    ``hexdigest()[:n]`` idiom at existing call sites.
    """
    hasher = internal_hasher(legacy_algorithm=legacy_algorithm)
    hasher.update(data)
    digest = hasher.hexdigest()
    return digest[:length] if length is not None else digest
