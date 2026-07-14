"""Tests for agent/internal_hash.py — SHA-256 fallback for internal legacy digests.

Simulates a hardened crypto-policy runtime that rejects MD5/SHA-1 even with
usedforsecurity=False (some FIPS configurations do), without requiring a FIPS
CI runner: hashlib.new is patched to raise for the legacy algorithms.
"""

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.internal_hash import internal_digest, internal_hasher

_ORIGINAL_HASHLIB_NEW = hashlib.new


def _reject_legacy_hashes(name, *args, **kwargs):
    if name in {"md5", "sha1"}:
        raise ValueError("legacy digest unavailable")
    return _ORIGINAL_HASHLIB_NEW(name, *args, **kwargs)


@pytest.fixture()
def strict_runtime():
    """Runtime that rejects MD5/SHA-1 outright, even for non-security use."""
    with patch("hashlib.new", side_effect=_reject_legacy_hashes):
        yield


def _legacy_hexdigest(algorithm: str, data: bytes) -> str:
    """Reference legacy digest; skips the test on runtimes that reject it."""
    try:
        return hashlib.new(algorithm, data, usedforsecurity=False).hexdigest()
    except (TypeError, ValueError):
        pytest.skip(f"runtime rejects {algorithm} even with usedforsecurity=False")


class TestPermissiveRuntime:
    """On permissive runtimes the legacy digests must be byte-for-byte unchanged."""

    def test_md5_output_unchanged(self):
        assert internal_digest(b"hello") == _legacy_hexdigest("md5", b"hello")

    def test_sha1_output_unchanged(self):
        assert internal_digest(b"hello", legacy_algorithm="sha1") == _legacy_hexdigest("sha1", b"hello")

    def test_truncation_matches_hexdigest_slice(self):
        assert internal_digest(b"hello", length=12) == _legacy_hexdigest("md5", b"hello")[:12]

    def test_incremental_hasher_matches_one_shot(self):
        hasher = internal_hasher(legacy_algorithm="md5")
        hasher.update(b"he")
        hasher.update(b"llo")
        assert hasher.hexdigest() == internal_digest(b"hello")


class TestStrictRuntime:
    def test_md5_falls_back_to_sha256(self, strict_runtime):
        assert internal_digest(b"hello") == hashlib.sha256(b"hello").hexdigest()

    def test_sha1_falls_back_to_sha256(self, strict_runtime):
        assert internal_digest(b"hello", legacy_algorithm="sha1") == hashlib.sha256(b"hello").hexdigest()

    def test_incremental_hasher_falls_back_to_sha256(self, strict_runtime):
        hasher = internal_hasher(legacy_algorithm="md5")
        hasher.update(b"he")
        hasher.update(b"llo")
        assert hasher.hexdigest() == hashlib.sha256(b"hello").hexdigest()

    def test_typeerror_also_falls_back(self):
        # Python built against an OpenSSL whose constructors lack the
        # usedforsecurity kwarg raises TypeError instead of ValueError.
        def _no_kwarg(name, *args, **kwargs):
            if "usedforsecurity" in kwargs:
                raise TypeError("usedforsecurity is an invalid keyword argument")
            return _ORIGINAL_HASHLIB_NEW(name, *args, **kwargs)

        with patch("hashlib.new", side_effect=_no_kwarg):
            assert internal_digest(b"hello") == hashlib.sha256(b"hello").hexdigest()


class TestContextCompressorDedup:
    def _compressor(self):
        from agent.context_compressor import ContextCompressor

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            return ContextCompressor(model="test/model", quiet_mode=True)

    def _messages(self):
        big = "x" * 500
        return [
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "c1", "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'}}]},
            {"role": "tool", "content": big, "tool_call_id": "c1"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "c2", "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'}}]},
            {"role": "tool", "content": big, "tool_call_id": "c2"},
        ]

    def test_dedup_still_works_on_strict_runtime(self, strict_runtime):
        result, pruned = self._compressor()._prune_old_tool_results(
            self._messages(), protect_tail_count=len(self._messages()),
        )
        assert pruned == 1
        assert "Duplicate tool output" in result[1]["content"]
        assert result[3]["content"] == "x" * 500

    def test_dedup_key_is_12_chars_on_both_runtimes(self, strict_runtime):
        content = "x" * 500
        assert len(internal_digest(content.encode("utf-8", errors="replace"), length=12)) == 12


class TestCodexFallbackId:
    def test_id_shape_unchanged_when_sha1_allowed(self):
        from agent.codex_responses_adapter import _derive_responses_function_call_id

        expected = _legacy_hexdigest("sha1", b"???")[:24]
        assert _derive_responses_function_call_id("???") == f"fc_{expected}"

    def test_id_shape_preserved_on_strict_runtime(self, strict_runtime):
        from agent.codex_responses_adapter import _derive_responses_function_call_id

        derived = _derive_responses_function_call_id("???")
        assert derived.startswith("fc_")
        assert len(derived) == len("fc_") + 24
        assert derived == f"fc_{hashlib.sha256(b'???').hexdigest()[:24]}"


class TestSkillsHubCacheKeys:
    def test_cache_key_generation_succeeds_on_strict_runtime(self, strict_runtime):
        from tools.skills_hub import WellKnownSkillSource

        seen_keys = []

        def _fake_read_cache(cache_key):
            seen_keys.append(cache_key)
            return {"skills": []}

        with patch("tools.skills_hub._read_index_cache", side_effect=_fake_read_cache):
            parsed = WellKnownSkillSource()._parse_index("https://example.com/.well-known/skills/index.json")

        assert parsed == {"skills": []}
        assert len(seen_keys) == 1
        prefix, _, digest = seen_keys[0].partition("well_known_index_")
        assert prefix == ""
        # SHA-256 fallback: 64 hex chars instead of MD5's 32.
        assert len(digest) == 64
        int(digest, 16)


class TestSkillsSyncDirHash:
    def _skill_dir(self, tmp_path: Path) -> Path:
        skill = tmp_path / "skill"
        (skill / "sub").mkdir(parents=True)
        (skill / "SKILL.md").write_bytes(b"hello")
        (skill / "sub" / "extra.py").write_bytes(b"world")
        return skill

    def test_dir_hash_unchanged_when_md5_allowed(self, tmp_path):
        from tools.skills_sync import _dir_hash

        try:
            expected = hashlib.md5(usedforsecurity=False)
        except (TypeError, ValueError):
            pytest.skip("runtime rejects md5 even with usedforsecurity=False")
        skill = self._skill_dir(tmp_path)
        for fpath in sorted(skill.rglob("*")):
            if fpath.is_file():
                expected.update(str(fpath.relative_to(skill)).encode("utf-8"))
                expected.update(fpath.read_bytes())

        assert _dir_hash(skill) == expected.hexdigest()

    def test_dir_hash_succeeds_on_strict_runtime(self, strict_runtime, tmp_path):
        from tools.skills_sync import _dir_hash

        skill = self._skill_dir(tmp_path)
        digest = _dir_hash(skill)
        assert len(digest) == 64
        int(digest, 16)
        # Deterministic across calls so change detection still works.
        assert digest == _dir_hash(skill)


class TestProtocolHashesUntouched:
    """Protocol-defined digests must keep calling hashlib directly.

    External APIs (QQ Bot chunk checksums, WeCom/Weixin upload MD5, Yuanbao
    media MD5 and request-signing SHA-1) verify these values server-side, so
    they must never silently degrade to SHA-256.
    """

    PROTOCOL_FILES = [
        "gateway/platforms/qqbot/chunked_upload.py",
        "gateway/platforms/weixin.py",
        "gateway/platforms/yuanbao_media.py",
        "plugins/platforms/wecom/adapter.py",
    ]

    @pytest.mark.parametrize("rel_path", PROTOCOL_FILES)
    def test_protocol_files_do_not_use_internal_hash(self, rel_path):
        repo_root = Path(__file__).resolve().parents[2]
        source = (repo_root / rel_path).read_text(encoding="utf-8")
        assert "internal_hash" not in source
        assert "internal_digest" not in source
        assert "internal_hasher" not in source
