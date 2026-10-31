from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from synapse.config import SynapseConfig
from synapse.security import (
    RedactionEngine,
    purge_expired_archive_files,
    scrub_credentials,
    strip_lone_surrogates,
)


FIXED_TIME = datetime(2026, 3, 1, 10, 30, tzinfo=UTC)


def test_redaction_engine_handles_default_patterns() -> None:
    text = (
        "Reach me at owner@example.com from 203.0.113.5 with key "
        "sk-1234567890abcdef1234567890abcdef and uuid "
        "123e4567-e89b-12d3-a456-426614174000 plus token "
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTYifQ.signature"
    )

    redacted, labels = RedactionEngine().redact(text)

    assert "[REDACTED_EMAIL]" in redacted
    assert "[REDACTED_IP]" in redacted
    assert "[REDACTED_API_KEY]" in redacted
    assert "[REDACTED_UUID]" in redacted
    assert "[REDACTED_JWT]" in redacted
    assert sorted(labels) == ["API_KEY", "EMAIL", "IP", "JWT", "UUID"]


def test_redaction_covers_token_assignments_and_authorization_headers() -> None:
    first = "opaqueTokenExample1234567890~abc"
    second = "anotherSecretExample1234567890"
    third = "bearerCredentialExample1234567890"
    text = (
        f'TOKEN = "{first}"\n'
        + r'{"command":"TOKEN = \"' + second + r'\""}' + "\n"
        + f"Authorization: Bearer {third}\n"
        + "ordinary_token_count = 120"
    )
    redacted, labels = RedactionEngine().redact(text)
    assert first not in redacted and second not in redacted and third not in redacted
    assert redacted.count("[REDACTED_SECRET]") == 3
    assert "ordinary_token_count = 120" in redacted
    assert sorted(labels) == ["AUTHORIZATION", "SECRET_ASSIGNMENT", "SECRET_ASSIGNMENT"]


def test_custom_patterns_from_config_are_applied() -> None:
    config = SynapseConfig.model_validate(
        {
            "sanitization": {
                "custom_patterns": [
                    {
                        "pattern": r"ACME-[0-9]{4}",
                        "replacement": "[REDACTED_ACME]",
                        "label": "ACME_TOKEN",
                    }
                ]
            }
        }
    )

    redacted, labels = RedactionEngine.from_config(config).redact("Secret ACME-4242 token, ACME-1111")

    assert redacted == "Secret [REDACTED_ACME] token, [REDACTED_ACME]"
    assert labels.count("ACME_TOKEN") == 2


def test_purge_expired_archive_files_removes_only_old_markdown(tmp_path: Path) -> None:
    archive_root = tmp_path / ".synapse" / ".archive"
    archive_root.mkdir(parents=True)
    stale = archive_root / "stale.md"
    fresh = archive_root / "fresh.md"
    stale.write_text("old", encoding="utf-8")
    fresh.write_text("new", encoding="utf-8")

    old_timestamp = (FIXED_TIME - timedelta(days=120)).timestamp()
    new_timestamp = (FIXED_TIME - timedelta(days=5)).timestamp()
    stale.touch()
    fresh.touch()
    stale.chmod(0o600)
    fresh.chmod(0o600)
    import os

    os.utime(stale, (old_timestamp, old_timestamp))
    os.utime(fresh, (new_timestamp, new_timestamp))

    removed = purge_expired_archive_files(archive_root, retention_days=90, now=FIXED_TIME)

    assert removed == [stale]
    assert stale.exists() is False
    assert fresh.exists() is True


def test_strip_lone_surrogates_repairs_unpaired_units_only() -> None:
    assert strip_lone_surrogates("low \uddc4 tail") == "low \ufffd tail"
    assert strip_lone_surrogates("high \ud83d tail") == "high \ufffd tail"
    # A valid astral pair (🗄️ = U+1F5C4, i.e. \ud83d\uddc4) is left untouched.
    assert strip_lone_surrogates("cabinet \U0001F5C4 ok") == "cabinet \U0001F5C4 ok"
    assert strip_lone_surrogates("") == ""


def test_scrub_credentials_redacts_values_and_keeps_identifiers() -> None:
    assert scrub_credentials("password=hunter2") == "password=[REDACTED_CREDENTIAL]"
    assert scrub_credentials("CLIENT_SECRET=3gKp9xQ2") == "CLIENT_SECRET=[REDACTED_CREDENTIAL]"
    assert scrub_credentials("postgres://appuser:s3cr3t@db.internal:5432/app") == (
        "postgres://appuser:[REDACTED_CREDENTIAL]@db.internal:5432/app"
    )
    assert scrub_credentials("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig") == (
        "Authorization: Bearer [REDACTED_CREDENTIAL]"
    )
    key_block = "-----BEGIN RSA PRIVATE KEY-----\nMIIEow\n-----END RSA PRIVATE KEY-----"
    assert scrub_credentials(key_block) == "[REDACTED_PRIVATE_KEY]"
    secret_manifest = "stringData:\n  username: svc-user\n  password: hunter2\nmetadata:\n  name: si\n"
    assert scrub_credentials(secret_manifest) == (
        "stringData:\n  username: svc-user\n  password: [REDACTED_CREDENTIAL]\nmetadata:\n  name: si\n"
    )

    # Knowledge keeps identifiers, env references, placeholders, and key names.
    identifiers = "CLIENT_ID=7dfb93a3-cd40-4963-a76f-20e0085c0ceb ip 192.0.2.5 mail a@b.com"
    assert scrub_credentials(identifiers) == identifiers
    assert scrub_credentials("CLIENT_SECRET=${SI_CLIENT_SECRET}") == "CLIENT_SECRET=${SI_CLIENT_SECRET}"
    assert scrub_credentials("password=<your-password>") == "password=<your-password>"
    assert scrub_credentials("the client_secret key lives in values.yaml") == (
        "the client_secret key lives in values.yaml"
    )
    assert scrub_credentials("data:\n  replicas: 3\n  name: authz\n") == "data:\n  replicas: 3\n  name: authz\n"
    assert scrub_credentials("") == ""

    # The full engine still redacts identifiers for outbound payloads.
    outbound, labels = RedactionEngine().redact("mail a@b.com ip 192.0.2.5 password=hunter2")
    assert "[REDACTED_EMAIL]" in outbound and "[REDACTED_IP]" in outbound
    assert "[REDACTED_CREDENTIAL]" in outbound and "CREDENTIAL_ASSIGNMENT" in labels
