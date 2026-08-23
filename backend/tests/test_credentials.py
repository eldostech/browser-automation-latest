"""The credential vault.

The property that matters most is the negative one: with no key configured the
vault refuses to store anything rather than falling back to plaintext.
"""

from __future__ import annotations

import pytest

from credentials import (
    Vault,
    VaultError,
    VaultUnavailable,
    generate_key,
    missing_slots,
    new_credential_id,
)

BUNDLE = {"username": "nitin", "password": "s3cret-Example-Pw!"}


@pytest.fixture
def vault() -> Vault:
    return Vault(generate_key())


# --- no key means no storage, never plaintext ------------------------------


def test_without_a_key_the_vault_is_unavailable():
    assert Vault(None).available is False
    assert Vault("").available is False


def test_sealing_without_a_key_raises_rather_than_storing_plaintext():
    with pytest.raises(VaultUnavailable, match="CREDENTIALS_KEY"):
        Vault(None).seal(BUNDLE)


def test_opening_without_a_key_raises():
    with pytest.raises(VaultUnavailable):
        Vault(None).open(b"anything")


def test_the_error_explains_how_to_generate_a_key():
    with pytest.raises(VaultUnavailable, match="Fernet.generate_key"):
        Vault(None).seal(BUNDLE)


def test_a_malformed_key_is_rejected_at_construction():
    with pytest.raises(VaultUnavailable, match="not a valid Fernet key"):
        Vault("obviously-not-a-fernet-key")


# --- round trip ------------------------------------------------------------


def test_a_bundle_round_trips(vault: Vault):
    assert vault.open(vault.seal(BUNDLE)) == BUNDLE


def test_the_ciphertext_does_not_contain_the_values(vault: Vault):
    ciphertext = vault.seal(BUNDLE)
    assert b"s3cret-Example-Pw!" not in ciphertext
    assert b"nitin" not in ciphertext


def test_empty_values_are_dropped_and_an_empty_bundle_is_refused(vault: Vault):
    assert vault.open(vault.seal({"a": "x", "b": ""})) == {"a": "x"}
    with pytest.raises(ValueError, match="at least one non-empty slot"):
        vault.seal({"a": ""})


def test_a_tampered_ciphertext_fails_loudly_rather_than_decrypting_to_garbage(vault: Vault):
    ciphertext = bytearray(vault.seal(BUNDLE))
    ciphertext[-5] ^= 0xFF
    with pytest.raises(VaultError, match="could not be decrypted"):
        vault.open(bytes(ciphertext))


def test_a_bundle_sealed_with_another_key_cannot_be_opened(vault: Vault):
    other = Vault(generate_key())
    with pytest.raises(VaultError, match="CREDENTIALS_KEY has changed"):
        other.open(vault.seal(BUNDLE))


def test_corrupt_input_is_reported_not_swallowed(vault: Vault):
    with pytest.raises(VaultError):
        vault.open(b"not a fernet token at all")


# --- helpers ---------------------------------------------------------------


def test_slots_are_reported_in_a_stable_order(vault: Vault):
    assert Vault.slots_of(BUNDLE) == ["password", "username"]


def test_missing_slots_reports_what_a_use_case_still_needs():
    assert missing_slots(["username", "password", "pin"], {"username": "u", "pin": ""}) == [
        "password",
        "pin",
    ]


def test_ids_are_unique():
    assert new_credential_id() != new_credential_id()


# --- storage ---------------------------------------------------------------


async def test_only_ciphertext_reaches_the_database(store, vault: Vault):
    ciphertext = vault.seal(BUNDLE)
    credential_id = await store.save_credential(
        new_credential_id(), "IXL account", Vault.slots_of(BUNDLE), ciphertext
    )

    await store.db.execute("PRAGMA wal_checkpoint(FULL)")
    await store.db.commit()
    assert b"s3cret-Example-Pw!" not in store.db_path.read_bytes()

    stored = await store.get_credential_ciphertext(credential_id)
    assert vault.open(stored) == BUNDLE


async def test_listing_credentials_never_returns_a_value(store, vault: Vault):
    await store.save_credential(
        new_credential_id(), "IXL account", Vault.slots_of(BUNDLE), vault.seal(BUNDLE)
    )
    rows = await store.list_credentials()

    assert rows[0]["name"] == "IXL account"
    assert rows[0]["slots"] == ["password", "username"]
    assert "s3cret-Example-Pw!" not in str(rows)
    assert "ciphertext" not in rows[0]


async def test_saving_the_same_name_replaces_it(store, vault: Vault):
    first = await store.save_credential(
        new_credential_id(), "IXL", ["password"], vault.seal({"password": "old"})
    )
    second = await store.save_credential(
        new_credential_id(), "IXL", ["password"], vault.seal({"password": "new"})
    )

    assert first == second, "the id is stable across a re-save"
    assert len(await store.list_credentials()) == 1
    assert vault.open(await store.get_credential_ciphertext(first)) == {"password": "new"}


async def test_deleting_a_credential(store, vault: Vault):
    credential_id = await store.save_credential(
        new_credential_id(), "IXL", ["password"], vault.seal({"password": "x"})
    )
    assert await store.delete_credential(credential_id) is True
    assert await store.delete_credential(credential_id) is False
    assert await store.list_credentials() == []


async def test_use_is_recorded(store, vault: Vault):
    credential_id = await store.save_credential(
        new_credential_id(), "IXL", ["password"], vault.seal({"password": "x"})
    )
    assert (await store.list_credentials())[0]["last_used_at"] is None

    await store.touch_credential(credential_id)
    assert (await store.list_credentials())[0]["last_used_at"] is not None
