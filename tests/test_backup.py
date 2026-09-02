from dataclasses import asdict
import json
from pathlib import Path
import hashlib
import sqlite3
import zipfile

import pytest

from tars import backup, state_store


def layout(root):
    config_root = root / "config"
    data = root / "data"
    state = root / "state"
    return backup.BackupPaths(
        config_root / "config.toml", data, state, config_root / "themes",
        config_root / "ui.toml", config_root / "persona", state / "tars-state.sqlite3")


@pytest.fixture
def source(monkeypatch, tmp_path):
    paths = layout(tmp_path / "source")
    monkeypatch.setattr(state_store, "STATE_DB_PATH", paths.state_db_path)
    monkeypatch.setattr(state_store, "TASK_ROOT", tmp_path / "legacy")
    monkeypatch.setattr(state_store, "TASK_EVENTS_ROOT", tmp_path / "legacy-events")
    monkeypatch.setattr(state_store, "TASK_INDEX_PATH", tmp_path / "legacy-index")
    state_store.ensure_state_store()
    paths.config_path.parent.mkdir(parents=True)
    paths.config_path.write_text(
        '[runtime]\nbackend="llama.cpp"\n[services]\ntoken="env:SERVICE_TOKEN"\n')
    paths.ui_prefs_path.write_text('theme="midnight"\n')
    paths.persona_root.mkdir()
    (paths.persona_root / "IDENTITY.md").write_text("identity")
    (paths.config_path.parent / "skills" / "fixture").mkdir(parents=True)
    (paths.config_path.parent / "skills" / "fixture" / "SKILL.md").write_text("skill")
    paths.data_root.mkdir(parents=True)
    (paths.data_root / "model-registry.toml").write_text(
        '[models.local]\npath="/missing/model.gguf"\n')
    (paths.data_root / "role-registry.toml").write_text('default_role="general"\n')
    memory_id = "mem-" + "1" * 32
    (paths.data_root / "memory" / "profile").mkdir(parents=True)
    (paths.data_root / "memory" / "profile" / f"{memory_id}.md").write_text(
        f'---\nid: "{memory_id}"\nkind: "profile"\nscope: "global"\ntitle: "Portable"\n'
        'source: "user"\ncreated_at: "now"\nupdated_at: "now"\nconfidence: 1.0\n'
        'supersedes: null\nexpiry: null\ntags: []\n---\n\nmemory\n')
    (paths.data_root / "model-artifacts" / "sha256").mkdir(parents=True)
    (paths.data_root / "model-artifacts" / "sha256" / "weight.gguf").write_text("WEIGHT")
    (paths.state_root / "browser" / "profile").mkdir(parents=True)
    (paths.state_root / "browser" / "profile" / "Cookies").write_text("COOKIE")
    with state_store.transaction() as conn:
        conn.execute(
            "INSERT INTO conversations(id,title,created_at,updated_at) VALUES(?,?,?,?)",
            ("conv-backup", "Portable", "now", "now"))
        conn.execute(
            "INSERT INTO core_clients(id,name,principal_id,token_salt,token_hash,permissions_json,"
            "state,created_at,metadata_json) VALUES(?,?,?,?,?,?,?,?,?)",
            ("client-backup", "Laptop", "owner", "salt-secret", "hash-secret", "[]",
             "active", "now", '{"platform":"linux","token":"metadata-secret"}'))
        conn.execute(
            "INSERT INTO core_pairings(id,code_hash,permissions_json,principal_id,created_at,expires_at)"
            " VALUES(?,?,?,?,?,?)", ("pair-backup", "pair-secret", "[]", "owner", "now", "later"))
    return paths


def test_real_backup_reset_restore_roundtrip_excludes_heavy_and_secret_state(source, tmp_path):
    bundle = tmp_path / "portable.tarsbundle"
    created = backup.create_bundle(bundle, paths=source)
    assert created.schema_version == state_store.SCHEMA_VERSION
    assert bundle.stat().st_mode & 0o077 == 0
    with zipfile.ZipFile(bundle) as archive:
        names = set(archive.namelist())
        payload = b"".join(archive.read(name) for name in names)
    assert "data/memory/profile/mem-11111111111111111111111111111111.md" in names
    assert not any("model-artifacts" in name or "browser" in name for name in names)
    assert b"WEIGHT" not in payload and b"COOKIE" not in payload
    assert b"salt-secret" not in payload and b"hash-secret" not in payload
    assert b"metadata-secret" not in payload

    destination = layout(tmp_path / "restored")
    report = backup.restore_bundle(bundle, paths=destination, replace=True)
    memory_path = destination.data_root / "memory/profile/mem-11111111111111111111111111111111.md"
    assert memory_path.read_text().endswith("memory\n")
    assert destination.state_db_path.stat().st_mode & 0o077 == 0
    conn = sqlite3.connect(destination.state_db_path)
    try:
        assert conn.execute("SELECT title FROM conversations WHERE id='conv-backup'").fetchone()[0] == "Portable"
        client = conn.execute(
            "SELECT name,state,token_hash FROM core_clients WHERE id='client-backup'").fetchone()
        assert client == ("Laptop", "revoked", "excluded")
        assert conn.execute("SELECT COUNT(*) FROM core_pairings").fetchone()[0] == 0
        indexed = conn.execute(
            "SELECT path FROM memory_index WHERE id='mem-11111111111111111111111111111111'").fetchone()
    finally:
        conn.close()
    assert indexed[0] == str(memory_path)
    assert report.reconciliation["model_assets_required"] == ["local"]
    assert report.reconciliation["external_effects_rolled_back"] is False
    assert report.reconciliation["memory_index_entries_rebuilt"] == 1
    assert report.reconciliation["runtime_and_calibration_revalidation_required"] is True
    assert "env:SERVICE_TOKEN" in report.reconciliation["unresolved_secret_references"]


def test_validation_precedes_restore_mutation_and_future_versions_are_truthful(source, tmp_path):
    bundle = tmp_path / "valid.tarsbundle"
    backup.create_bundle(bundle, paths=source)
    corrupt = tmp_path / "corrupt.tarsbundle"
    corrupt.write_bytes(bundle.read_bytes())
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(corrupt, "a") as archive:
            archive.writestr("data/memory/profile/mem-11111111111111111111111111111111.md",
                             "tampered")
    destination = layout(tmp_path / "destination")
    destination.config_path.parent.mkdir(parents=True)
    destination.config_path.write_text("sentinel")
    with pytest.raises(ValueError, match="duplicate"):
        backup.restore_bundle(corrupt, paths=destination, replace=True)
    assert destination.config_path.read_text() == "sentinel"

    future = tmp_path / "future.tarsbundle"
    with zipfile.ZipFile(future, "w") as archive:
        archive.writestr("manifest.json", json.dumps({
            "format": backup.BUNDLE_FORMAT, "bundle_version": backup.BUNDLE_VERSION + 1,
            "schema_version": state_store.SCHEMA_VERSION, "files": {}}))
    with pytest.raises(ValueError, match="newer than supported"):
        backup.inspect_bundle(future)


def test_backup_refuses_plaintext_like_configured_secrets(source, tmp_path):
    source.config_path.write_text('[service]\npassword="plaintext"\n')
    with pytest.raises(ValueError, match="refused plaintext"):
        backup.create_bundle(tmp_path / "unsafe.tarsbundle", paths=source)
    assert not (tmp_path / "unsafe.tarsbundle").exists()


def test_backup_destination_parent_replacement_cannot_redirect_output(
        monkeypatch, source, tmp_path):
    destination_root = tmp_path / "destination"
    destination_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    destination = destination_root / "portable.tarsbundle"
    snapshot = backup._snapshot_database

    def snapshot_then_swap(*args, **kwargs):
        result = snapshot(*args, **kwargs)
        destination_root.rename(tmp_path / "displaced-destination")
        destination_root.symlink_to(outside, target_is_directory=True)
        return result

    monkeypatch.setattr(backup, "_snapshot_database", snapshot_then_swap)
    report = backup.create_bundle(destination, paths=source)
    assert report.files > 0
    assert not (outside / destination.name).exists()
    assert (tmp_path / "displaced-destination" / destination.name).is_file()


def test_restore_requires_explicit_replace(source, tmp_path):
    bundle = tmp_path / "portable.tarsbundle"
    backup.create_bundle(bundle, paths=source)
    with pytest.raises(PermissionError, match="replace=True"):
        backup.restore_bundle(bundle, paths=layout(tmp_path / "restored"))


def test_older_schema_bundle_is_migrated_in_the_real_restore_path(source, tmp_path):
    current = tmp_path / "current.tarsbundle"
    backup.create_bundle(current, paths=source)
    with zipfile.ZipFile(current) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(entries.pop("manifest.json"))
    database = tmp_path / "older.sqlite3"
    database.write_bytes(entries[backup.DB_MEMBER])
    older_version = state_store.SCHEMA_VERSION - 1
    conn = sqlite3.connect(database)
    try:
        for name, introduced in state_store._SCHEMA_INTRODUCED.items():
            if introduced > older_version:
                object_type = conn.execute(
                    "SELECT type FROM sqlite_master WHERE name=?", (name,),
                ).fetchone()[0]
                conn.execute(f'DROP {object_type.upper()} "{name}"')
        conn.execute("UPDATE meta SET value=? WHERE key='schema_version'",
                     (str(older_version),))
        conn.commit()
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='messages_lineage_insert'"
        ).fetchone() is None
    finally:
        conn.close()
    entries[backup.DB_MEMBER] = database.read_bytes()
    manifest["schema_version"] = older_version
    manifest["files"][backup.DB_MEMBER] = {
        "size": len(entries[backup.DB_MEMBER]),
        "sha256": hashlib.sha256(entries[backup.DB_MEMBER]).hexdigest(),
    }
    older = tmp_path / "older.tarsbundle"
    with zipfile.ZipFile(older, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in entries.items():
            archive.writestr(name, value)
        archive.writestr("manifest.json", json.dumps(manifest))

    destination = layout(tmp_path / "migrated")
    backup.restore_bundle(older, paths=destination, replace=True)
    conn = sqlite3.connect(destination.state_db_path)
    try:
        assert conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] == str(
                state_store.SCHEMA_VERSION)
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='messages_lineage_insert'"
        ).fetchone()
    finally:
        conn.close()


def test_post_apply_failure_restores_prior_local_state(monkeypatch, source, tmp_path):
    bundle = tmp_path / "portable.tarsbundle"
    backup.create_bundle(bundle, paths=source)
    destination = layout(tmp_path / "destination")
    destination.config_path.parent.mkdir(parents=True)
    destination.config_path.write_text("prior-config")
    monkeypatch.setattr(backup.memory, "rebuild_index",
                        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("rebuild failed")))
    with pytest.raises(RuntimeError, match="rebuild failed"):
        backup.restore_bundle(bundle, paths=destination, replace=True)
    assert destination.config_path.read_text() == "prior-config"
    assert not destination.state_db_path.exists()
