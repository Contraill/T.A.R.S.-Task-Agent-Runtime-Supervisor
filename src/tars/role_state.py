from __future__ import annotations

from .roles import resolve_role_id
from .state_store import connect, ensure_state_store, json_dumps, json_loads, now_utc, transaction


def load_role_state(role_name: str) -> dict:
    ensure_state_store()
    role_id = resolve_role_id(role_name)
    conn = connect()
    try:
        row = conn.execute("SELECT state_json FROM role_state WHERE role_id=?", (role_id,)).fetchone()
        return json_loads(row[0], {}) if row else {}
    finally:
        conn.close()


def save_role_state(role_name: str, state: dict) -> dict:
    ensure_state_store()
    role_id = resolve_role_id(role_name)
    with transaction(immediate=True) as conn:
        conn.execute(
            """INSERT INTO role_state(role_id,state_json,updated_at) VALUES(?,?,?)
               ON CONFLICT(role_id) DO UPDATE SET state_json=excluded.state_json,
               updated_at=excluded.updated_at""",
            (role_id, json_dumps(state), now_utc()),
        )
    return load_role_state(role_id)
