from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import tomllib

from .config import STATE_ROOT, THEME_ROOT, UI_PREFS_PATH
from .file_transactions import (
    atomic_write_anchored_text,
    exclusive_file_lock,
    installation_transaction,
    read_anchored_text,
    regular_file_exists,
)


VALID_LOGOS = ("auto", "compact", "minimal", "none")


@dataclass(frozen=True)
class Theme:
    id: str
    name: str
    colors: dict[str, str]
    source: str = "builtin"


_BUILTINS: dict[str, Theme] = {
    "terminal": Theme(
        id="terminal",
        name="Terminal",
        colors={
            "accent": "ansibrightcyan",
            "muted": "ansibrightblack",
            "success": "ansibrightgreen",
            "warning": "ansibrightyellow",
            "error": "ansibrightred",
            "reasoning": "ansibrightmagenta",
            "tool": "ansibrightblue",
        },
    ),
    "tars": Theme(
        id="tars",
        name="T.A.R.S.",
        colors={
            "accent": "#6bdcff",
            "muted": "#7f8c8d",
            "success": "#77dd77",
            "warning": "#ffd166",
            "error": "#ff6b6b",
            "reasoning": "#c6a0f6",
            "tool": "#7dcfff",
        },
    ),
    "monochrome": Theme(
        id="monochrome",
        name="Monochrome",
        colors={
            "accent": "",
            "muted": "",
            "success": "",
            "warning": "",
            "error": "",
            "reasoning": "",
            "tool": "",
        },
    ),
    "high-contrast": Theme(
        id="high-contrast",
        name="High Contrast",
        colors={
            "accent": "ansibrightcyan",
            "muted": "ansiwhite",
            "success": "ansibrightgreen",
            "warning": "ansibrightyellow",
            "error": "ansibrightred",
            "reasoning": "ansibrightmagenta",
            "tool": "ansibrightblue",
        },
    ),
}


_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def _prefs_lock_path() -> Path:
    return UI_PREFS_PATH.parent / ".tars-ui-prefs.lock"


def _write_prefs_unlocked(values: dict[str, str], anchor) -> None:
    theme = values.get("theme", "terminal")
    logo = values.get("logo", "auto")
    atomic_write_anchored_text(
        anchor,
        UI_PREFS_PATH.name,
        f'theme = "{theme}"\nlogo = "{logo}"\n',
    )


def _ensure_prefs_unlocked(anchor) -> None:
    if not regular_file_exists(anchor, UI_PREFS_PATH.name):
        _write_prefs_unlocked({"theme": "terminal", "logo": "auto"}, anchor)


def ensure_ui_store() -> None:
    with installation_transaction(STATE_ROOT):
        THEME_ROOT.mkdir(parents=True, exist_ok=True)
        with exclusive_file_lock(_prefs_lock_path()) as anchor:
            _ensure_prefs_unlocked(anchor)


def _load_prefs_unlocked(anchor) -> dict[str, str]:
    _ensure_prefs_unlocked(anchor)
    try:
        raw = tomllib.loads(read_anchored_text(anchor, UI_PREFS_PATH.name))
    except (OSError, tomllib.TOMLDecodeError):
        raw = {}
    theme = str(raw.get("theme", "terminal"))
    logo = str(raw.get("logo", "auto"))
    if logo not in VALID_LOGOS:
        logo = "auto"
    return {"theme": theme, "logo": logo}


def load_ui_prefs() -> dict[str, str]:
    with installation_transaction(STATE_ROOT):
        with exclusive_file_lock(_prefs_lock_path()) as anchor:
            return _load_prefs_unlocked(anchor)


def set_theme(theme_id: str) -> Theme:
    with installation_transaction(STATE_ROOT):
        theme = get_theme(theme_id)
        with exclusive_file_lock(_prefs_lock_path()) as anchor:
            prefs = _load_prefs_unlocked(anchor)
            prefs["theme"] = theme.id
            _write_prefs_unlocked(prefs, anchor)
    return theme


def set_logo(mode: str) -> str:
    mode = mode.strip().lower()
    if mode not in VALID_LOGOS:
        raise ValueError(f"invalid logo mode {mode!r}; choose: {', '.join(VALID_LOGOS)}")
    with installation_transaction(STATE_ROOT):
        with exclusive_file_lock(_prefs_lock_path()) as anchor:
            prefs = _load_prefs_unlocked(anchor)
            prefs["logo"] = mode
            _write_prefs_unlocked(prefs, anchor)
    return mode


def _load_custom_theme(path: Path) -> Theme:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    theme_id = str(raw.get("id") or path.stem).strip().lower()
    if not _ID_RE.match(theme_id):
        raise ValueError(f"invalid theme id in {path.name}: {theme_id!r}")
    name = str(raw.get("name") or theme_id)
    colors_raw = raw.get("colors") or {}
    if not isinstance(colors_raw, dict):
        raise ValueError(f"[colors] must be a table in {path.name}")
    colors = {str(k): str(v) for k, v in colors_raw.items()}
    return Theme(theme_id, name, colors, source=str(path))


def list_themes() -> list[Theme]:
    ensure_ui_store()
    found = dict(_BUILTINS)
    for path in sorted(THEME_ROOT.glob("*.toml")):
        try:
            theme = _load_custom_theme(path)
        except Exception:
            continue
        if theme.id in _BUILTINS:
            continue
        found[theme.id] = theme
    return [found[key] for key in sorted(found)]


def get_theme(theme_id: str | None = None) -> Theme:
    ensure_ui_store()
    if theme_id is None:
        theme_id = load_ui_prefs()["theme"]
    wanted = str(theme_id).strip().lower()
    for theme in list_themes():
        if theme.id == wanted:
            return theme
    raise KeyError(f"unknown theme: {theme_id}")


def current_theme() -> Theme:
    prefs = load_ui_prefs()
    try:
        return get_theme(prefs["theme"])
    except KeyError:
        # Broken/deleted custom theme should never make chat unusable.
        return get_theme("terminal")


def current_logo() -> str:
    return load_ui_prefs()["logo"]


def prompt_toolkit_style(theme: Theme) -> dict[str, str]:
    c = {
        "accent": "",
        "muted": "",
        "success": "",
        "warning": "",
        "error": "",
        "reasoning": "",
        "tool": "",
        **theme.colors,
    }

    def fg(value: str) -> str:
        value = str(value or "").strip()
        return value

    accent = fg(c["accent"])
    muted = fg(c["muted"])
    success = fg(c["success"])
    warning = fg(c["warning"])
    error = fg(c["error"])
    reasoning = fg(c["reasoning"])
    tool = fg(c["tool"])

    def join(*parts: str) -> str:
        return " ".join(p for p in parts if p)

    return {
        "frame.border": accent,
        "frame.label": join("bold", accent),
        "prompt": join("bold", accent),
        "logo": join("bold", accent),
        "role": "bold",
        "model": accent,
        "dim": muted,
        "ok": success,
        "warn": warning,
        "bad": join("bold", error),
        "accent": accent,
        "context.good": success,
        "context.mid": warning,
        "context.high": join("bold", warning),
        "context.critical": join("bold", error),
        "footer": muted,
        "command": join("bold", accent),
        "meta": muted,
        "reasoning": reasoning,
        "tool": tool,
    }
