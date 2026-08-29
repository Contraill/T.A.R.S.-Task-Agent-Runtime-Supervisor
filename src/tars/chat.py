"""Chat UI selector.

Enhanced full-screen UI uses prompt_toolkit when available. The classic Rich/readline
UI remains as a zero-dependency fallback.
"""


def run_chat(cfg, *, initial_role=None):
    try:
        from .chat_tui import run_chat as run_enhanced
    except ImportError:
        from .chat_classic import run_chat as run_classic
        return run_classic(cfg, initial_role=initial_role)
    return run_enhanced(cfg, initial_role=initial_role)
