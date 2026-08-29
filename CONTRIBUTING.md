# Contributing

T.A.R.S. is still changing quickly. Small, focused pull requests are easier to review than broad rewrites.

Before opening a pull request:

1. Open or reference an issue when the behavior change is non-trivial.
2. Keep runtime state, model files, credentials and personal memory out of the repository.
3. Add a test for deterministic behavior when practical.
4. Do not claim an external provider is tested unless it has actually been exercised and the support matrix says so.
5. Keep security decisions deterministic. Prompt instructions are not a substitute for ScopeGuard or permission checks.

Development setup:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
```

The project is Linux-first before 1.0. Platform support should be documented as tested only after somebody has run it successfully.
