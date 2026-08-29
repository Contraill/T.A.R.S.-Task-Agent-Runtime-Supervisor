# Development notes

The public Git history begins at v0.4.2. Earlier experimental installers and private calibration logs are not copied into the repository as fake history.

## Source vs installed tree

Keep the clone separate from user state:

```text
~/Projects/tars/                 source checkout
~/.config/tars/                  user configuration
~/.local/share/tars/             installed/share data and models
~/.local/state/tars/             conversations/tasks/calibration
~/.cache/tars/                   cache
```

Do not commit anything from the XDG state tree.

## Tests

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
python -m compileall -q src
```

Until the generic setup wizard lands, end-to-end runtime tests are expected to use a configured development machine or an isolated fixture environment.
