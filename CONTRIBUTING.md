# Contributing

## Local checks

Run all checks before every commit:

```bash
pixi run check        # lint + typecheck + tests
pixi run fmt          # auto-fix, then format
```

Install the git hooks once per clone, so the same checks run on `git commit`:

```bash
pixi run pre-commit install
pixi run pre-commit run --all-files   # optional: run all hooks on the whole tree
```

The hooks call ruff and mypy through `pixi run`, so `pixi` must be on `PATH` when you commit.
