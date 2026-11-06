# Contributing

## Workflow

1. Every change starts from an issue (story, bug or spike) and goes through a pull request. `main` is protected: no direct pushes, no force pushes, CI must pass on Ubuntu and Windows.
2. Create a short-lived branch from an up-to-date `main`:

   | Prefix | Use |
   |---|---|
   | `feat/<task-id>-<slug>` | new functionality, e.g. `feat/T6-block-protocol` |
   | `fix/<issue>-<slug>` | bug fixes, e.g. `fix/12-window-length-one` |
   | `docs/<task-id>-<slug>` | documentation only, e.g. `docs/T5-repo-hygiene` |
   | `ci/<task-id>-<slug>` | CI and workflows, e.g. `ci/T4-github-actions` |
   | `chore/<task-id>-<slug>` | tooling, build, repository setup, e.g. `chore/T1-skeleton` |

3. Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/): `feat:`, `fix:`, `test:`, `docs:`, `ci:`, `chore:`, `refactor:`. The changelog is written from them.
4. Add an entry under `## [Unreleased]` in `CHANGELOG.md`.
5. Open the pull request with the template and link the issue (`Closes #<n>`).
6. Pull requests are merged with **squash merge**; the PR title becomes the commit message on `main`, so it must be a Conventional Commit too.

A change is done when it meets the [Definition of Done](docs/ROADMAP.md#35-definition-of-done).

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
