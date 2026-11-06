# CLAUDE.md

Instructions for Claude Code in this repository.

## Project

**sigstack**: C++/Python toolkit for signal processing with a uniform block-processing model (offline and streaming), plus a simulation-based design loop (Monte Carlo, optimisation). Architecture and process: `docs/ROADMAP.md`. Backlog and schedule: `docs/plan/` (`backlog.toml`, `plan.toml`, generated `PLAN.md`). Planning process: `docs/plan/PLANNING.md`. Current sprint: highest-numbered file in `docs/sprints/`.

## Ground rules

- All code is written from scratch, based on textbooks and public documentation. Do not copy code from other libraries; SciPy is used as a **test reference**, not as the implementation.
- No code, data or parameters from any thesis or employer.
- Do not add a dependency without asking first. Allowed so far: numpy, scipy, pytest, hypothesis, ruff, mypy, pre-commit.
- Never change git config (author is set by the owner). Never force-push, never push to `main` directly.

## Layout

```
python/sigstack/   Python package (src-style layout)
tests/             pytest tests, mirror the package structure
cpp/               C++ core (from S0.3/S6.1 on, empty until then)
docs/              ROADMAP.md, plan/, sprints/, later MkDocs
scripts/           plan.py (backlog check and schedule), sync_issues.py (GitHub issues)
```

## Commands

All tooling runs through pixi.

| Command | Purpose |
|---|---|
| `pixi install` | create / update environment |
| `pixi run test` | pytest |
| `pixi run lint` | ruff check + ruff format --check |
| `pixi run fmt` | ruff check --fix, then ruff format |
| `pixi run typecheck` | mypy |
| `pixi run plan` | validate backlog, regenerate `docs/plan/PLAN.md` and `BACKLOG.md` |
| `pixi run check` | lint + typecheck + test + `plan.py --check` (run before every commit) |

Before T2 exists: `python3 scripts/plan.py` works with the system Python (≥ 3.11, standard library only).

## Code conventions

- Python ≥ 3.11, full type hints, `numpy.typing.NDArray`.
- Docstrings in NumPy style. Formulas in docstrings as LaTeX (`:math:` or `$...$`), with a reference (book, chapter or equation).
- Array convention: **samples on the last axis**, shape `(..., n_samples)`; multichannel = `(n_channels, n_samples)`.
- Default dtype `float64`. Every block must also accept `complex128`, unless documented otherwise.
- No hidden global state; randomness only through an explicit `numpy.random.Generator` or seed argument.
- Public API exported in the package `__init__.py` with `__all__`.

## Testing rules

- Every public function has a test.
- Numerical comparisons use `numpy.testing.assert_allclose` with an explicit `rtol`/`atol` and a short comment why that tolerance.
- Stateful blocks must pass the chunking test: chunked processing equals one-shot processing.
- Tests must be deterministic (fixed seeds).

## Workflow per task

1. Read the task in the sprint file. Restate scope and the "Done when" list before coding.
2. Work on a branch `feat/<task-id>-<slug>` (or `chore/`, `docs/`, `ci/`).
3. Small commits, Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, `ci:`, `chore:`, `refactor:`).
4. Run `pixi run check`. A task is not done while it fails.
5. Add an entry under `## [Unreleased]` in `CHANGELOG.md`.
6. Summarise: what changed, how it was verified, open points. Do not tick tasks that are not verified.

## Planning rules

- Never edit `docs/plan/PLAN.md`, `BACKLOG.md` or `history.csv` by hand; they are generated.
- Story status changes only through `/sprint-review`, `/plan-sprint`, `/replan`, `/refine`, and only after Oli confirms.
- If a task reveals that a story is wrong (scope, estimate, missing dependency), say so in the summary and propose the backlog change; do not silently change scope.
- Plan changes are separate commits: `docs(plan): ...`.

## Custom commands

`/task <id>`, `/review <id>`, `/refine`, `/plan-sprint <n>`, `/sprint-review <n>`, `/replan <reason>`, `/release <version>`.

## Writing style for docs and README

Plain, technical English. No marketing language, no emojis, no filler sentences.
