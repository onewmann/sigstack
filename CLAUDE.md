# CLAUDE.md

Instructions for Claude Code in this repository.

## Project

**sigstack**: C++/Python toolkit for signal processing with a uniform block-processing model (offline and streaming), plus a simulation-based design loop (Monte Carlo, optimisation). Full plan: `docs/ROADMAP.md`. Current sprint: `docs/sprints/sprint-01.md`.

## Ground rules

- All code is written from scratch, based on textbooks and public documentation. Do not copy code from other libraries; SciPy is used as a **test reference**, not as the implementation.
- No code, data or parameters from any thesis or employer.
- Do not add a dependency without asking first. Allowed so far: numpy, scipy, pytest, hypothesis, ruff, mypy, pre-commit.
- Never change git config (author is set by the owner). Never force-push, never push to `main` directly.

## Layout

```
python/sigstack/   Python package (src-style layout)
tests/             pytest tests, mirror the package structure
cpp/               C++ core (from Sprint 6 on, empty until then)
docs/              ROADMAP.md, sprints/, later MkDocs
```

## Commands

All tooling runs through pixi.

| Command | Purpose |
|---|---|
| `pixi install` | create / update environment |
| `pixi run test` | pytest |
| `pixi run lint` | ruff check + ruff format --check |
| `pixi run fmt` | ruff format + ruff check --fix |
| `pixi run typecheck` | mypy |
| `pixi run check` | lint + typecheck + test (run before every commit) |

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

## Writing style for docs and README

Plain, technical English. No marketing language, no emojis, no filler sentences.
