# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `docs/conventions.md` with the block contract: array layout, empty chunks, dtype and precision rules (D-016, D-017), shape-changing blocks, complex opt-out.
- `BaseBlock.SUPPORTS_COMPLEX` (complex input rejected with `TypeError` when `False`) and `BaseBlock.output_shape(input_shape)`.

## [0.0.1] - 2026-11-07

Sprint 1: tooling, CI, the block interface and window functions.

### Added

- Window functions in `sigstack.core.windows`: rectangular, Hann, Hamming, Blackman, 4-term Blackman-Harris, flat-top and Kaiser, symmetric or periodic (DFT-even), from their closed-form definitions; `general_cosine`, `get_window`, `coherent_gain` and `enbw`. Agree with `scipy.signal.get_window` to 1e-12.
- `Block` protocol in `sigstack.core.block` with the block contract (samples on the last axis, state between calls, chunk invariance, reset, dtype rules), `BaseBlock` helper that validates sampling rates and input arrays, and the memoryless `Gain` block. Boolean and integer input is converted to `float64`. Exported from `sigstack` and `sigstack.core`.
- Planning system from the kickoff kit: `docs/plan/` (backlog and calendar as TOML, generated plan, planning process, decision log), `scripts/plan.py`, `scripts/sync_issues.py`, Sprint 2 draft, Claude Code commands `/refine`, `/plan-sprint`, `/sprint-review`, `/replan`, `/release`; pixi tasks `plan` and `plan-check`, the latter part of `check` and CI; Claude Code may run `plan.py`, `gh pr list` and `gh issue list` without asking. Replaces `scripts/create_backlog.sh`.
- Pull request template, issue forms (story, bug, spike), CONTRIBUTING with branch naming, commit and merge rules.
- GitHub Actions CI on Ubuntu and Windows: lint, type check, tests; JUnit report uploaded on failure. CI badge in the README.
- Code quality tooling: ruff (lint + format) and mypy (strict) configured in `pyproject.toml`, pre-commit hooks, `.clang-format`, CONTRIBUTING section on local checks.
- pixi environment (`pixi.toml`, `pixi.lock`) for linux-64 and win-64 with tasks `test`, `lint`, `fmt`, `typecheck`, `check`.
- Repository skeleton: license, README, changelog, `pyproject.toml` (hatchling), package `sigstack` with `core` subpackage, import test.

[Unreleased]: https://github.com/onewmann/sigstack/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/onewmann/sigstack/releases/tag/v0.0.1
