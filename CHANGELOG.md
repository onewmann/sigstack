# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Planning system from the kickoff kit: `docs/plan/` (backlog and calendar as TOML, generated plan, planning process, decision log), `scripts/plan.py`, `scripts/sync_issues.py`, Sprint 2 draft, Claude Code commands `/refine`, `/plan-sprint`, `/sprint-review`, `/replan`, `/release`; pixi tasks `plan` and `plan-check`, the latter part of `check` and CI. Replaces `scripts/create_backlog.sh`.
- Pull request template, issue forms (story, bug, spike), CONTRIBUTING with branch naming, commit and merge rules.
- GitHub Actions CI on Ubuntu and Windows: lint, type check, tests; JUnit report uploaded on failure. CI badge in the README.
- Code quality tooling: ruff (lint + format) and mypy (strict) configured in `pyproject.toml`, pre-commit hooks, `.clang-format`, CONTRIBUTING section on local checks.
- pixi environment (`pixi.toml`, `pixi.lock`) for linux-64 and win-64 with tasks `test`, `lint`, `fmt`, `typecheck`, `check`.
- Repository skeleton: license, README, changelog, `pyproject.toml` (hatchling), package `sigstack` with `core` subpackage, import test.
