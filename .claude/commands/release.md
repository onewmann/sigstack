---
description: Cut a release - checks, changelog, tag, notes
argument-hint: <version, e.g. v0.2.0>
---

Prepare release **$ARGUMENTS**.

1. Run `python scripts/plan.py` and check in `PLAN.md` that every story of $ARGUMENTS is done. List any that are not and stop.
2. `pixi run check` on a clean checkout of `main`; CI on `main` green.
3. Move the `[Unreleased]` entries of `CHANGELOG.md` under `## [$ARGUMENTS] - <date>`; bump the version in `pyproject.toml` and `python/sigstack/__init__.py`.
4. Update the README if the release adds a headline result (figure, benchmark table, example). Numbers only from generated results, with hardware stated.
5. Draft a one-line CV/LinkedIn entry for this release (plain, technical, no marketing words) and show it to Oli.
6. Commit `chore(release): $ARGUMENTS` on branch `chore/release-$ARGUMENTS`, open a PR. After merge, Oli tags: `git tag -a $ARGUMENTS -m "$ARGUMENTS" && git push origin $ARGUMENTS` (from S0.7 on, the tag builds the GitHub release).
7. Remind Oli to mirror the plan to the vault.
