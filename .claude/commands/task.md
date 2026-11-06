---
description: Implement a task from the current sprint file
argument-hint: <task-id, e.g. T6>
---

Implement task **$ARGUMENTS** from the current sprint file in `docs/sprints/` (highest sprint number).

1. Read `CLAUDE.md`, the task section for $ARGUMENTS in the sprint file, and the ROADMAP sections it references.
2. Restate in a few lines: scope, files you will create or change, the "Done when" list, and anything unclear. If something is unclear or needs a new dependency, ask before coding.
3. Create the branch named in the task (`git switch -c ...` from an up-to-date `main`).
4. Implement in small commits (Conventional Commits).
5. Run `pixi run check` and fix everything until it passes.
6. Add a CHANGELOG entry under `[Unreleased]`.
7. Push the branch and open a PR with `gh pr create`. The description lists what changed, how it was verified, and the Definition of Done checklist.
8. In the sprint file, set the task status to `[x]` only if every "Done when" item is verified. Otherwise list what is missing.
