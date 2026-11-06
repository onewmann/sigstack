# sigstack kickoff kit

Copy the content of this folder into the root of the new repository (task T0 in `docs/sprints/sprint-01.md`). In the vault copy, the Claude Code config is stored as `claude-config/`; rename it to `.claude/` in the repo. Delete this file afterwards or keep it as `docs/KICKOFF.md`.

## Contents

| File | Purpose |
|---|---|
| `CLAUDE.md` | Project rules for Claude Code: conventions, commands, workflow per task |
| `docs/ROADMAP.md` | Architecture and agile plan |
| `docs/sprints/sprint-01.md` | Sprint 1 tasks T0–T9, each with a ready prompt and "Done when" list |
| `.claude/commands/task.md` | `/task T6` runs a task from the sprint file |
| `.claude/commands/review.md` | `/review T6` checks the branch against the Definition of Done |
| `.claude/settings.json` | Allows pixi/git/gh read commands, blocks force-push and git config changes |
| `scripts/create_backlog.sh` | Creates labels, milestones and the Sprint 1+2 issues via `gh` |

## Working with Claude Code

1. Start Claude Code in the repo root: `claude`
2. One task per session: `/task T1`. Use `/clear` before the next task so context stays small.
3. For T6 and T7, let Claude Code show the design first (plan mode) and approve it before it writes code.
4. After each task: `/review <task-id>`, read the diff yourself, then merge the PR (squash).
5. You do T0, the Windows check in T2, branch protection in T5, T8 and T9 yourself.

## Decisions taken as defaults (change before T1 if needed)

- Name `sigstack`, license Apache-2.0, Python ≥ 3.11 (environment pins 3.12)
- hatchling as a temporary build backend; scikit-build-core follows with the C++ part (S0.3)
- Sprint 1 starts Monday, 09.11.2026

Sprint 2 is currently planned at 11 points, above the assumed velocity of 8. After Sprint 1, move S0.3 to Sprint 3 if the measured velocity is about 8.
