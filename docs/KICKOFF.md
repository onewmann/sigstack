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
| `docs/plan/` | Backlog (`backlog.toml`), calendar and velocity (`plan.toml`), generated `PLAN.md` and `BACKLOG.md`, planning process, decision log |
| `scripts/plan.py` | Validates the backlog, schedules it into sprints, writes the plan |
| `scripts/sync_issues.py` | Creates labels, milestones and issues from the backlog via `gh` (idempotent) |
| `.claude/commands/{refine,plan-sprint,sprint-review,replan,release}.md` | Planning rhythm, see `docs/plan/PLANNING.md` |

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

Sprint 2 is planned at 8 points, matching the assumed velocity of 8: S0.3 (scikit-build-core) moved to the C++ work package (D-008 in `docs/plan/DECISIONS.md`, still to be confirmed).
