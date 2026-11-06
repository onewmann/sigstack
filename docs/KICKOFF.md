# sigstack kickoff kit

Copy the content of this folder into the root of the new repository (task T0 in `docs/sprints/sprint-01.md`). In the vault copy, the Claude Code config is stored as `claude-config/`; rename it to `.claude/` in the repo. Delete this file afterwards or keep it as `docs/KICKOFF.md`.

## Contents

| File | Purpose |
|---|---|
| `CLAUDE.md` | Project rules for Claude Code: conventions, commands, planning rules |
| `docs/ROADMAP.md` | Vision, architecture, way of working, release strategy, risks, open decisions |
| `docs/plan/backlog.toml` | All epics and stories (single source of truth) |
| `docs/plan/plan.toml` | Calendar with capacity factors, velocity, measured points, release profiles |
| `docs/plan/PLAN.md`, `BACKLOG.md` | Generated: release forecast, sprint schedule, profile comparison, backlog tables |
| `docs/plan/history.csv` | Generated: forecast per sprint review |
| `docs/plan/PLANNING.md` | How the plan is kept current: cadence, triggers, replanning options |
| `docs/plan/DECISIONS.md` | Decision log |
| `docs/sprints/sprint-01.md` | Sprint 1 tasks T0–T9 with ready prompts |
| `docs/sprints/sprint-02.md` | Sprint 2 draft (committed via `/plan-sprint 2`) |
| `scripts/plan.py` | Validate backlog, schedule sprints, write PLAN/BACKLOG (standard library only) |
| `scripts/sync_issues.py` | Labels, milestones and issues from the backlog via `gh` (dry run by default) |
| `.claude/commands/` | `/task`, `/review`, `/refine`, `/plan-sprint`, `/sprint-review`, `/replan`, `/release` |
| `.claude/settings.json` | Allows pixi, `plan.py` and read-only git/gh; blocks force-push and git config |

## Working with Claude Code

1. Start Claude Code in the repo root: `claude`
2. One task per session: `/task T1`. `/clear` before the next task.
3. For design-heavy tasks (T6/T7 in Sprint 1, T2 in Sprint 2) let Claude Code show the design first.
4. After each task: `/review <task-id>`, read the diff, squash-merge the PR.
5. Sprint rhythm: `/refine` mid-sprint, `/sprint-review N` on the last day, `/replan` only if a trigger fires, `/plan-sprint N+1` on the first day.

## Check the plan

```bash
python3 scripts/plan.py --check     # validation only
python3 scripts/plan.py             # regenerate docs/plan/PLAN.md and BACKLOG.md
python3 scripts/plan.py --profile showcase --no-write   # other release order
```

## Defaults (change before T1 if needed)

Name `sigstack`, license Apache-2.0, Python ≥ 3.11 (environment 3.12), hatchling until S0.3, Sprint 1 from Monday 09.11.2026, velocity 8 points per sprint, profile `default`. All listed in `docs/plan/DECISIONS.md`.
