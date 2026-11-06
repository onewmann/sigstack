# Planning process

How the plan stays current. The plan is data (`backlog.toml`, `plan.toml`); forecasts are computed, never typed in. Every change of direction gets a line in `DECISIONS.md`.

## Cadence

| When | What | Tool | Output |
|---|---|---|---|
| Every work session | one log line in the sprint file (date, task, time spent, blocker) | by hand | `docs/sprints/sprint-NN.md` → Log |
| Middle of the sprint (end of week 1) | refine the stories of the next sprint until they meet the Definition of Ready | `/refine` | `backlog.toml`, `plan.py --check` green |
| Last day of the sprint | review: statuses, measured points, retro, new forecast | `/sprint-review N` | sprint file, `plan.toml` `[[actual]]`, `history.csv`, `PLAN.md` |
| After the review, only if a trigger fires | rework the plan | `/replan` | options, decision, `DECISIONS.md` |
| First day of the sprint | commit the stories, task breakdown with prompts, issues | `/plan-sprint N` | `sprint-NN.md`, issues |
| End of a release | tag, release notes, README results, CV line, vault mirror | `/release vX.Y.Z` (manual until S0.7) | tag, GitHub release |
| Start of each semester | update calendar periods and time budget | by hand | `plan.toml` |

## Replanning triggers

`/sprint-review` checks these and recommends `/replan` when one fires.

| Trigger | Threshold |
|---|---|
| Velocity off plan | measured velocity differs by more than 25 % from the planned value over two sprints |
| Milestone slip | a CV milestone or v1.0 moves by more than one sprint compared with the previous row in `history.csv` |
| Carry-over | a story is carried over a second time |
| Spike result | a spike changes the design or the dependencies |
| Release size | `plan.py` warns that a release exceeds three sprints |
| Context change | new semester dates, specialisation chosen in the master, a job application with a deadline, new hardware |

## Replanning options, in this order

1. Split stories that are too large (each part delivers something testable).
2. Reorder inside a release.
3. Move a group to a later release in the active profile.
4. Switch the active profile (`settings.profile` in `plan.toml`).
5. Drop a story: `status = "dropped"`, reason in `notes`, line in `DECISIONS.md`. Ids are never reused.

Not an option: weakening the Definition of Done or the tests to save time.

## Rules for the data files

- `PLAN.md`, `BACKLOG.md` and `history.csv` are generated. Do not edit them.
- Story status moves `todo → committed → done` (or `dropped`). `committed` and `done` carry the sprint number.
- Unfinished committed stories go back to `todo` at the review and get "carried over from sprint N" in `notes`.
- New stories get the next free id in their epic and need story, acceptance criteria and points before they enter a sprint.
- Plan changes are committed separately with `docs(plan): ...`, so `git log -- docs/plan` is the change history of the plan.
- `python scripts/plan.py --check` is part of `pixi run check` and of CI.

## Velocity

- Before the first review: `default_velocity` (8 points per full sprint).
- After that: mean of up to the last three measured sprints, normalised by their capacity factor (a half sprint with 4 done points counts as 8).
- The trigger "velocity off plan" compares against `planned_velocity` in `plan.toml`. When `/replan` accepts a new velocity as realistic, it sets `planned_velocity` to that value and logs the decision; otherwise the trigger would keep firing.
- `plan.py` evaluates all triggers and prints them as `TRIGGER ...`; `PLAN.md` shows them under *Replanning check*.
- Capacity of a sprint = velocity × factor of the calendar period (1.0 lecture time, 0.5 last lecture weeks, 0 exams and holidays, 1.5 lecture-free time).

## Mirroring to the Obsidian vault

The repository is the source of truth. After each review and each release, mirror in a Claude session that is linked to x1 ("sigstack in den Vault spiegeln"):

| Repo | Vault (`sigstack/`) |
|---|---|
| `docs/ROADMAP.md` | `sigstack Roadmap.md` |
| `docs/plan/PLAN.md` | `sigstack Plan.md` |
| `docs/plan/BACKLOG.md` | `sigstack Backlog.md` |
| `docs/plan/PLANNING.md` | `sigstack Planungsprozess.md` |
| `docs/plan/DECISIONS.md` | `sigstack Entscheidungslog.md` |
| `docs/sprints/sprint-NN.md` | `sigstack Sprint NN.md` |

Edits made in the vault are merged back into the repository before the next mirror, not overwritten.
