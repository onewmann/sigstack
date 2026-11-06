---
description: Close a sprint - statuses, measured points, retro, new forecast
argument-hint: <sprint number>
---

Run the sprint review for sprint **$ARGUMENTS**. Follow `docs/plan/PLANNING.md`.

1. Read `docs/sprints/sprint-$ARGUMENTS.md` (two-digit file name), `docs/plan/backlog.toml`, `docs/plan/plan.toml`, the last rows of `docs/plan/history.csv`.
2. Collect evidence for the sprint dates: `gh pr list --state merged --limit 50 --json number,title,mergedAt,body` and `gh issue list --state closed --limit 50 --json number,title,closedAt`.
3. For each story committed to this sprint decide: done (every acceptance criterion met, merged PR, Definition of Done) or not done. Show the table (story, evidence, verdict) and **wait for Oli to confirm** before changing files.
4. In `backlog.toml`: set `status = "done"` for finished stories. For unfinished ones set `status = "todo"`, remove `sprint`, append "carried over from sprint $ARGUMENTS" to `notes`.
5. In `plan.toml`: add an `[[actual]]` entry with `sprint`, `done_points` (sum of done stories) and a short `note`.
6. Ask Oli the retro questions and wait for the answers: actual hours spent, one thing to keep, one thing to change, anything that changes priorities. Write review notes and retro into the sprint file.
7. Run `python scripts/plan.py --record`. Compare the new forecast with the previous `history.csv` row and `git diff docs/plan/PLAN.md`.
8. Read the `TRIGGER` lines printed by `plan.py` and the *Replanning check* table in `PLAN.md`. Report: velocity (planned vs measured), forecast changes per milestone, triggers fired. If a trigger fired, recommend `/replan` with the reason; do not replan yourself.
9. Commit on branch `docs/sprint-$ARGUMENTS-review` with `docs(plan): review sprint $ARGUMENTS` and open a PR.
