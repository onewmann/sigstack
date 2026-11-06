---
description: Plan a sprint - commit stories, write task breakdown with prompts, sync issues
argument-hint: <sprint number>
---

Plan sprint **$ARGUMENTS**. Follow `docs/plan/PLANNING.md`.

1. Run `python scripts/plan.py` and read the stories scheduled for sprint $ARGUMENTS in `docs/plan/PLAN.md`, plus their entries in `backlog.toml` and the ROADMAP sections they touch.
2. Check each story against the Definition of Ready (ROADMAP 3.4). If one is not ready, stop and propose the refinement (split, clarify acceptance criteria, fix dependencies) instead of planning around it.
3. Propose the sprint goal (one sentence) and the story list with points against the capacity in `PLAN.md`. **Wait for Oli's OK.**
4. In `backlog.toml` set those stories to `status = "committed"`, `sprint = $ARGUMENTS`. Run `python scripts/plan.py` again; there must be no capacity warning for this sprint.
5. Write `docs/sprints/sprint-NN.md` (two-digit number) with the same structure as `docs/sprints/sprint-01.md`: goal, task table (T1, T2, ... with story, owner, estimate, status), one section per task with a fenced prompt block (scope, files, references, out of scope, "Done when", branch name), review and retro sections, a Log section. If a draft of this file exists, update it instead of starting over, and remove the `(draft)` marker from the title and the draft note.
6. Tasks that need a design decision say "show the design before implementing".
7. Run `python scripts/sync_issues.py --sprint $ARGUMENTS` (dry run), show the result, and only after Oli confirms run it with `--apply`.
8. Commit with `docs(plan): plan sprint $ARGUMENTS` on branch `docs/sprint-$ARGUMENTS-plan` and open a PR.
