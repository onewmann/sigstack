---
description: Rework the plan when a replanning trigger fired
argument-hint: [reason]
---

Rework the plan. Reason given: $ARGUMENTS

1. Read `docs/plan/PLANNING.md`, `PLAN.md`, `history.csv`, `DECISIONS.md`, `plan.toml`, `backlog.toml` and the retros of the last two sprint files.
2. Diagnose: which trigger fired, how large the effect is (velocity gap, milestone slip in sprints and dates), what causes it.
3. Work out two or three options from the ordered list in `PLANNING.md` (split, reorder, move group, switch profile, drop). For each option, compute its effect: make the change on a scratch copy of the TOML files (or use `--profile`), run `python scripts/plan.py --no-write --root <copy>`, and report the milestone dates. Do not touch the real files yet.
4. Present the options as a table (change, effect on each CV milestone and v1.0, risk). Recommend one and say why. **Wait for Oli's choice.**
5. Apply the chosen option. If the decision accepts a different velocity as the new normal, set `planned_velocity` in `plan.toml`.
   Then update `backlog.toml` / `plan.toml`, add a `D-NNN` line to `DECISIONS.md`, run `python scripts/plan.py --record`.
6. If the current sprint is affected, update its sprint file.
7. Commit with `docs(plan): replan - <short reason>` on branch `docs/replan-<slug>` and open a PR.
