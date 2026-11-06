---
description: Mid-sprint refinement of the next sprint's stories
argument-hint: [sprint number to refine, default next]
---

Refine the stories for sprint **$ARGUMENTS** (if empty: the sprint after the current one).

1. Run `python scripts/plan.py` and list the stories scheduled for that sprint in `PLAN.md`.
2. For each story check: clear user value, testable acceptance criteria, points at most 5, dependencies correct and done or in an earlier position, design questions known. Look at the code that exists now; earlier estimates may be stale.
3. Propose changes as a table (story, problem, proposed change): sharper acceptance criteria, splits into new ids, new dependencies, re-estimates, notes on design. Mention stories two releases ahead whose estimate looks wrong given what was learned.
4. **Wait for Oli's OK**, then edit `backlog.toml` and run `python scripts/plan.py --check` and `python scripts/plan.py`.
5. Report the forecast change, if any. Commit with `docs(plan): refine sprint <n>`.
