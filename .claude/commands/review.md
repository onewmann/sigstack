---
description: Review the current branch against the Definition of Done
argument-hint: [optional task-id]
---

Review the changes of the current branch against `main` (`git diff main...HEAD`). Task context: $ARGUMENTS.

Check and report as a table (item, status, note):

- Definition of Done from `docs/ROADMAP.md` section 3.5
- Conventions in `CLAUDE.md` (array axis convention, dtypes, type hints, docstrings with formula and source, explicit tolerances, deterministic tests)
- Numerical correctness: edge cases (empty input, length 1, odd/even lengths, complex input, multichannel), off-by-one in indices, dtype promotion
- Tests: does each test actually fail if the implementation is wrong?
- Scope: anything changed that the task did not ask for

Do not change code during the review. End with a short list of concrete fixes, most important first.
