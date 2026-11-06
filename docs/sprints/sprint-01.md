# Sprint 1 — 09.11. – 22.11.2026

**Sprint goal:** `pixi install && pixi run check` works on Fedora and Windows, CI runs on every pull request, the `Block` interface and window functions exist.

**Stories:** S0.1, S0.2, S0.5, S1.1, S1.3 (8 points)

| ID | Task | Story | Who | Est. | Status |
|---|---|---|---|---|---|
| T0 | Prerequisites and repo creation | S0.1 | Oli | 0.5 h | [ ] |
| T1 | Repository skeleton | S0.1 | Claude Code | 0.5 h | [ ] |
| T2 | pixi environment and tasks | S0.1 | Claude Code + Oli (Windows check) | 1.5 h | [ ] |
| T3 | Linting, formatting, type checking, pre-commit | S0.5 | Claude Code | 1 h | [ ] |
| T4 | GitHub Actions CI (Ubuntu + Windows) | S0.2 | Claude Code | 1.5 h | [ ] |
| T5 | GitHub hygiene: templates, CONTRIBUTING, branch protection | S0.2 | Claude Code + Oli | 0.5 h | [ ] |
| T6 | `Block` protocol and `Gain` block | S1.1 | Claude Code | 2 h | [ ] |
| T7 | Window functions | S1.3 | Claude Code | 1.5 h | [ ] |
| T8 | Backlog on GitHub (labels, milestones, issues, board) | infra | Oli (script) | 1 h | [ ] |
| T9 | Review, retro, tag `v0.0.1` | — | Oli | 0.5 h | [ ] |

Order: T0 → T1 → T2 → T3 → T4 → T5 → T8 (any time after T1), then T6 and T7 (independent of each other).

How to run a task in Claude Code: `/task T1` (custom command in `.claude/commands/task.md`), or paste the prompt block below.

---

## T0 — Prerequisites (manual)

1. Install tools on Fedora (and later on the Windows laptop):
   ```bash
   curl -fsSL https://pixi.sh/install.sh | bash      # pixi
   sudo dnf install gh git                            # GitHub CLI
   gh auth login
   ```
   Claude Code per the official install instructions.
2. Check the name: `sigstack` on GitHub and PyPI. If taken, pick another and replace it in `CLAUDE.md`, `docs/`, and later in the code.
3. Create the repo (public, empty, no README):
   ```bash
   gh repo create onewmann/sigstack --public --description "Signal processing toolkit: block-based DSP, C++ core, Monte Carlo design loop"
   git clone git@github.com:onewmann/sigstack.git && cd sigstack
   ```
4. Set the noreply author for this repo (number from GitHub → Settings → Emails):
   ```bash
   git config user.name "Oli ..."
   git config user.email "<ID>+onewmann@users.noreply.github.com"
   ```
5. Copy the content of this kickoff kit into the repo root, commit on `main`:
   ```bash
   git add . && git commit -m "chore: add roadmap, sprint plan and Claude Code config" && git push -u origin main
   ```

**Done when:** repo exists, kickoff files are on `main`, `git log` shows the noreply address.

---

## T1 — Repository skeleton

```text
Task T1 (Sprint 1, story S0.1): create the repository skeleton.

Read CLAUDE.md and docs/ROADMAP.md (sections 2.4, 2.6) first.

Create:
- LICENSE: Apache-2.0, copyright holder "Oli" (leave a TODO for the full name), year 2026
- README.md stub: title, one-sentence description, "Status: early development", sections
  Install / Quickstart / Roadmap (link docs/ROADMAP.md) / License; no badges yet
- CHANGELOG.md in Keep a Changelog format with an [Unreleased] section
- .gitignore for Python, C++/CMake build dirs, pixi (.pixi/), IDE folders, OS files
- .editorconfig (utf-8, lf, 4 spaces for py/cpp, 2 for yaml/toml/json, final newline)
- pyproject.toml with hatchling as build backend (temporary; replaced by scikit-build-core
  in story S0.3), project name "sigstack", version "0.0.1", requires-python ">=3.11",
  dependencies numpy and scipy, package located at python/sigstack
- python/sigstack/__init__.py with __version__ and an empty __all__
- python/sigstack/core/__init__.py (empty package for T6/T7)
- python/sigstack/py.typed
- tests/test_import.py: imports sigstack and checks __version__

Out of scope: pixi, CI, linting config (next tasks).

Done when:
- `python -m pip install -e .` in a fresh venv works and `pytest` passes
- tree matches CLAUDE.md layout
Branch: chore/T1-skeleton. One commit "chore: add repository skeleton". Open a PR with gh.
```

---

## T2 — pixi environment

```text
Task T2 (Sprint 1, story S0.1): set up the pixi environment.

Check the current pixi documentation for the manifest format before writing (pixi.toml,
[workspace] vs [project] table). Use conda-forge.

Requirements:
- platforms: linux-64, win-64 (osx-arm64 optional, only if it costs nothing)
- dependencies: python 3.12, numpy, scipy, pytest, hypothesis, ruff, mypy, pre-commit
- the package itself installed editable via [pypi-dependencies]
- tasks: test, lint, fmt, typecheck, check (= lint + typecheck + test, via depends-on)
- commit pixi.lock
- README: Install section with `pixi install` and `pixi run check`

Do not add compilers or CMake yet (they come with the C++ work).

Done when:
- `pixi install` from a clean clone works
- `pixi run test` passes
- `pixi run check` exists (lint/typecheck may still be no-ops until T3; say so in the PR)
Branch: chore/T2-pixi. PR with a short description.
Note for Oli: run `pixi install && pixi run test` on the Windows laptop and report back.
```

---

## T3 — Lint, format, type check, pre-commit

```text
Task T3 (Sprint 1, story S0.5): code quality tooling.

- ruff config in pyproject.toml: line length 100, target py311, rules E, F, W, I, B, UP,
  N, NPY, RUF, D (numpy docstring convention; D rules not enforced on tests/)
- ruff format as formatter
- mypy config in pyproject.toml: strict = true for python/sigstack, relaxed for tests/,
  numpy plugin not needed (numpy ships types)
- .pre-commit-config.yaml: ruff (check --fix), ruff-format, mypy (local hook through pixi
  so it uses the same environment), end-of-file-fixer, trailing-whitespace,
  check-yaml, check-toml, check-added-large-files (500 kB)
- .clang-format (Google style, 4-space indent, column limit 100) for the later C++ code
- pixi tasks lint / fmt / typecheck wired to the real tools
- CONTRIBUTING.md section "Local checks": `pixi run pre-commit install`

Done when:
- `pixi run check` passes on the current code
- `pre-commit run --all-files` passes
Branch: chore/T3-quality.
```

---

## T4 — CI

```text
Task T4 (Sprint 1, story S0.2): GitHub Actions CI.

Workflow .github/workflows/ci.yml:
- triggers: push to main, pull_request
- concurrency group per ref, cancel in progress
- matrix: ubuntu-latest, windows-latest
- steps: checkout, prefix-dev/setup-pixi (check the current major version and its
  cache option), `pixi run lint`, `pixi run typecheck`, `pixi run test`
- pytest writes a JUnit XML; upload it as artifact on failure
- minimal permissions (contents: read)

Also:
- README: CI badge
- a second tiny workflow is NOT needed; keep everything in ci.yml

Done when:
- the PR for this task shows green checks on both OS
- a deliberately failing test (local experiment, not committed) makes CI red
Branch: ci/T4-github-actions.
```

---

## T5 — GitHub hygiene

```text
Task T5 (Sprint 1, story S0.2): repository conventions.

Create:
- .github/pull_request_template.md: summary, linked issue ("Closes #"), checklist from the
  Definition of Done in docs/ROADMAP.md section 3.5
- .github/ISSUE_TEMPLATE/story.yml (fields: story ID, user story, acceptance criteria,
  points dropdown 1/2/3/5/8, epic)
- .github/ISSUE_TEMPLATE/bug.yml (steps, expected, actual, environment)
- .github/ISSUE_TEMPLATE/spike.yml (question, time box, expected output)
- .github/ISSUE_TEMPLATE/config.yml (blank issues disabled)
- CONTRIBUTING.md: branch naming, Conventional Commits, squash merge, local checks,
  Definition of Done (link)

Then print, but do not run, the gh commands Oli needs to protect main:
require PR, require status checks from ci.yml on both OS, no force pushes.

Done when: templates render on GitHub (check after merge), commands are in the PR description.
Branch: docs/T5-repo-hygiene.
```

---

## T6 — `Block` protocol and `Gain` block

```text
Task T6 (Sprint 1, story S1.1): the block interface.

Read docs/ROADMAP.md section 2.2. Design for stories S1.2 (chunking test) and S1.4
(complex + multichannel) already now, even though they are implemented in Sprint 2.

Create python/sigstack/core/block.py:
- `Block` as typing.Protocol (runtime_checkable) with:
  fs_in: float, fs_out: float, latency (property, int, samples),
  reset() -> None, process(x: NDArray) -> NDArray
- contract in the docstring:
  * samples on the last axis, any leading axes are channels / batch
  * process() keeps state between calls
  * chunk invariance: process(x1) ++ process(x2) == process(x1 ++ x2) for any split
  * reset() restores the initial state
  * real and complex input supported unless documented otherwise
- `BaseBlock` abstract helper class (optional, keep it small) that validates fs > 0
  and input arrays (ndim >= 1, numeric dtype)
- `Gain` block (gain: float | complex): stateless, latency 0, fs_out == fs_in
- export Block, BaseBlock, Gain from sigstack.core and sigstack

Tests tests/core/test_block.py:
- Gain satisfies isinstance(g, Block)
- real, complex, 1-D and (3, N) input, dtype preserved where sensible
- chunked == one-shot for random split points (plain loop with fixed seed is fine;
  the generic hypothesis-based test is story S1.2)
- invalid input raises ValueError / TypeError with a clear message

Done when: `pixi run check` passes, API docstrings complete.
Branch: feat/T6-block-protocol. Show me the Protocol design (signatures only) before
implementing the tests.
```

---

## T7 — Window functions

```text
Task T7 (Sprint 1, story S1.3): window functions.

Create python/sigstack/core/windows.py, implemented from the closed-form definitions
(Harris 1978, "On the use of windows for harmonic analysis with the DFT"; Oppenheim/Schafer),
not by calling scipy:
- rectangular, hann, hamming, blackman, blackman_harris (4-term), flattop, kaiser(beta)
- signature: get_window(name: str, n: int, *, sym: bool = False, **params) -> NDArray[float64]
  sym=False gives the periodic (DFT-even) window, like scipy's default fftbins=True
- generalised cosine helper: w[k] = sum_m (-1)^m a_m cos(2 pi m k / L)
- kaiser via numpy.i0
- coherent gain and equivalent noise bandwidth (ENBW, in bins) as helper functions:
  CG = (1/N) sum w[n],  ENBW = N sum w[n]^2 / (sum w[n])^2

Tests tests/core/test_windows.py:
- every window vs scipy.signal.get_window for n in {1, 2, 7, 64, 255}, sym True/False,
  atol 1e-12 (comment: pure float64 arithmetic, only rounding differences)
- known ENBW values (e.g. Hann 1.5 bins, rectangular 1.0) to 1e-3
- n < 1 and unknown names raise ValueError

Done when: `pixi run check` passes, each window has a docstring with its formula and source.
Branch: feat/T7-windows.
```

---

## T8 — Backlog on GitHub (manual, scripted)

```bash
bash scripts/create_backlog.sh          # labels, milestones, issues for Sprint 1 and 2
```

Then in the browser: create a Project (board) "sigstack", add fields *Points* (number), *Epic* (single select), *Sprint* (iteration, 2 weeks, start 09.11.2026), add all issues.

---

## T9 — Review and retro

- Run `pixi run check` on Fedora and Windows from a fresh clone.
- Update CHANGELOG, tag `v0.0.1`, push the tag.
- Retro (in this file, below): one thing to keep, one thing to change, measured points.

### Review notes

- Done points: 
- Not done:

### Retro

- Keep:
- Change:
