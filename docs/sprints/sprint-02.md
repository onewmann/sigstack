# Sprint 2 — 09.11. – 22.11.2026

**Sprint goal:** every block is discoverable, typed and stream-safe: the block contract written down in `docs/conventions.md`, a block registry with parameter metadata, a generic chunking test over all registered blocks, and release v0.1.0. Reviewers can run the checks in GitHub Codespaces.

**Stories:** S1.4, S1.5, S1.2, S0.6 (8 points, capacity 8). Committed with `/plan-sprint 2` on 07.11.2026, after the refinement (#23, D-017 to D-019).

| ID | Task | Story | Who | Est. | Status |
|---|---|---|---|---|---|
| T1 | Block contract: complex opt-out, output shape, conventions | S1.4 | Claude Code | 1 h | [x] |
| T2 | Block registry and parameter metadata | S1.5 | Claude Code (design first) | 3.5 h | [ ] |
| T3 | Generic chunking test, `Delay` block | S1.2 | Claude Code | 3.5 h | [ ] |
| T4 | Devcontainer for GitHub Codespaces | S0.6 | Claude Code + Oli (opens the Codespace) | 1 h | [ ] |
| T5 | Mid-sprint refinement of Sprint 3 | — | `/refine 3` | 0.5 h | [ ] |
| T6 | Release v0.1.0 | — | `/release v0.1.0`, Oli tags | 0.5 h | [ ] |
| T7 | `/sprint-review 2`, `/plan-sprint 3`, vault mirror | — | Oli + Claude Code | 0.5 h | [ ] |

Order: T1 → T2 → T3 → T6. T1 and T2 both edit `BaseBlock` and `docs/conventions.md`, and T3 uses T1's `SUPPORTS_COMPLEX` and T2's registry. T4 any time, lowest priority: it is dropped first if T1–T3 overrun, and T6 does not wait for it. T5 at the end of week 1.

How to run a task in Claude Code: `/task T1`, or paste the prompt block below.

---

## T1 — Block contract (S1.4)

```text
Task T1 (Sprint 2, story S1.4): finish the block contract.

Read CLAUDE.md, docs/ROADMAP.md section 2.2, docs/plan/DECISIONS.md (D-016, D-017)
and python/sigstack/core/block.py first.

Already done in Sprint 1, do not change that behaviour: array layout (..., n_samples),
empty chunks, bool/int input to float64 (D-016), input precision kept (Gain Notes table),
the validation helper _as_signal, the Gain dtype and shape tests.

Add:
1. SUPPORTS_COMPLEX: ClassVar[bool] = True on BaseBlock, checked in BaseBlock.process
   after _as_signal. Complex input to a block with SUPPORTS_COMPLEX = False raises
   TypeError naming the block class, also when the imaginary part is zero.
   Not a member of the Block protocol.
2. BaseBlock.output_shape(input_shape: tuple[int, ...]) -> tuple[int, ...]: the shape
   process() returns for that input shape; default unchanged. Gain uses the default.
3. docs/conventions.md (new, short): array layout and empty chunks; bool/int to float64;
   precision rule (D-017: output has the input precision, a block may compute at higher
   precision internally); the last axis is time, blocks that add, remove or mix axes
   document output_shape; SUPPORTS_COMPLEX. block.py docstrings keep a short summary and
   link to it.

Tests tests/core/test_contract.py, parametrised:
dtypes {bool, int16, float32, float64, complex64, complex128} x shapes {(0,), (1,), (17,),
(1, 17), (3, 17), (2, 3, 17), (3, 0)} x gain real and complex: output shape equals
input shape and output_shape(), dtype follows the documented rule, values exact.
Move test_gain_values_and_shapes and test_gain_dtype_rules from test_block.py here
instead of keeping two copies. A test-local block with SUPPORTS_COMPLEX = False checks
the TypeError.

Out of scope: registry (T2), generic chunking test (T3), precision policy (S1.6, v0.3.0).
Done when: `pixi run check` passes, the rules are in docs/conventions.md and the
docstrings link to it.
Branch: feat/T1-block-contract.
```

---

## T2 — Block registry and parameter metadata (S1.5)

```text
Task T2 (Sprint 2, story S1.5): block registry and typed parameters.

Read docs/conventions.md (from T1) and D-017/D-018 in docs/plan/DECISIONS.md.
Needed later by the generic chunking test (S1.2), pipelines (S5.1), YAML configs (S5.2)
and optimisers (S17.1). Show me the design (signatures and Gain as the example)
before implementing.

Design defaults from D-018 (change only after asking):
- python/sigstack/core/params.py: frozen dataclass Param(name, kind, default=REQUIRED,
  low=None, high=None, low_open=False, high_open=False, choices=None, log=False, doc="").
  Kinds float, int, complex, bool, str; kind complex rejects low/high/log. Bounds are
  validity bounds (search ranges belong to S17.1). validate(value) returns a plain
  Python value and raises TypeError (wrong type, including bool for numeric kinds) or
  ValueError (out of range, not in choices).
- python/sigstack/core/registry.py: a Registry class with one documented default
  instance; @register reads NAME from cls.__dict__ and accepts only concrete BaseBlock
  subclasses; duplicate names raise; get(name) of an unknown name raises ValueError
  listing the known names; all_blocks(). Tests use a fresh Registry.
- each block declares NAME: ClassVar[str] and PARAMS: ClassVar[tuple[Param, ...]] and
  provides params() -> dict (excludes fs and fs_out), classmethod
  from_params(params, *, fs) -> Self (unknown or missing keys raise, naming the valid
  keys) and classmethod example() -> Self (valid instance, used by T3).
- each __init__ calls Param.validate for its own parameters; this replaces _check_gain
  (_check_rate stays for fs). Existing Gain tests stay green.
- Gain registered as "gain" with Param("gain", complex, REQUIRED).
- export Param, register, get, all_blocks from sigstack.core.

Tests tests/core/test_registry.py and test_params.py:
- Gain is listed and get("gain") returns it; duplicate names raise (fresh Registry)
- for every registered block: from_params(b.params(), fs=b.fs_in) gives equal params()
  and fs_in; params() holds plain Python scalars
- PARAMS names and defaults match inspect.signature(cls) minus fs/fs_out
- example() is valid for every registered block
- out-of-range and wrong-type values raise ValueError / TypeError

Out of scope: array parameters (S3.3), search ranges (S17.1), YAML (S5.2).
Done when: `pixi run check` passes, docstrings complete, a "Registry and parameters"
section appended to docs/conventions.md.
Branch: feat/T2-block-registry.
```

---

## T3 — Generic chunking test (S1.2)

```text
Task T3 (Sprint 2, story S1.2): generic chunk-invariance test over all registered blocks.

Depends on T1 and T2 being merged; take the names from the merged T2 before starting.

1. A stateful block so the test has something to catch:
   python/sigstack/core/delay.py, Delay(n, *, fs): y[k] = x[k-n] with zeros before;
   n a non-bool integer >= 0 (TypeError/ValueError like windows._check_length);
   latency = n, fs_out = fs_in; state kept across process() calls and copied, never a
   view of the caller's chunk; reset() restores the zero state; Delay(0) returns a new
   array; leading shape and dtype are fixed by the first chunk after construction or
   reset(), a mismatch raises. Registered as "delay", PARAMS, example() with n = 5.
   Export Delay from sigstack.core and sigstack.
2. tests/core/test_delay.py: values against an np.concatenate reference for n in
   {0, 1, 5} and n > N, complex and (2, 3, N) input, impulse peak at index latency,
   reset, invalid n.
3. tests/core/test_chunking.py: a helper that takes a block instance and checks the
   properties, parametrised over cls.example() for each cls in registry.all_blocks():
   - input: seeded random signal, real; complex unless the class sets
     SUPPORTS_COMPLEX = False; shapes (N,), (3, N), (2, 3, N)
   - hypothesis strategy for split points, including empty chunks and chunks of
     length 1, plus fixed @example cases: empty first chunk, empty last chunk, all
     chunks of length 1, one chunk, signal shorter than the latency
   - property: concatenated chunked output == one-shot output (assert_allclose,
     rtol=1e-12, atol=0; comment why)
   - property: process, reset, process again gives the first result
   - hypothesis settings: derandomize=True, max_examples=50 (deterministic CI)
4. Meta-test: a broken block defined in the test module (forgets its state between
   calls) is passed to the helper directly and must fail it; it is never registered.

Done when: `pixi run check` passes on both OS, the chunking test runs in under 10 s.
Branch: feat/T3-chunking-test.
```

---

## T4 — Devcontainer for GitHub Codespaces (S0.6)

```text
Task T4 (Sprint 2, story S0.6): devcontainer, target GitHub Codespaces (D-019).
Local Podman with Flatpak VS Code is S0.9 (Sprint 4), not this task.

Create .devcontainer/devcontainer.json:
- base image mcr.microsoft.com/devcontainers/base with a pinned Ubuntu LTS tag
  (check the current tag)
- the prefix-dev pixi devcontainer feature, pinned, with pixi v0.84.1; a comment
  "keep in sync with .github/workflows/ci.yml pixi-version" here and in ci.yml
- mounts: a named volume for ${containerWorkspaceFolder}/.pixi, so the host .pixi/
  (with host paths baked in) is never used inside the container
- postCreateCommand: chown the volume to the container user, then pixi install --locked
- customizations.vscode: extensions ms-python.python and charliermarsh.ruff; settings
  python.defaultInterpreterPath = ${containerWorkspaceFolder}/.pixi/envs/default/bin/python
  and ruff from the environment
- do not run pre-commit install inside the container (it would rewrite the host hook)
- README: "Open in GitHub Codespaces" link and the supported hosts

Out of scope: arm64 hosts, a CI job that builds the container, Podman/Flatpak (S0.9),
C++ toolchain.
Done when: a Codespace created from the PR branch builds with no manual step and
`pixi run check` passes in it (Oli opens it and reports back); pixi.toml and pixi.lock
unchanged.
Branch: chore/T4-devcontainer.
```

---

## T5 — Refinement of Sprint 3

Run `/refine 3` at the end of week 1. Sprint 3 holds S0.4 (docs site), S2.1 (test signals), S2.2 (metrics). Points to look at:

- S0.4: the docstrings in `block.py` and `windows.py` use Sphinx roles and `.. math::`, which mkdocstrings does not render as is (convert to `$...$` or choose a plugin). mkdocs-material and mkdocstrings are new dependencies: ask Oli. The Pages deploy workflow needs the gh `workflow` scope (granted) and Pages enabled.
- S2.1: ground-truth format; it is reused by Monte Carlo and the radar simulator later.
- S2.2: which metrics exactly. S2.3 (plots, Sprint 4) needs matplotlib: ask first.

## T6 — Release v0.1.0

`/release v0.1.0` after T3 is merged. S0.7 (release automation) comes in Sprint 4, so the tag and the GitHub release are created by hand this time. T4 is not required for the release.

## T7 — Close the sprint

`/sprint-review 2`, then `/plan-sprint 3`, then mirror to the vault.

---

### Review notes

- Done points:
- Not done:

### Retro

- Keep:
- Change:

### Log

| Date | Task | Time | Note |
|---|---|---|---|
| 07.11.2026 | — | | `/plan-sprint 2` with Oli: goal and stories S1.4, S1.5, S1.2, S0.6 committed (8 points); issues #7, #8, #13, #14 updated from the backlog (#14 to milestone v0.2.0) |
| 07.11.2026 | T1 | | `SUPPORTS_COMPLEX`, `output_shape()`, `docs/conventions.md`, contract test matrix (dtype x shape x gain) |
