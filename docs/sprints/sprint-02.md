# Sprint 2 — 23.11. – 06.12.2026 (draft)

> Draft from 06.11.2026, generated from the schedule in `docs/plan/PLAN.md`. It becomes the committed plan through `/plan-sprint 2` after the Sprint 1 review; stories, estimates and prompts may change then.

**Sprint goal:** every block is discoverable, typed and stream-safe: complex and multichannel input in the contract, a block registry with parameter metadata, and a generic chunking test that runs over all registered blocks. Release v0.1.0.

**Stories:** S1.4, S1.5, S1.2, S0.6 (8 points, capacity 8)

| ID | Task | Story | Who | Est. | Status |
|---|---|---|---|---|---|
| T1 | Complex and multichannel block contract | S1.4 | Claude Code | 2 h | [ ] |
| T2 | Block registry and parameter metadata | S1.5 | Claude Code (design first) | 2.5 h | [ ] |
| T3 | Generic chunking test, `Delay` block | S1.2 | Claude Code | 3.5 h | [ ] |
| T4 | Devcontainer | S0.6 | Claude Code + Oli (test) | 1 h | [ ] |
| T5 | Mid-sprint refinement of Sprint 3 | — | `/refine 3` | 0.5 h | [ ] |
| T6 | Release v0.1.0 | — | `/release v0.1.0`, Oli tags | 0.5 h | [ ] |
| T7 | `/sprint-review 2`, `/plan-sprint 3`, vault mirror | — | Oli + Claude Code | 0.5 h | [ ] |

Order: T1 and T2 (independent) → T3 → T6. T4 any time. T5 at the end of week 1.

---

## T1 — Complex and multichannel block contract (S1.4)

```text
Task T1 (Sprint 2, story S1.4): complex and multichannel input in the block contract.

Read CLAUDE.md, docs/ROADMAP.md section 2.2 and python/sigstack/core/block.py first.

Define and document the dtype and shape rules in the Block docstring and in
docs/conventions.md (new, short):
- shape (..., n_samples); leading axes are channels / batch and are preserved
- n_samples = 0 is allowed and returns an empty array with the same leading shape
- dtype: float32, float64, complex64, complex128 preserved; integer and bool input
  converted to float64; the output dtype of a block with complex parameters follows
  numpy.result_type(input dtype, parameter dtype)
- blocks that do not support complex input declare SUPPORTS_COMPLEX = False and raise
  TypeError with a clear message

Implement the rules in BaseBlock input validation (one helper, reused by all blocks).
Update Gain accordingly.

Tests tests/core/test_contract.py, parametrised over dtype x shape:
dtypes {int16, float32, float64, complex64, complex128}, shapes {(0,), (1,), (17,), (3, 17), (2, 3, 17)},
gain real and complex. Check output shape, dtype and values.

Out of scope: registry (T2), generic chunking test (T3).
Done when: `pixi run check` passes, rules documented in both places.
Branch: feat/S1.4-dtype-shape-contract.
```

---

## T2 — Block registry and parameter metadata (S1.5)

```text
Task T2 (Sprint 2, story S1.5): block registry and typed parameters.

Needed later by the generic chunking test (S1.2), YAML pipelines (S5.2) and optimisers
(S17.1). Show me the design (signatures and one example block) before implementing.

Design to implement after OK:
- python/sigstack/core/params.py: frozen dataclass Param(name, kind, default, low=None,
  high=None, choices=None, log=False, doc="") with validate(value)
- python/sigstack/core/registry.py: register (class decorator), get(name), all_blocks(),
  duplicate names raise
- each block class declares NAME: ClassVar[str] and PARAMS: ClassVar[tuple[Param, ...]],
  and provides params() -> dict, classmethod from_params(d) -> Self,
  classmethod example() -> Self (a valid instance with defaults, used by generic tests)
- construction validates parameter values against PARAMS (range, choices, type)
- Gain registered with a Param for gain (real or complex)

Tests tests/core/test_registry.py:
- Gain is listed, get("gain") returns the class
- params()/from_params() round-trip for every registered block
- out-of-range and wrong-type values raise ValueError / TypeError
- registering two blocks with the same name raises

Done when: `pixi run check` passes, docstrings complete, design note added to
docs/conventions.md.
Branch: feat/S1.5-block-registry.
```

---

## T3 — Generic chunking test (S1.2)

```text
Task T3 (Sprint 2, story S1.2): generic chunk-invariance test over all registered blocks.

Depends on T1 and T2 being merged.

1. Add a stateful block so the test has something to catch:
   python/sigstack/core/delay.py, Delay(n: int >= 0) shifts the signal by n samples
   (zeros at the start), latency = n, state kept across process() calls, reset() clears it.
   Registered, with PARAMS and example().
2. tests/core/test_chunking.py, parametrised over registry.all_blocks():
   - instance from Block.example()
   - input: seeded random signal, real; complex if SUPPORTS_COMPLEX; shapes (N,) and (3, N)
   - hypothesis strategy for split points, including empty chunks and chunks of length 1
   - property: concatenation of chunked outputs == one-shot output (assert_allclose,
     rtol=1e-12, atol=0; comment why)
   - property: process, reset, process again gives the same result as the first run
   - hypothesis settings: derandomize=True, max_examples=50 (deterministic CI)
3. Meta-test: a deliberately broken block defined only inside the test module (forgets its
   state between calls) must make the property fail. This shows the test can catch bugs.

Done when: `pixi run check` passes on both OS, runtime of the chunking test < 10 s.
Branch: feat/S1.2-chunking-test.
```

---

## T4 — Devcontainer (S0.6)

```text
Task T4 (Sprint 2, story S0.6): devcontainer.

Create .devcontainer/devcontainer.json:
- base image mcr.microsoft.com/devcontainers/base:ubuntu (current LTS tag)
- install pixi via the official devcontainer feature from prefix-dev (check its current
  name and version before writing)
- postCreateCommand: pixi install
- VS Code extensions: ms-python.python, charliermarsh.ruff
- README: "Open in GitHub Codespaces" section with one sentence

Done when: the container builds (devcontainer CLI or Codespaces) and `pixi run check`
passes inside. Oli tests it locally with Podman or in Codespaces and reports back.
Branch: chore/S0.6-devcontainer.
```

---

## T5 — Refinement of Sprint 3

Run `/refine 3` at the end of week 1. Sprint 3 is scheduled to hold S0.4 (docs site), S2.1 (test signals), S2.2 (metrics). Points to look at: which metrics exactly, ground-truth format of S2.1 (it is reused by Monte Carlo and the radar simulator later).

## T6 — Release v0.1.0

`/release v0.1.0` after T3 is merged. S0.7 (release automation) comes in Sprint 4, so the tag and the GitHub release are created by hand this time.

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
| | | | |
