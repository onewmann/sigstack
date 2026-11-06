
# sigstack roadmap

Architecture, way of working and release strategy. Status: draft, 06.11.2026.

The backlog and the sprint schedule are not maintained here. They live in `docs/plan/` as data and are generated: see section 4.

---

## 1. Product vision

**For** engineers and students who process sampled signals (audio, vibration, biosignals, radar, RF/communications, sensor data),
**sigstack** is a C++/Python toolkit with a uniform block-processing model,
**that** covers the standard DSP chain from filter design to feature extraction, works offline and in streaming mode, and makes every result reproducible.
**Unlike** a loose collection of SciPy calls, it offers composable pipelines, a fast C++ core with identical Python reference implementations, built-in benchmarking, and a simulation-based design loop: Monte Carlo verification and automated (Bayesian / surrogate-model based) optimisation of DSP parameters.

### Goals

| Goal | Measure |
|---|---|
| Reusable across domains | Same pipeline model runs on audio, EEG, vibration, radar and RF examples with config changes only |
| Correct | Every C++ kernel tested against a NumPy reference, tolerance documented |
| Fast | C++ path at least on par with SciPy for the core kernels |
| Reproducible | `pixi run reproduce` regenerates all figures from scratch |
| Verified by simulation | Performance claims (detection rate, SNR gain, EVM, error) come from Monte Carlo runs with stated confidence intervals |
| Optimised automatically | Pipeline parameters can be tuned by an optimiser against a simulated scenario, with cost/accuracy trade-off as Pareto front |
| Presentable | README, docs site, CI badges, tagged releases |

### Non-goals (for v1.0)

- GUI, real-time audio I/O drivers, SDR hardware drivers (recorded IQ files and simulated data only; an SDR adapter can come after v1.0)
- Learned models for the signal task, e.g. an NN detector (only feature extraction as input for them; the comparison with CFAR, S18.3, comes after v1.0). The GP surrogates and differentiable DSP in E18 are optimisation tools and stay in scope
- GPU kernels

### Ground rules

- Written from scratch, based on the literature (Oppenheim/Schafer, Vaidyanathan, Mallat, Haykin) and public docs
- No code, data, parameters or results from the thesis or any employer

---

## 2. Architecture

### 2.1 Layers

```
┌──────────────────────────────────────────────────────────┐
│ apps / examples  audio · EEG · vibration · radar · RF     │
├──────────────────────────────────────────────────────────┤
│ design loop          sim (Monte Carlo, scenarios)         │
│                      optim (Bayesian, multi-objective,    │
│                      surrogates, differentiable DSP)      │
├──────────────────────────────────────────────────────────┤
│ pipeline             Block, Pipeline, YAML config, CLI    │
├──────────────────────────────────────────────────────────┤
│ modules (Python API, C++ fast path where it matters)      │
│  filters  spectral  multirate  wavelets  adaptive         │
│  features detection estimation  radar  rf  array          │
├──────────────────────────────────────────────────────────┤
│ core                 buffers, state, dtypes, windows      │
├──────────────────────────────────────────────────────────┤
│ support              signals (generators), io, metrics,   │
│                      bench, plots                         │
└──────────────────────────────────────────────────────────┘
```

### 2.2 Block model

Every processing step implements the same interface, so it works on a whole array or on a stream of chunks:

```python
class Block(Protocol):
    fs_in: float
    fs_out: float              # changes for decimation / resampling
    def reset(self) -> None: ...
    def process(self, x: np.ndarray) -> np.ndarray: ...   # keeps internal state between calls
    @property
    def latency(self) -> int: ...                          # samples
```

Requirement for every block: chunked processing gives the same result as one-shot processing.

$$
\text{process}(x_{1}) \,\|\, \text{process}(x_{2}) \;=\; \text{process}(x_{1} \,\|\, x_{2}), \qquad \| = \text{concatenation}
$$

This rule is tested automatically for every block (property-based test with `hypothesis`, random chunk sizes).

Example pipeline:

```python
pipe = Pipeline([
    Bandpass(1, 40, order=4, fs=250),
    STFT(nfft=256, hop=64, window="hann"),
    BandEnergy(bands={"alpha": (8, 13), "beta": (13, 30)}),
])
features = pipe.process(eeg)
```

The same pipeline as YAML, runnable with `sigstack run config.yaml`.

### 2.3 Modules

| Module | Content v1.0 | Later |
|---|---|---|
| `core` | buffers, dtype handling (float32/64, complex), window functions | fixed-point types |
| `filters` | FIR design (window, Parks-McClellan, LS), IIR (Butterworth, Chebyshev, elliptic) as SOS, streaming FIR/IIR, zero-phase offline | lattice, all-pass |
| `spectral` | FFT wrapper, STFT/ISTFT, Welch PSD, band energy, cross-spectrum, coherence | multitaper |
| `multirate` | decimation, interpolation, polyphase resampler, uniform DFT filter bank, WOLA | cosine-modulated banks, non-uniform trees |
| `wavelets` | DWT, wavelet packets (own code + PyWavelets comparison) | CWT, scalograms |
| `adaptive` | LMS, NLMS, RLS | Kalman, frequency-domain LMS |
| `features` | RMS, envelope (Hilbert), spectral centroid/flatness, zero crossings, band power | MFCC |
| `detection` | threshold with hysteresis, peak picking, CA-CFAR, OS-CFAR, 2D CFAR | onset detection |
| `rf` | complex baseband (IQ) handling, NCO/mixer, digital down-conversion (DDC), IQ imbalance and DC offset correction, carrier/frequency offset estimation, AM/FM/PSK/QAM modulation and demodulation, matched filter, raised-cosine pulse shaping, constellation and EVM | symbol timing recovery, OFDM, channel models |
| `radar` | waveforms (pulsed LFM, FMCW, phase codes: Barker, Frank), pulse compression, range FFT, range-Doppler map, MTI / clutter filter, ambiguity function, range/velocity resolution calculator | SAR basics, micro-Doppler, tracking (α-β, Kalman) |
| `array` | uniform linear array model, steering vectors, delay-and-sum beamforming | MVDR, MUSIC/ESPRIT direction of arrival |
| `signals` | multitone, chirp, noise (white/pink), impulses, AM/FM, seeded; radar scene simulator (point targets with range, velocity, RCS, noise); RF test signals (modulated symbols, carrier offset, multipath) | realistic biosignal models |
| `io` | WAV, CSV, NumPy, HDF5, raw IQ, SigMF | EDF (biosignals) |
| `metrics` | SNR, frequency response, ripple, attenuation, group delay, cost model | |
| `bench` | timing harness, results as JSON, comparison plots | |
| `sim` | scenario definition (signal + impairments + parameter distributions), Monte Carlo runner (seeded, parallel, resumable), estimators with confidence intervals, ROC / $P_d$–SNR curves, Latin hypercube and Sobol sampling, sensitivity analysis | importance sampling for rare events ($P_{fa} < 10^{-6}$) |
| `embedded` (C++ only) | | allocation-free API, Q15/Q31 fixed point, Cortex-M port with cycle counts, hardware-in-the-loop test (E19) |
| `optim` | parameter spaces from block configs, Bayesian optimisation (Optuna, TPE / GP), multi-objective search (NSGA-II) with Pareto front, robust optimisation (objective = MC statistic, e.g. worst-case or 5th percentile) | surrogate models (Gaussian process, small NN) replacing expensive simulations, differentiable DSP in JAX/PyTorch for gradient-based filter design |

### 2.4 Tech stack

| Topic | Choice |
|---|---|
| Environment | pixi (conda-forge), lockfile, Linux + Windows |
| Container | Devcontainer on top of pixi |
| Build | CMake + scikit-build-core (hatchling as a stand-in until S0.3) |
| C++ | C++20, header-mostly core library, usable without Python |
| Bindings | nanobind |
| FFT | pocketfft default, FFTW optional (`-DSIGSTACK_USE_FFTW=ON`) |
| Tests | pytest, hypothesis, GoogleTest |
| Quality | pre-commit: ruff, mypy, clang-format, clang-tidy |
| Docs | MkDocs Material, MathJax, mkdocstrings |
| CI | GitHub Actions: build, tests, coverage, docs deploy to Pages |
| License | Apache-2.0 (or MIT) |
| Monte Carlo execution | joblib (local, all cores), optional Dask; NumPy `SeedSequence` for independent streams |
| Sampling / sensitivity | SciPy `qmc` (Latin hypercube, Sobol), SALib |
| Optimisation | Optuna (TPE, GP, NSGA-II), optional BoTorch |
| Surrogates / learned DSP | scikit-learn GP, PyTorch or JAX (optional extra `sigstack[ml]`) |
| Experiment tracking | Hydra-style sweeps from YAML, results as Parquet, optional MLflow |
| Planning | `docs/plan/backlog.toml` + `plan.toml`, `scripts/plan.py` (standard library only) |
| Embedded (E19) | arm-none-eabi-gcc, CMake toolchain file, CMSIS-DSP as comparison baseline |

### 2.5 Simulation-based design loop

```
 scenario.yaml ──► sim: draw N realisations ──► pipeline ──► metrics
   (signal,          (seeds, impairments,                       │
    impairments,      parameter distributions)                  ▼
    distributions)                                   MC statistics + CI
        ▲                                                       │
        │                                                       ▼
   optim proposes new parameters ◄── objective(s): accuracy, cost, latency
        │
        └──► Pareto front, best config, report
```

Every performance number in the docs comes out of this loop, never from a single run.

**Monte Carlo accuracy.** For an estimated probability $\hat p = k/N$ (e.g. false-alarm rate), the standard error is

$$
\sigma_{\hat p} = \sqrt{\frac{p(1-p)}{N}}, \qquad
\text{relative error} \approx \frac{1}{\sqrt{N p}} .
$$

Rule used in the runner: at least $N \approx 100/p$ trials for ±10 % relative error. For $P_{fa} = 10^{-4}$ that means $10^{6}$ trials, which is why the C++ fast path and parallel execution matter. Intervals are reported as Wilson intervals.

**Optimisation example (radar):** tune CFAR guard/training cells, threshold factor and window type to maximise $P_d$ at fixed $P_{fa}$ over a distribution of target SNRs and clutter levels, while minimising operations per frame. Result: Pareto front of detection performance vs. compute cost.

**Where "AI" fits, without overselling:**

| Technique | Role | Status |
|---|---|---|
| Bayesian optimisation | sample-efficient search over expensive MC objectives | core (v0.6) |
| Surrogate model (GP / small NN) | learns objective from past runs, predicts promising regions, cuts simulation time | v0.9 |
| Differentiable DSP | filter coefficients or window shapes trained by gradient descent through the pipeline | v0.9, optional |
| Learned components | e.g. NN detector compared against CFAR on the same MC scenarios | after v1.0 |

The README states plainly what each part does. Learned methods are always benchmarked against the classical baseline in the same Monte Carlo setup.

### 2.6 Repository layout

```
sigstack/
├── cpp/{include/sigstack, src, tests, bench}
├── python/sigstack/{core, filters, spectral, multirate, wavelets,
│                    adaptive, features, detection, signals, io,
│                    rf, radar, array, metrics, bench, sim, optim,
│                    pipeline, plots}
├── examples/{audio, eeg, vibration, radar, rf}
├── experiments/{configs, run.py}
├── docs/{plan, sprints}   ROADMAP, PLANNING, DECISIONS, generated PLAN/BACKLOG
├── scripts/              plan.py, sync_issues.py
├── tests/
├── pixi.toml  pyproject.toml  CMakeLists.txt
├── .devcontainer/  .github/
└── README.md  CHANGELOG.md  CONTRIBUTING.md
```

### 2.7 Reference relations

Used as test oracles and in the docs.

FMCW radar (bandwidth $B$, chirp duration $T_c$, $N$ chirps per frame, wavelength $\lambda$, beat frequency $f_b$):

$$
\Delta R = \frac{c}{2B}, \qquad
R = \frac{c\, f_b\, T_c}{2B}, \qquad
\Delta v = \frac{\lambda}{2 N T_c}, \qquad
v_{\max} = \frac{\lambda}{4 T_c}
$$

Uniform linear array (element spacing $d$, element index $n$, angle $\theta$):

$$
a(\theta)_n = e^{-j 2\pi n d \sin\theta / \lambda}
$$

CA-CFAR with $N$ training cells on square-law detected Gaussian noise (threshold factor $\alpha$ for a given $P_{fa}$):

$$
\alpha = N\left(P_{fa}^{-1/N} - 1\right)
$$

---

## 3. Way of working (solo Scrum)

Scrum reduced to what makes sense for one developer next to the master's programme.

### 3.1 Cadence

| Element | Setting |
|---|---|
| Sprint length | 2 weeks, Monday to Sunday |
| Capacity | ~6 h per week, ~12 h per sprint |
| Sprint planning | 30 min on Monday: pick stories up to capacity, write sprint goal |
| Daily | none; a one-line log entry per work session instead |
| Sprint review | 20 min on Sunday: run the demo notebook, write release notes draft |
| Retrospective | 10 min: one thing to keep, one thing to change |
| Backlog refinement | 20 min mid-sprint: split and estimate the next stories |

### 3.2 Estimation

Story points on a Fibonacci scale (1, 2, 3, 5, 8). Calibration: 1 point ≈ 1 to 1.5 h focused work.
Start value for planning: **8 points per sprint**. After three sprints use the mean of the measured velocity.
Stories with 8 points are split before they enter a sprint.

### 3.3 Tooling on GitHub

- **Issues** for every story and bug, with templates (user story, bug, spike)
- **Projects board**: Backlog → Ready → In progress → Review → Done
- **Fields**: points, epic, sprint (iteration field), priority
- **Milestones** = releases of the active profile (created by `scripts/sync_issues.py`)
- **Labels**: `epic:*`, `type:feature|bug|docs|infra|spike`, `good first issue`
- **Branches**: `main` protected, short-lived feature branches, squash merge via pull request, CI must pass
- **Commits**: Conventional Commits (`feat:`, `fix:`, `docs:`), changelog generated from them

### 3.4 Definition of Ready

- User story with clear value
- Acceptance criteria written
- Estimated, at most 5 points
- Dependencies done or in the same sprint

### 3.5 Definition of Done

- Code merged to `main` via pull request, CI green
- Unit tests; for blocks also the chunking test
- C++ kernels: tested against Python reference, tolerance stated
- Public API has docstrings and appears in the API docs (from S0.4 on; before that, docstrings only)
- Example or test with a synthetic signal
- Changelog entry

---

## 4. Backlog, releases and schedule

### 4.1 Where things live

| File | Content | Edited by |
|---|---|---|
| `docs/plan/backlog.toml` | epics and all stories (story, acceptance criteria, points, dependencies, status) | Oli / Claude Code during refinement and review |
| `docs/plan/plan.toml` | calendar with capacity factors, velocity, actual points per sprint, release profiles | sprint review |
| `docs/plan/PLAN.md` | release forecast, sprint schedule, profile comparison, warnings | generated by `scripts/plan.py` |
| `docs/plan/BACKLOG.md` | all stories per epic with planned sprint | generated |
| `docs/sprints/sprint-NN.md` | task breakdown with Claude Code prompts, review and retro | `/plan-sprint`, `/sprint-review` |
| `docs/plan/DECISIONS.md` | decision log | whenever the plan changes direction |

`python scripts/plan.py --check` validates the backlog (unique ids, known epics, allowed points, dependencies exist, no cycles, no dependency on a later release). It runs in CI from Sprint 1 on, so the plan cannot silently break.

### 4.2 Epics

| ID | Epic |
|---|---|
| E0 | Project infrastructure |
| E1 | Core and block model |
| E2 | Signal generators and metrics |
| E3 | Filters |
| E4 | Spectral analysis |
| E5 | Pipeline, config, CLI |
| E6 | C++ core and bindings |
| E7 | Multirate and filter banks |
| E8 | Wavelets |
| E9 | Adaptive filters |
| E10 | Features and detection |
| E11 | Benchmarking and cost model |
| E12 | Domain examples, docs, v1.0 |
| E13 | RF / communications |
| E14 | Radar |
| E15 | Antenna arrays and beamforming |
| E16 | Simulation and Monte Carlo |
| E17 | Optimisation |
| E18 | Surrogates and learned components |
| E19 | Embedded port (Cortex-M) |

Stories are grouped into work packages (`group`, e.g. `filters`, `mc`, `radar-core`). A release profile in `plan.toml` maps groups to releases. Changing the order of the roadmap therefore means editing one list, not 90 stories.

### 4.3 Release profiles

Two orders are kept side by side until the decision in section 6 is made:

- **default**: breadth first. General modules, Monte Carlo and optimisation, then radar, RF, arrays. Embedded port after v1.0.
- **showcase**: FMCW radar chain directly after the pipeline, then Monte Carlo, optimisation and the embedded port. Breadth afterwards.

Snapshot from 06.11.2026 with velocity 8 (the current numbers are always in `PLAN.md`):

| Milestone | default | showcase |
|---|---|---|
| v0.2.0, first version worth linking | 31.01.2027 | 31.01.2027 |
| Simulate-verify-optimise loop | 10.10.2027 (generic) | 26.09.2027 (radar) |
| Radar showcase | 05.12.2027 | 26.09.2027 |
| Radar chain on Cortex-M | 18.06.2028 | 05.12.2027 |
| v1.0.0 | 23.04.2028 | 04.06.2028 |

Rules for releases: at most about three sprints of points per release, every release ends with something that runs and a tag, release notes from the changelog.

### 4.4 Planning horizon (rolling wave)

| Horizon | Detail |
|---|---|
| Current sprint | tasks with prompts in `docs/sprints/sprint-NN.md` |
| Next sprint | stories meet the Definition of Ready (refinement mid-sprint) |
| Next two releases | stories with acceptance criteria and estimates |
| Later | stories may be coarse; re-estimated when they come within two releases |

The update process is described in `docs/plan/PLANNING.md`.

---

## 5. Risks

| Risk | Effect | Countermeasure |
|---|---|---|
| Study workload higher than planned | Sprints miss their goal | Capacity 6 h/week is conservative, pause rules for exam periods |
| C++/Python build issues on Windows | CI blocks | Spike S6.1 early, Linux-only fallback for the C++ part |
| Scope too broad | Nothing finished | Releases of at most three sprints, rolling-wave planning, showcase profile as fallback |
| Forecast drifts (v1.0 about spring 2028) | Late CV milestones | Forecast recalculated every review; scope cuts decided in `/replan`, logged in DECISIONS.md |
| Exam windows differ from the calendar assumption | Wrong capacity | Check against the ESE exam plan, edit `plan.toml` periods |
| Hardware for E19 (Cortex-M board) | Embedded epic blocked | Spike S19.1 chooses and orders the board two sprints ahead |
| Duplicating SciPy without benefit | Weak portfolio value | Focus on block model, streaming, C++ path and benchmarks |
| Package name taken on PyPI | Rename later | Check name in T0 |

---

## 6. Open decisions

Details and status in `docs/plan/DECISIONS.md`.

- [ ] Name `sigstack` free on GitHub and PyPI (T0)
- [ ] License: Apache-2.0 or MIT
- [ ] Release order: profile `default` or `showcase` (see `PLAN.md`, profile comparison). Decide before Sprint 8, when the orders start to differ.
- [ ] Embedded epic E19: keep, and if yes, which board
- [ ] Weekly time budget (6 h assumed, velocity 8)
- [ ] Exam windows in `plan.toml` against the ESE exam plan
- [ ] Dependencies when due: YAML library (S5.2), Optuna (S17.2), ML backend JAX or PyTorch (S18.2)
