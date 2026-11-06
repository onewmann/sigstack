# sigstack: Reusable Signal Processing Toolkit

Architecture and agile development plan
Status: draft, 2026-11-06

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
├── docs/
├── tests/
├── pixi.toml  pyproject.toml  CMakeLists.txt
├── .devcontainer/  .github/
└── README.md  CHANGELOG.md  CONTRIBUTING.md
```

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
- **Milestones** = releases (v0.1.0 … v1.0.0)
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
- Public API has docstrings, appears in the docs
- Example or test with a synthetic signal
- Changelog entry

---

## 4. Product backlog

### 4.1 Epics

| ID | Epic | Release |
|---|---|---|
| E0 | Project infrastructure | v0.1 |
| E1 | Core and block model | v0.1 |
| E2 | Signal generators and metrics | v0.1 |
| E3 | Filters | v0.2 |
| E4 | Spectral analysis | v0.2 |
| E5 | Pipeline, config, CLI | v0.3 |
| E6 | C++ core and bindings | v0.3 |
| E7 | Multirate and filter banks | v0.4 |
| E8 | Wavelets | v0.5 |
| E9 | Adaptive filters | v0.5 |
| E10 | Features and detection | v0.6 |
| E11 | Benchmarking | v0.4 |
| E12 | Domain examples and docs | ongoing, finished in v1.0 |
| E13 | RF / communications | v0.7 |
| E14 | Radar | v0.7 |
| E15 | Antenna arrays and beamforming | v0.8 |
| E16 | Simulation and Monte Carlo | v0.4 |
| E17 | Optimisation | v0.6 |
| E18 | Surrogates and differentiable DSP | v0.9 |

### 4.2 User stories (priority order)

**E0 Infrastructure**

| ID | Story | AC | Pts |
|---|---|---|---|
| S0.1 | As a developer I want a pixi environment so that the setup runs with one command on Fedora and Windows | `pixi install && pixi run check` works on both | 2 |
| S0.2 | As a developer I want CI on every push so that broken code never reaches `main` | Workflow runs lint + tests, required for merge | 2 |
| S0.3 | As a user I want `pip install .` to build the package so that I can use it in other projects | scikit-build-core build, wheel installs in a clean env | 3 |
| S0.4 | As a reader I want a docs site so that I can learn the API | MkDocs on GitHub Pages, auto-deploy | 2 |
| S0.5 | As a developer I want pre-commit hooks so that style is consistent | ruff, mypy, clang-format active | 1 |

**E1 Core and block model**

| ID | Story | AC | Pts |
|---|---|---|---|
| S1.1 | As a user I want a common `Block` interface so that all steps are interchangeable | Protocol defined, documented, used by one dummy block | 2 |
| S1.2 | As a developer I want an automatic chunking test so that every stateful block is stream-safe | Hypothesis test applied to all registered blocks | 3 |
| S1.3 | As a user I want standard windows (Hann, Hamming, Blackman, Blackman-Harris, Kaiser, flat-top) | Match `scipy.signal.get_window` to 1e-12 | 1 |
| S1.4 | As a developer I want complex dtypes and multichannel arrays (channels × samples) in the block contract from the start, so that RF, radar and array processing need no refactor later | Dummy block passes tests for real, complex, 1-channel and N-channel input | 2 |

**E2 Signals and metrics**

| ID | Story | AC | Pts |
|---|---|---|---|
| S2.1 | As a tester I want seeded test signals (multitone, chirp, noise, impulse, AM/FM) with known ground truth | Generators return signal + metadata (frequencies, energies) | 3 |
| S2.2 | As a user I want SNR, frequency response, ripple, attenuation and group delay metrics | Validated on textbook filters | 3 |
| S2.3 | As a user I want standard plots (magnitude/phase, spectrogram, band energies) | One function per plot, consistent style | 2 |

**E3 Filters**

| ID | Story | AC | Pts |
|---|---|---|---|
| S3.1 | As a user I want FIR design by window method, Parks-McClellan and least squares | Results match SciPy, spec checked with metrics | 3 |
| S3.2 | As a user I want IIR design (Butterworth, Chebyshev I/II, elliptic) as SOS | Stable, matches SciPy | 2 |
| S3.3 | As a user I want streaming FIR and IIR blocks | Pass chunking test | 3 |
| S3.4 | As a user I want zero-phase offline filtering | Same result as `filtfilt` within tolerance | 1 |

**E4 Spectral**

| ID | Story | AC | Pts |
|---|---|---|---|
| S4.1 | As a user I want STFT/ISTFT with perfect reconstruction check | COLA check, reconstruction error < 1e-10 | 3 |
| S4.2 | As a user I want Welch PSD and band energy for arbitrary band edges | Parseval check on multitone ground truth | 3 |
| S4.3 | As a user I want cross-spectrum and coherence | Matches SciPy | 2 |

**E5 Pipeline**

| ID | Story | AC | Pts |
|---|---|---|---|
| S5.1 | As a user I want to chain blocks in a `Pipeline` | Sample rates and latency propagate correctly | 3 |
| S5.2 | As a user I want to define pipelines in YAML | Round trip config ↔ object | 3 |
| S5.3 | As a user I want a CLI `sigstack run config.yaml` | Reads input file, writes results + figures | 2 |

**E6 C++ core**

| ID | Story | AC | Pts |
|---|---|---|---|
| S6.1 | Spike: nanobind setup with one kernel (FIR) | Builds in CI on Linux + Windows | 3 |
| S6.2 | C++ streaming FIR and SOS IIR | Equal to Python reference, faster than pure NumPy | 5 |
| S6.3 | C++ STFT on pocketfft, optional FFTW | Equal results with both backends | 5 |
| S6.4 | C++ library usable without Python | CMake target + minimal C++ example | 2 |

**E7 Multirate**

| ID | Story | AC | Pts |
|---|---|---|---|
| S7.1 | Decimator and interpolator blocks with anti-alias design | Alias attenuation meets spec | 3 |
| S7.2 | Polyphase rational resampler | Matches `resample_poly` | 3 |
| S7.3 | Uniform DFT polyphase analysis bank, $M$ channels, $P$ taps per branch | Equals windowed FFT for $P=1$ | 5 |
| S7.4 | WOLA analysis/synthesis | Reconstruction error documented | 3 |

**E13 RF / communications**

| ID | Story | AC | Pts |
|---|---|---|---|
| S13.1 | As a user I want complex IQ signals handled consistently by all blocks | Complex dtype passes through filters, spectral, multirate; tests with complex input | 3 |
| S13.2 | As a user I want an NCO/mixer and a DDC block (mix, low-pass, decimate) | Tone at $f_c$ lands at 0 Hz, image rejection meets spec | 3 |
| S13.3 | As a user I want to read raw IQ and SigMF recordings | Metadata (fs, centre frequency) mapped to block parameters | 2 |
| S13.4 | As a user I want AM/FM demodulation | Recovers a known message tone, SNR measured | 3 |
| S13.5 | As a user I want PSK/QAM modulation with raised-cosine pulse shaping and a matched filter | Zero ISI at symbol instants, EVM function, constellation plot | 5 |
| S13.6 | As a user I want IQ imbalance, DC offset and frequency offset estimation and correction | Image suppression and residual offset documented on simulated impairments | 3 |

**E14 Radar**

| ID | Story | AC | Pts |
|---|---|---|---|
| S14.1 | As a user I want radar waveforms (pulsed LFM, FMCW chirp, Barker and Frank codes) | Bandwidth and duration match parameters | 2 |
| S14.2 | As a tester I want a point-target scene simulator (range, radial velocity, amplitude, noise) | Ground truth returned with the data cube | 3 |
| S14.3 | As a user I want pulse compression via matched filter | Peak at true range, sidelobe level matches theory for the waveform | 3 |
| S14.4 | As a user I want a range-Doppler map from an FMCW data cube (fast-time × slow-time FFT) | Simulated targets appear at correct range and velocity bins | 5 |
| S14.5 | As a user I want 2D CA-CFAR on the range-Doppler map | Detection and false-alarm rate checked against set $P_{fa}$ on noise-only data | 3 |
| S14.6 | As a user I want MTI / clutter cancellation | Stationary clutter suppressed by stated amount, moving target kept | 2 |
| S14.7 | As a user I want an ambiguity function and a resolution calculator | Results match the formulas below | 2 |

Reference relations used in tests and docs (FMCW, bandwidth $B$, chirp duration $T_c$, $N$ chirps per frame, wavelength $\lambda$):

$$
\Delta R = \frac{c}{2B}, \qquad
R = \frac{c\, f_b\, T_c}{2B}, \qquad
\Delta v = \frac{\lambda}{2 N T_c}, \qquad
v_{\max} = \frac{\lambda}{4 T_c}
$$

with beat frequency $f_b$.

**E15 Arrays**

| ID | Story | AC | Pts |
|---|---|---|---|
| S15.1 | As a user I want a uniform linear array model with steering vectors | Array response matches $a(\theta)_n = e^{-j 2\pi n d \sin\theta / \lambda}$ | 2 |
| S15.2 | As a user I want delay-and-sum beamforming | Beam pattern main lobe and grating lobes as predicted | 3 |
| S15.3 | As a user I want angle estimation (angle FFT, later MUSIC) for a MIMO radar cube | Simulated target angles recovered within resolution | 5 |

**E16 Simulation and Monte Carlo**

| ID | Story | AC | Pts |
|---|---|---|---|
| S16.1 | As a user I want to define a scenario in YAML (signal model, impairments, parameter distributions) | Scenario object validated, reproducible from seed | 3 |
| S16.2 | As a user I want a Monte Carlo runner that executes a pipeline over N realisations in parallel | Same seed gives identical results with 1 and 8 workers; runs can be resumed | 5 |
| S16.3 | As a user I want estimators with confidence intervals (mean, quantiles, Wilson interval for probabilities) | Coverage of the 95 % interval checked on a known distribution | 2 |
| S16.4 | As a user I want ROC and $P_d$–SNR curves from MC results | CA-CFAR on Gaussian noise matches the analytic $P_{fa}$ within the CI | 3 |
| S16.5 | As a user I want Latin hypercube / Sobol sweeps and Sobol sensitivity indices | Indices match a test function with known values (Ishigami) | 3 |

**E17 Optimisation**

| ID | Story | AC | Pts |
|---|---|---|---|
| S17.1 | As a user I want parameter spaces derived from block configs (ranges, choices, log scale) | Space round-trips to YAML, invalid configs rejected | 2 |
| S17.2 | As a user I want Bayesian optimisation of a pipeline against an MC objective | Finds the known optimum of a filter design toy problem within budget | 3 |
| S17.3 | As a user I want multi-objective optimisation with a Pareto front (e.g. accuracy vs. cost) | Pareto plot and table of non-dominated configs exported | 3 |
| S17.4 | As a user I want robust optimisation (objective = worst case or percentile over scenarios) | Robust optimum degrades less under impairment shift than nominal optimum, shown in a test | 3 |
| S17.5 | As a user I want an optimisation report (best config, history, sensitivity) | Generated as Markdown + figures from one command | 2 |

**E18 Surrogates and differentiable DSP**

| ID | Story | AC | Pts |
|---|---|---|---|
| S18.1 | Spike: GP surrogate trained on MC history, used to pre-screen candidates | Same optimum with fewer full simulations, saving documented | 5 |
| S18.2 | Differentiable FIR / window design in JAX or PyTorch | Gradient-designed filter meets spec, compared with Parks-McClellan | 5 |
| S18.3 | After v1.0 (see non-goals): small NN detector vs. CFAR on identical MC scenarios | ROC comparison with CIs, honest result either way | 5 |

**E8 to E12** are refined when they come closer (backlog refinement), currently listed as epics only.

---

## 5. Release roadmap

| Release | Content | Sprints |
|---|---|---|
| v0.1.0 | Infra, block model, signals, metrics | 1 to 2 |
| v0.2.0 | Filters, spectral analysis | 3 to 5 |
| v0.3.0 | Pipeline + CLI, first C++ kernels | 6 to 7 |
| v0.4.0 | Multirate, filter banks, benchmark harness, Monte Carlo runner (E16) | 8 to 10 |
| v0.5.0 | Wavelets, adaptive filters | 11 to 12 |
| v0.6.0 | Features, detection, optimisation (E17) | 13 to 14 |
| v0.7.0 | RF/IQ basics, radar waveforms, pulse compression, range-Doppler, CFAR | 15 to 17 |
| v0.8.0 | Modulation/demodulation, impairment correction, arrays and beamforming | 18 to 19 |
| v0.9.0 | Surrogates and differentiable DSP (E18 without S18.3) | 20 to 21 |
| v1.0.0 | API freeze, domain examples (audio, EEG, vibration, FMCW radar, digital comms), full docs | 22 |

If radar/RF becomes the main focus (e.g. for job applications), E13 and E14 can move ahead of E8 to E10, directly after v0.4.0. They only depend on filters, spectral, multirate and complex support.

**Earliest point to put it on the CV:** v0.2.0 (usable, documented, CI green).

---

## 6. Sprint calendar

Dates assume a start on Monday, 9 Nov 2026. Lecture-free and exam periods get reduced capacity or a pause.

| Sprint | Dates | Goal | Stories | Pts |
|---|---|---|---|---|
| S1 | 09.11. – 22.11. | Environment and CI run | S0.1, S0.2, S0.5, S1.1, S1.3 | 8 |
| S2 | 23.11. – 06.12. | v0.1.0: tested block model with test signals | S0.3, S1.2, S1.4, S2.1 | 11 |
| S3 | 07.12. – 20.12. | Metrics, plots, first filters | S2.2, S2.3, S3.1 | 8 |
| — | 21.12. – 03.01. | Break (optional: docs S0.4) | | |
| S4 | 04.01. – 17.01. | Filters complete, STFT | S3.2, S3.3, S3.4, S4.1 | 9 |
| S5 | 18.01. – 31.01. | v0.2.0: band energy, coherence, docs site | S4.2, S4.3, S0.4 | 7 |
| — | Feb – mid Mar | Exam period: pause or bugs only | | |
| S6 … | from mid Mar | Pipeline + C++ (v0.3.0), then per roadmap | E5, E6 | |

The plan for S6 onward is set after the first velocity measurement.

---

## 7. Sprint 1 in detail

**Sprint goal:** `pixi install && pixi run check` works on Fedora and Windows, CI runs on every PR, the `Block` interface and window functions exist. Tasks with ready prompts: `docs/sprints/sprint-01.md`.

| Task | Story | Est. |
|---|---|---|
| Create repo `onewmann/sigstack`, license, README stub, noreply author | S0.1 | 0.5 h |
| `pixi.toml` with Python, NumPy, SciPy, pytest, hypothesis, ruff, mypy, pre-commit (compilers and CMake come with E6) | S0.1 | 1.5 h |
| Test on Fedora desktop and Windows laptop | S0.1 | 1 h |
| GitHub Actions matrix (ubuntu, windows): lint + tests | S0.2 | 2 h |
| Branch protection, PR template, issue templates | S0.2 | 0.5 h |
| pre-commit config | S0.5 | 1 h |
| `Block` protocol + dummy gain block + test | S1.1 | 2 h |
| Window functions + comparison tests against SciPy | S1.3 | 1.5 h |
| GitHub Projects board with fields, enter backlog | infra | 1 h |
| Review + retro | — | 0.5 h |

Total ≈ 11.5 h.

---

## 8. Risks

| Risk | Effect | Countermeasure |
|---|---|---|
| Study workload higher than planned | Sprints miss their goal | Capacity 6 h/week is conservative, pause rules for exam periods |
| C++/Python build issues on Windows | CI blocks | Spike S6.1 early, Linux-only fallback for the C++ part |
| Scope too broad | Nothing finished | Release after each 2 sprints, epics E8+ only refined when due |
| Duplicating SciPy without benefit | Weak portfolio value | Focus on block model, streaming, C++ path and benchmarks |
| Package name taken on PyPI | Rename later | Check name in S1 |

---

## 9. Open decisions

- [ ] Name (`sigstack` free on PyPI and GitHub?)
- [ ] License: Apache-2.0 or MIT
- [ ] Sprint start date
- [ ] Weekly time budget (6 h assumed)
- [ ] Order after v0.4.0: general modules first (E8 to E10) or radar/RF first (E13, E14)
- [ ] ML backend for E18: JAX or PyTorch
- [ ] Narrow scope to one showcase (e.g. FMCW radar: simulate → optimise → deploy) or keep the broad roadmap
