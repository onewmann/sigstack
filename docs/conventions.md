# Conventions

Rules that every block in sigstack follows. The code is in `python/sigstack/core/block.py`; decisions D-016 and D-017 in `docs/plan/DECISIONS.md` give the reasons.

## Array layout

- The last axis is time: a signal has shape `(..., n_samples)`.
- Leading axes are channels or batch dimensions. A multichannel signal has shape `(n_channels, n_samples)`; `(n_samples,)` is one channel.
- By default a block processes leading axes independently and returns them unchanged.
- A block that adds, removes or mixes axes before the last one, or changes the number of samples, documents its output shape and overrides `BaseBlock.output_shape(input_shape)`. Examples: an STFT turns `(..., N)` into `(..., n_freq, n_frames)`, a beamformer turns `(n_channels, N)` into `(n_beams, N)`. A pipeline can then check a chain before it runs.
- `n_samples = 0` is allowed. An empty chunk returns an empty array with the leading shape of the output.

## Streaming

- `process()` keeps the block's state between calls, so consecutive calls receive consecutive pieces of one signal.
- Chunk invariance: for any split of a signal along the last axis, processing the pieces one after the other gives the same result as processing the whole signal at once.
- `reset()` restores the state the block had right after construction.

## Dtypes and precision

- Boolean and integer arrays are converted to `float64` (`True` becomes 1.0), as NumPy does in arithmetic (D-016). String, object and date/time arrays raise `TypeError`.
- Scalar parameters stay strict: `fs=True`, `gain=True` or `n=True` raise `TypeError`.
- **The output has the precision of the input** (D-017): `float32` in gives `float32` out (or `complex64` for complex output), `float64` gives `float64` (or `complex128`). Parameters such as a gain are rounded to that precision.
- A block may compute internally at a higher precision, for example an IIR filter that keeps its coefficients and state in `float64`, but it returns the input precision.
- You choose the precision once, with the dtype of the input. For results like SciPy's, use `float64` input, which is the default anyway. A precision setting for a whole pipeline follows with story S1.6.

| Input dtype | Real output | Complex output |
|---|---|---|
| bool, integer | `float64` | `complex128` |
| `float32` | `float32` | `complex64` |
| `float64` | `float64` | `complex128` |
| `complex64` | — | `complex64` |
| `complex128` | — | `complex128` |

## Complex input

- Complex input gives complex output.
- A block that cannot process complex input sets the class attribute `SUPPORTS_COMPLEX = False`. `BaseBlock.process()` then raises `TypeError` for complex input, also when the imaginary part is zero, and names the block in the message.
