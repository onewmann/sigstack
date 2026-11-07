# sigstack

[![CI](https://github.com/onewmann/sigstack/actions/workflows/ci.yml/badge.svg)](https://github.com/onewmann/sigstack/actions/workflows/ci.yml)

C++/Python toolkit for signal processing with a uniform block-processing model for offline and streaming use, and a simulation-based design loop.

Status: early development. Nothing here is usable yet.

## Install

The package is not published. Development setup uses [pixi](https://pixi.sh) (Linux and Windows):

```bash
git clone https://github.com/onewmann/sigstack.git
cd sigstack
pixi install      # creates the environment from pixi.lock
pixi run check    # lint + typecheck + tests
```

Without pixi, from a clone (pip 25.1 or newer for `--group`):

```bash
python -m pip install -e . --group test
python -m pytest
```

## Quickstart

```python
import numpy as np

from sigstack import Gain, enbw, get_window

w = get_window("hann", 1024)  # periodic Hann window
print(round(enbw(w), 3))  # 1.5 (equivalent noise bandwidth in bins)

gain = Gain(0.5, fs=48_000.0)
x = np.ones((2, 4))  # 2 channels, 4 samples: samples on the last axis
print(gain.process(x).shape)  # (2, 4)
```

Every block follows the same contract (`sigstack.Block`): `process()` keeps its state between calls, so a signal can be processed in one call or in chunks with the same result.

## Roadmap

Architecture, backlog and release plan: [docs/ROADMAP.md](docs/ROADMAP.md).
Current sprint: the highest-numbered file in [docs/sprints/](docs/sprints/) that is not marked as a draft.

## License

Apache-2.0, see [LICENSE](LICENSE).
