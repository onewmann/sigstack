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

There is no public API yet. The first blocks (`Block` interface, window functions) are planned for Sprint 1.

## Roadmap

Architecture, backlog and release plan: [docs/ROADMAP.md](docs/ROADMAP.md).
Current sprint: [docs/sprints/sprint-01.md](docs/sprints/sprint-01.md).

## License

Apache-2.0, see [LICENSE](LICENSE).
