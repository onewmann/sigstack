#!/usr/bin/env bash
# Creates labels, milestones and the issues for Sprint 1 and 2 in the current GitHub repo.
# Requires: gh (authenticated). Safe to re-run for labels; issues are created again on re-run.
set -euo pipefail

REPO="$(gh repo view --json nameWithOwner -q .nameWithOwner)"
echo "Repository: $REPO"

# ---------- labels ----------
label() { gh label create "$1" --color "$2" --description "$3" --force >/dev/null; echo "label $1"; }

label "type:feature" "1d76db" "New functionality"
label "type:bug"     "d73a4a" "Something is wrong"
label "type:docs"    "0075ca" "Documentation"
label "type:infra"   "5319e7" "Build, CI, tooling"
label "type:spike"   "fbca04" "Time-boxed investigation"
label "epic:E0-infra"   "c5def5" "Project infrastructure"
label "epic:E1-core"    "c5def5" "Core and block model"
label "epic:E2-signals" "c5def5" "Signal generators and metrics"
label "sprint:1" "bfdadc" "Sprint 1"
label "sprint:2" "bfdadc" "Sprint 2"

# ---------- milestones ----------
milestone() {
  if gh api "repos/$REPO/milestones?state=all" -q '.[].title' | grep -qx "$1"; then
    echo "milestone $1 exists"
  else
    gh api "repos/$REPO/milestones" -f title="$1" -f description="$2" >/dev/null
    echo "milestone $1"
  fi
}

milestone "v0.1.0" "Infra, block model, signals, metrics"
milestone "v0.2.0" "Filters, spectral analysis"
milestone "v0.3.0" "Pipeline + CLI, first C++ kernels"
milestone "v0.4.0" "Multirate, filter banks, benchmarks, Monte Carlo runner"

# ---------- issues ----------
issue() { # id, title, story, ac, points, labels
  local body
  body=$(cat <<EOF
**Story ID:** $1
**Points:** $5

### User story
$3

### Acceptance criteria
$4

### Definition of Done
See docs/ROADMAP.md, section 3.5.
EOF
)
  gh issue create --title "[$1] $2" --body "$body" --label "$6" --milestone "v0.1.0" >/dev/null
  echo "issue $1"
}

# Sprint 1
issue "S0.1" "pixi environment" \
  "As a developer I want a pixi environment so that the setup runs with one command on Fedora and Windows." \
  "- \`pixi install && pixi run check\` works on Fedora and Windows" \
  2 "type:infra,epic:E0-infra,sprint:1"

issue "S0.2" "CI on every push and PR" \
  "As a developer I want CI on every push so that broken code never reaches main." \
  "- Workflow runs lint, typecheck and tests on Ubuntu and Windows
- Required status check for merging into main" \
  2 "type:infra,epic:E0-infra,sprint:1"

issue "S0.5" "pre-commit hooks" \
  "As a developer I want pre-commit hooks so that style is consistent." \
  "- ruff, ruff-format, mypy active
- \`pre-commit run --all-files\` passes" \
  1 "type:infra,epic:E0-infra,sprint:1"

issue "S1.1" "Block interface" \
  "As a user I want a common Block interface so that all steps are interchangeable." \
  "- Protocol defined and documented (contract incl. chunk invariance)
- Used by a Gain block with tests" \
  2 "type:feature,epic:E1-core,sprint:1"

issue "S1.3" "Window functions" \
  "As a user I want standard windows (Hann, Hamming, Blackman, Blackman-Harris, Kaiser, flat-top)." \
  "- Match scipy.signal.get_window to 1e-12
- Coherent gain and ENBW helpers" \
  1 "type:feature,epic:E1-core,sprint:1"

# Sprint 2
issue "S0.3" "pip-installable build with scikit-build-core" \
  "As a user I want pip install . to build the package so that I can use it in other projects." \
  "- scikit-build-core build backend
- Wheel installs in a clean environment" \
  3 "type:infra,epic:E0-infra,sprint:2"

issue "S1.2" "Generic chunking test" \
  "As a developer I want an automatic chunking test so that every stateful block is stream-safe." \
  "- Hypothesis-based test applied to all registered blocks
- Random chunk sizes incl. length 0 and 1" \
  3 "type:feature,epic:E1-core,sprint:2"

issue "S1.4" "Complex and multichannel support in the block contract" \
  "As a developer I want complex dtypes and multichannel arrays (channels x samples) in the block contract from the start." \
  "- Dummy block passes tests for real, complex, 1-channel and N-channel input" \
  2 "type:feature,epic:E1-core,sprint:2"

issue "S2.1" "Seeded test signals with ground truth" \
  "As a tester I want seeded test signals (multitone, chirp, noise, impulse, AM/FM) with known ground truth." \
  "- Generators return signal + metadata (frequencies, energies)
- Same seed gives identical output" \
  3 "type:feature,epic:E2-signals,sprint:2"

echo "Done. Next: create the Project board in the browser and add the issues."
