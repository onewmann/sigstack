#!/usr/bin/env python3
"""Create GitHub labels, milestones and issues from docs/plan/backlog.toml.

Idempotent: existing labels are updated, existing milestones and issues (matched by the
"[<story id>]" title prefix) are left alone. Without --apply it only prints what it would do.

Usage:
    python scripts/sync_issues.py --sprint 2            dry run for stories planned in sprint 2
    python scripts/sync_issues.py --sprint 2 --apply    create them
    python scripts/sync_issues.py --release v0.2.0 --apply
    python scripts/sync_issues.py --labels-only --apply

Requires an authenticated GitHub CLI (`gh auth status`). Uses the active profile from
plan.toml unless --profile is given.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plan as P  # noqa: E402

TYPE_COLORS = {
    "type:feature": ("1d76db", "New functionality"),
    "type:infra": ("5319e7", "Build, CI, tooling"),
    "type:docs": ("0075ca", "Documentation"),
    "type:bug": ("d73a4a", "Something is wrong"),
    "type:spike": ("fbca04", "Time-boxed investigation"),
}


def gh(args: list[str], apply: bool, capture: bool = False) -> str:
    shown = "gh " + " ".join(f'"{a}"' if " " in a else a for a in args)
    if not apply and not capture:
        print(f"[dry-run] {shown}")
        return ""
    res = subprocess.run(["gh", *args], check=True, text=True, capture_output=True)
    return res.stdout


def story_type(s: P.Story) -> str:
    if s.title.lower().startswith("spike"):
        return "type:spike"
    if s.epic == "E0":
        return "type:infra"
    if s.epic == "E12" and "example" not in s.title.lower():
        return "type:docs"
    return "type:feature"


def issue_body(s: P.Story, release: str) -> str:
    ac = "\n".join(f"- [ ] {a}" for a in s.ac)
    deps = ", ".join(s.deps) or "none"
    notes = f"\n\n**Note:** {s.notes}" if s.notes else ""
    return (
        f"**Story ID:** {s.id}  \n**Epic:** {s.epic}  \n**Points:** {s.points}  \n"
        f"**Release:** {release}  \n**Depends on:** {deps}\n\n"
        f"### User story\n{s.story}\n\n### Acceptance criteria\n{ac}{notes}\n\n"
        "### Definition of Done\nSee docs/ROADMAP.md, section 3.5.\n\n"
        "_Generated from docs/plan/backlog.toml by scripts/sync_issues.py. Edit the backlog, not this text._"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--profile")
    sel = ap.add_mutually_exclusive_group(required=True)
    sel.add_argument("--sprint", type=int)
    sel.add_argument("--release")
    sel.add_argument("--labels-only", action="store_true")
    ap.add_argument("--apply", action="store_true", help="really call gh (default: dry run)")
    args = ap.parse_args()

    backlog, plan = P.load(args.root)
    epics, stories, errors = P.parse_stories(backlog)
    v_err, _ = P.validate(epics, stories)
    profile = args.profile or plan["settings"]["profile"]
    sch = P.schedule(plan, stories, profile)
    if errors or v_err or sch.errors:
        print("Backlog has errors, run scripts/plan.py --check first.", file=sys.stderr)
        return 1

    # labels
    for name, (color, desc) in TYPE_COLORS.items():
        gh(["label", "create", name, "--color", color, "--description", desc, "--force"], args.apply)
    for eid, title in epics.items():
        gh(["label", "create", f"epic:{eid}", "--color", "c5def5", "--description", title, "--force"], args.apply)
    if args.labels_only:
        return 0

    by_id = {s.id: s for s in stories}
    if args.sprint is not None:
        chosen = [by_id[i] for i, sp in sch.placed.items() if sp == args.sprint]
        gh(["label", "create", f"sprint:{args.sprint}", "--color", "bfdadc", "--description",
            f"Sprint {args.sprint}", "--force"], args.apply)
    else:
        ri = next((i for i, r in enumerate(sch.releases) if r.version == args.release), None)
        if ri is None:
            print(f"unknown release {args.release} in profile {profile}", file=sys.stderr)
            return 1
        chosen = [s for s in stories if sch.release_of.get(s.id) == ri]
    chosen = [s for s in chosen if s.status not in {"done", "dropped"}]
    chosen.sort(key=lambda s: s.order)

    # milestones
    needed = sorted({sch.releases[sch.release_of[s.id]].version for s in chosen})
    existing_ms: set[str] = set()
    if args.apply:
        out = gh(["api", "repos/{owner}/{repo}/milestones?state=all&per_page=100"], True, capture=True)
        existing_ms = {m["title"] for m in json.loads(out or "[]")}
    for version in needed:
        if version not in existing_ms:
            title = next(r.title for r in sch.releases if r.version == version)
            gh(["api", "repos/{owner}/{repo}/milestones", "-f", f"title={version}", "-f", f"description={title}"],
               args.apply)

    # issues
    existing_titles: set[str] = set()
    if args.apply:
        out = gh(["issue", "list", "--state", "all", "--limit", "500", "--json", "title"], True, capture=True)
        existing_titles = {i["title"].split("]")[0] + "]" for i in json.loads(out or "[]")}
    for s in chosen:
        key = f"[{s.id}]"
        if key in existing_titles:
            print(f"skip {key} (exists)")
            continue
        release = sch.releases[sch.release_of[s.id]].version
        labels = [story_type(s), f"epic:{s.epic}"]
        if args.sprint is not None:
            labels.append(f"sprint:{args.sprint}")
        gh(["issue", "create", "--title", f"{key} {s.title}", "--body", issue_body(s, release),
            "--label", ",".join(labels), "--milestone", release], args.apply)
    print(f"{len(chosen)} stories selected ({'applied' if args.apply else 'dry run'}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
