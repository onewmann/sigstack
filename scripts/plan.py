#!/usr/bin/env python3
"""Validate the sigstack backlog, schedule it into sprints and render the plan.

Inputs (single source of truth):
    docs/plan/backlog.toml   epics and stories
    docs/plan/plan.toml      calendar, velocity, actuals, release profiles

Outputs (generated, never edit by hand):
    docs/plan/PLAN.md        release forecast, sprint schedule, profile comparison, warnings
    docs/plan/BACKLOG.md     all stories per epic with their planned sprint

Usage:
    python scripts/plan.py                 validate, schedule active profile, write outputs
    python scripts/plan.py --check         validate only, exit code 1 on errors (for CI)
    python scripts/plan.py --profile showcase
    python scripts/plan.py --no-write      print summary only
    python scripts/plan.py --record        also append the forecast to docs/plan/history.csv
                                           (used by /sprint-review, once per review)

Only the standard library is used (Python >= 3.11 for tomllib).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ALLOWED_POINTS = {1, 2, 3, 5, 8}
STATUSES = {"todo", "committed", "done", "dropped"}


# ------------------------------------------------------------------ data model
@dataclass
class Story:
    id: str
    epic: str
    group: str
    title: str
    story: str
    ac: list[str]
    points: int
    deps: list[str]
    status: str
    sprint: int | None
    notes: str = ""
    order: int = 0


@dataclass
class Release:
    version: str
    title: str
    groups: list[str]
    cv: str = ""


@dataclass
class Slot:
    index: int
    start: dt.date
    end: dt.date
    factor: float
    label: str
    sprint: int | None = None  # None for pause slots
    capacity: int = 0


@dataclass
class Schedule:
    profile: str
    velocity: float
    velocity_source: str
    slots: list[Slot]
    placed: dict[str, int]  # story id -> sprint number
    unscheduled: list[str]
    release_of: dict[str, int]  # story id -> release index
    releases: list[Release]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ loading
def load(root: Path) -> tuple[dict, dict]:
    with (root / "docs/plan/backlog.toml").open("rb") as f:
        backlog = tomllib.load(f)
    with (root / "docs/plan/plan.toml").open("rb") as f:
        plan = tomllib.load(f)
    return backlog, plan


def parse_stories(backlog: dict) -> tuple[dict[str, str], list[Story], list[str]]:
    errors: list[str] = []
    epics = {e["id"]: e["title"] for e in backlog.get("epic", [])}
    stories: list[Story] = []
    seen: set[str] = set()
    for i, raw in enumerate(backlog.get("story", [])):
        sid = raw.get("id", f"<story #{i}>")
        missing = [k for k in ("id", "epic", "group", "title", "story", "ac", "points") if k not in raw]
        if missing:
            errors.append(f"{sid}: missing fields {missing}")
            continue
        if sid in seen:
            errors.append(f"{sid}: duplicate id")
        seen.add(sid)
        stories.append(
            Story(
                id=sid,
                epic=raw["epic"],
                group=raw["group"],
                title=raw["title"],
                story=raw["story"],
                ac=list(raw["ac"]),
                points=int(raw["points"]),
                deps=list(raw.get("deps", [])),
                status=raw.get("status", "todo"),
                sprint=raw.get("sprint"),
                notes=raw.get("notes", ""),
                order=i,
            )
        )
    return epics, stories, errors


def parse_profile(plan: dict, name: str) -> tuple[str, list[Release]]:
    prof = plan["profiles"][name]
    releases = [
        Release(r["version"], r["title"], list(r["groups"]), r.get("cv", ""))
        for r in prof["release"]
    ]
    return prof.get("description", ""), releases


# ------------------------------------------------------------------ validation
def validate(epics: dict[str, str], stories: list[Story]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    ids = {s.id for s in stories}
    by_id = {s.id: s for s in stories}
    for s in stories:
        if s.epic not in epics:
            errors.append(f"{s.id}: unknown epic {s.epic}")
        if s.points not in ALLOWED_POINTS:
            errors.append(f"{s.id}: points {s.points} not in {sorted(ALLOWED_POINTS)}")
        elif s.points == 8:
            warnings.append(f"{s.id}: 8 points, split before it enters a sprint")
        if s.status not in STATUSES:
            errors.append(f"{s.id}: unknown status {s.status!r}")
        if s.status in {"committed", "done"} and not s.sprint:
            errors.append(f"{s.id}: status {s.status} needs a sprint number")
        if not s.ac:
            warnings.append(f"{s.id}: no acceptance criteria (not ready)")
        for d in s.deps:
            if d not in ids:
                errors.append(f"{s.id}: unknown dependency {d}")
            elif by_id[d].status == "dropped" and s.status != "dropped":
                warnings.append(f"{s.id}: depends on dropped story {d}")
            elif s.status == "done" and by_id[d].status != "done":
                errors.append(f"{s.id}: done, but dependency {d} is {by_id[d].status}")
    # cycle detection
    state: dict[str, int] = {}

    def visit(sid: str, path: list[str]) -> None:
        if state.get(sid) == 2:
            return
        if state.get(sid) == 1:
            errors.append("dependency cycle: " + " -> ".join(path + [sid]))
            return
        state[sid] = 1
        for d in by_id[sid].deps:
            if d in by_id:
                visit(d, path + [sid])
        state[sid] = 2

    for s in stories:
        visit(s.id, [])
    return errors, warnings


# ------------------------------------------------------------------ calendar
def build_slots(plan: dict, velocity: float) -> list[Slot]:
    st = plan["settings"]
    start = dt.date.fromisoformat(st["start"])
    days = int(st["sprint_days"])
    periods = [
        (dt.date.fromisoformat(p["from"]), dt.date.fromisoformat(p["to"]), float(p["factor"]), p.get("label", ""))
        for p in plan.get("period", [])
    ]
    slots: list[Slot] = []
    sprint_no = 0
    for i in range(int(st.get("horizon_slots", 80))):
        s = start + dt.timedelta(days=i * days)
        e = s + dt.timedelta(days=days - 1)
        factor, label = 1.0, ""
        for p_from, p_to, p_factor, p_label in periods:
            if p_from <= s <= p_to:
                factor, label = p_factor, p_label
                break
        slot = Slot(i, s, e, factor, label)
        if factor > 0:
            sprint_no += 1
            slot.sprint = sprint_no
            slot.capacity = int(math.floor(velocity * factor + 0.5))
        slots.append(slot)
    return slots


def measured_velocity(plan: dict, factors: dict[int, float]) -> tuple[float, str]:
    st = plan["settings"]
    actuals = sorted(plan.get("actual", []), key=lambda a: a["sprint"])
    if not actuals:
        return float(st["default_velocity"]), "default, no actuals yet"
    window = actuals[-int(st.get("velocity_window", 3)) :]
    normed = [a["done_points"] / factors.get(a["sprint"], 1.0) for a in window if factors.get(a["sprint"], 1.0) > 0]
    v = sum(normed) / len(normed)
    sprints = ", ".join(str(a["sprint"]) for a in window)
    return v, f"mean of sprints {sprints}, normalised to a full sprint"


# ------------------------------------------------------------------ scheduling
def schedule(plan: dict, stories: list[Story], profile: str) -> Schedule:
    _, releases = parse_profile(plan, profile)
    st = plan["settings"]
    errors: list[str] = []
    warnings: list[str] = []

    # factors per sprint number (for velocity normalisation), first with default velocity
    probe = build_slots(plan, float(st["default_velocity"]))
    factors = {s.sprint: s.factor for s in probe if s.sprint}
    velocity, vsrc = measured_velocity(plan, factors)
    slots = build_slots(plan, velocity)
    sprints = [s for s in slots if s.sprint]
    by_sprint = {s.sprint: s for s in sprints}

    group_release: dict[str, int] = {}
    for ri, rel in enumerate(releases):
        for g in rel.groups:
            if g in group_release:
                errors.append(f"[{profile}] group {g} assigned to two releases")
            group_release[g] = ri

    active = [s for s in stories if s.status != "dropped"]
    release_of: dict[str, int] = {}
    for s in active:
        if s.group not in group_release:
            errors.append(f"[{profile}] {s.id}: group {s.group!r} not in any release")
        else:
            release_of[s.id] = group_release[s.group]
    by_id = {s.id: s for s in active}
    for s in active:
        for d in s.deps:
            if d in release_of and s.id in release_of and release_of[d] > release_of[s.id]:
                errors.append(
                    f"[{profile}] {s.id} ({releases[release_of[s.id]].version}) depends on "
                    f"{d} ({releases[release_of[d]].version}), which comes later"
                )

    placed: dict[str, int] = {}
    load: dict[int, int] = {}
    for s in active:
        if s.status in {"done", "committed"} and s.sprint:
            placed[s.id] = s.sprint
            load[s.sprint] = load.get(s.sprint, 0) + s.points
    for sp, pts in load.items():
        if sp in by_sprint and pts > by_sprint[sp].capacity and not any(
            a["sprint"] == sp for a in plan.get("actual", [])
        ):
            warnings.append(f"Sprint {sp}: committed {pts} points, capacity {by_sprint[sp].capacity}")

    done_sprints = {a["sprint"] for a in plan.get("actual", [])}
    current = max(done_sprints, default=0) + 1
    pending = sorted(
        (s for s in active if s.status == "todo" and s.id in release_of),
        key=lambda s: (release_of[s.id], releases[release_of[s.id]].groups.index(s.group), s.order),
    )
    pull_ahead = int(st.get("pull_ahead", 1))

    for slot in sprints:
        if not pending:
            break
        if slot.sprint < current:
            continue
        remaining = slot.capacity - load.get(slot.sprint, 0)
        changed = True
        while changed and remaining > 0 and pending:
            changed = False
            head = min(release_of[s.id] for s in pending)
            for s in list(pending):
                if release_of[s.id] > head + pull_ahead:
                    break
                if s.points > remaining:
                    continue
                if all(d in placed and placed[d] <= slot.sprint for d in s.deps if d in by_id):
                    placed[s.id] = slot.sprint
                    load[slot.sprint] = load.get(slot.sprint, 0) + s.points
                    remaining -= s.points
                    pending.remove(s)
                    changed = True
                    break
    unscheduled = [s.id for s in pending]
    if unscheduled:
        warnings.append(f"[{profile}] not scheduled within horizon: {', '.join(unscheduled)}")
    return Schedule(profile, velocity, vsrc, slots, placed, unscheduled, release_of, releases, errors, warnings)


# ------------------------------------------------------------------ replanning checks
def parse_date(text: str) -> dt.date | None:
    try:
        return dt.datetime.strptime(text, "%d.%m.%Y").date()
    except (TypeError, ValueError):
        return None


def replan_checks(sch: Schedule, stories: list[Story], plan: dict, previous: dict | None) -> list[tuple[str, bool, str]]:
    """Evaluate the triggers from PLANNING.md. Returns (trigger, fired, detail)."""
    st = plan["settings"]
    planned = float(st.get("planned_velocity", st["default_velocity"]))
    days = int(st["sprint_days"])
    actuals = sorted(plan.get("actual", []), key=lambda a: a["sprint"])
    factors = {s.sprint: s.factor for s in sch.slots if s.sprint}
    checks: list[tuple[str, bool, str]] = []

    if len(actuals) >= 2:
        last2 = [a["done_points"] / factors.get(a["sprint"], 1.0) for a in actuals[-2:] if factors.get(a["sprint"], 1.0) > 0]
        v = sum(last2) / len(last2)
        dev = (v - planned) / planned
        checks.append(("Velocity off plan", abs(dev) > 0.25,
                       f"last two sprints {v:.1f} vs planned {planned:.0f} ({dev:+.0%}); threshold 25 %"))
    else:
        checks.append(("Velocity off plan", False, f"needs two measured sprints ({len(actuals)} so far)"))

    fc = release_forecast(sch, stories)
    if previous and len(actuals) >= 2:
        next_cv = next((r for r in fc if r["cv"] and r["points"] > r["done"]), None)
        for r, limit, label in [(next_cv, 1, "next CV milestone"),
                                (next((x for x in fc if x["version"].startswith("v1.0")), None), 3, "v1.0")]:
            if not r:
                continue
            old, new = parse_date(previous.get(r["version"], "")), parse_date(r["date"])
            if old and new:
                slip = (new - old).days / days
                checks.append((f"Milestone slip ({label} {r['version']})", slip > limit,
                               f"{previous[r['version']]} -> {r['date']} ({slip:+.1f} sprints); threshold {limit}"))
            else:
                checks.append((f"Milestone slip ({label} {r['version']})", False, "no previous forecast to compare"))
    else:
        checks.append(("Milestone slip", False, "needs two measured sprints and a previous forecast"))

    carried = [s.id for s in stories if s.notes.count("carried over") >= 2]
    checks.append(("Repeated carry-over", bool(carried), ", ".join(carried) or "none"))

    limit = 3 * planned
    big = []
    for ri, rel in enumerate(sch.releases):
        pts = sum(s.points for s in stories if sch.release_of.get(s.id) == ri and s.status != "dropped")
        if pts > limit:
            big.append(f"{rel.version} ({pts})")
    checks.append(("Release size", bool(big), (", ".join(big) or "none") + f"; limit {limit:.0f} points"))
    return checks


# ------------------------------------------------------------------ rendering
def fmt(d: dt.date) -> str:
    return d.strftime("%d.%m.%Y")


def release_forecast(sch: Schedule, stories: list[Story]) -> list[dict]:
    by_sprint = {s.sprint: s for s in sch.slots if s.sprint}
    rows = []
    for ri, rel in enumerate(sch.releases):
        members = [s for s in stories if sch.release_of.get(s.id) == ri and s.status != "dropped"]
        pts = sum(s.points for s in members)
        done = sum(s.points for s in members if s.status == "done")
        sprints = [sch.placed[s.id] for s in members if s.id in sch.placed]
        complete = len(sprints) == len(members) and members
        last = max(sprints) if sprints and complete else None
        rows.append(
            {
                "version": rel.version,
                "title": rel.title,
                "points": pts,
                "done": done,
                "sprint": last,
                "date": fmt(by_sprint[last].end) if last else "beyond horizon",
                "cv": rel.cv,
            }
        )
    return rows


def render_plan(sch: Schedule, others: list[Schedule], stories: list[Story], epics: dict[str, str],
                plan: dict, description: str, warnings: list[str],
                checks: list[tuple[str, bool, str]] | None = None) -> str:
    today = dt.date.today()
    by_id = {s.id: s for s in stories}
    active = [s for s in stories if s.status != "dropped"]
    total = sum(s.points for s in active)
    done = sum(s.points for s in active if s.status == "done")
    committed = sum(s.points for s in active if s.status == "committed")
    fc = release_forecast(sch, stories)
    last_sprint = max(sch.placed.values(), default=0)
    out: list[str] = []
    w = out.append
    w("# sigstack plan (generated)\n")
    w(f"> Generated by `scripts/plan.py` on {fmt(today)} from `backlog.toml` and `plan.toml`. "
      "Do not edit; change the inputs and run the script again.\n")
    w(f"**Active profile:** `{sch.profile}`: {description}  ")
    w(f"**Velocity used:** {sch.velocity:.1f} points per full sprint ({sch.velocity_source})  ")
    w(f"**Backlog:** {len(active)} stories, {total} points; done {done}, committed {committed}, "
      f"open {total - done - committed}  ")
    v1 = next((r for r in fc if r["version"].startswith("v1.0")), None)
    if v1:
        w(f"**Forecast v1.0.0:** {v1['date']} (sprint {v1['sprint']})\n")
    w("Forecasts assume the velocity above. They move with every sprint review.\n")

    w("## Release forecast\n")
    w("| Release | Content | Points | Done | Last sprint | Ready by | CV milestone |")
    w("|---|---|---|---|---|---|---|")
    for r in fc:
        w(f"| {r['version']} | {r['title']} | {r['points']} | {r['done']} | {r['sprint'] or '—'} | {r['date']} | {r['cv']} |")
    w("")

    if others:
        w("## Profile comparison\n")
        w("Same stories, different order. Use this to decide the open question on ordering.\n")
        w("| Profile | Milestone | Release | Ready by |")
        w("|---|---|---|---|")
        for o in [sch, *others]:
            for r in release_forecast(o, stories):
                if r["cv"] or r["version"].startswith("v1.0"):
                    w(f"| `{o.profile}` | {r['cv'] or 'v1.0 complete'} | {r['version']} | {r['date']} |")
        w("")

    if checks is not None:
        w("## Replanning check\n")
        w("Triggers from `PLANNING.md`. If one fires, run `/replan`.\n")
        w("| Trigger | Fired | Detail |")
        w("|---|---|---|")
        for name, fired, detail in checks:
            w(f"| {name} | {'**yes**' if fired else 'no'} | {detail} |")
        w("")

    w("## Sprint schedule\n")
    w("| Sprint | Dates | Capacity | Planned | Stories | Release work |")
    w("|---|---|---|---|---|---|")
    for slot in sch.slots:
        if slot.sprint is None:
            if slot.start <= max((s.end for s in sch.slots if s.sprint == last_sprint), default=slot.start):
                w(f"| — | {fmt(slot.start)} – {fmt(slot.end)} | 0 | | {slot.label} | |")
            continue
        if slot.sprint > last_sprint:
            break
        ids = [sid for sid, sp in sch.placed.items() if sp == slot.sprint]
        ids.sort(key=lambda i: by_id[i].order)
        pts = sum(by_id[i].points for i in ids)
        rels = sorted({sch.release_of[i] for i in ids if i in sch.release_of})
        rel_txt = ", ".join(sch.releases[r].version for r in rels)
        cap = f"{slot.capacity}" + (f" ({slot.label})" if slot.label else "")
        story_txt = ", ".join(f"{i} {by_id[i].title} ({by_id[i].points})" for i in ids)
        w(f"| {slot.sprint} | {fmt(slot.start)} – {fmt(slot.end)} | {cap} | {pts} | {story_txt} | {rel_txt} |")
    w("")

    w("## Epic progress\n")
    w("| Epic | Title | Stories | Points | Done |")
    w("|---|---|---|---|---|")
    for eid, title in epics.items():
        m = [s for s in active if s.epic == eid]
        w(f"| {eid} | {title} | {len(m)} | {sum(s.points for s in m)} | {sum(s.points for s in m if s.status == 'done')} |")
    w("")

    w("## Warnings\n")
    if warnings:
        for x in warnings:
            w(f"- {x}")
    else:
        w("None.")
    w("")

    hist = plan.get("_history", [])
    if hist:
        w("## Forecast history\n")
        w("One row per sprint review (`--record`). A date moving right means the plan slipped.\n")
        keys = list(hist[0].keys())
        w("| " + " | ".join(keys) + " |")
        w("|" + "---|" * len(keys))
        for row in hist[-12:]:
            w("| " + " | ".join(row[k] for k in keys) + " |")
        w("")

    w("## Calendar assumptions\n")
    w("| From | To | Factor | Label |")
    w("|---|---|---|---|")
    for p in plan.get("period", []):
        w(f"| {fmt(dt.date.fromisoformat(p['from']))} | {fmt(dt.date.fromisoformat(p['to']))} | {p['factor']} | {p.get('label', '')} |")
    w("")
    return "\n".join(out)


def render_backlog(sch: Schedule, stories: list[Story], epics: dict[str, str]) -> str:
    out: list[str] = []
    w = out.append
    w("# sigstack backlog (generated)\n")
    w(f"> Generated by `scripts/plan.py` on {fmt(dt.date.today())} from `backlog.toml`. "
      f"Planned sprints from profile `{sch.profile}`. Do not edit.\n")
    for eid, title in epics.items():
        m = [s for s in stories if s.epic == eid]
        if not m:
            continue
        w(f"## {eid} {title}\n")
        w("| ID | Title | Story | Acceptance criteria | Pts | Deps | Release | Status | Sprint |")
        w("|---|---|---|---|---|---|---|---|---|")
        for s in m:
            ac = "<br>".join(s.ac)
            if s.notes:
                ac += f"<br>*Note: {s.notes}*"
            rel = sch.releases[sch.release_of[s.id]].version if s.id in sch.release_of else "—"
            sp = sch.placed.get(s.id, "—")
            w(f"| {s.id} | {s.title} | {s.story} | {ac} | {s.points} | {', '.join(s.deps) or '—'} | {rel} | {s.status} | {sp} |")
        w("")
    return "\n".join(out)


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--check", action="store_true", help="validate only")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--record", action="store_true", help="append forecast to history.csv")
    args = ap.parse_args()

    backlog, plan = load(args.root)
    epics, stories, errors = parse_stories(backlog)
    v_err, warnings = validate(epics, stories)
    errors += v_err
    profile = args.profile or plan["settings"]["profile"]
    if profile not in plan["profiles"]:
        print(f"unknown profile {profile!r}", file=sys.stderr)
        return 2
    description, _ = parse_profile(plan, profile)

    schedules = {name: schedule(plan, stories, name) for name in plan["profiles"]}
    for sch in schedules.values():
        errors += sch.errors
        warnings += sch.warnings
    main_sch = schedules[profile]
    others = [s for n, s in schedules.items() if n != profile]

    for e in errors:
        print(f"ERROR   {e}")
    for x in warnings:
        print(f"WARNING {x}")
    if args.check:
        print(f"{len(stories)} stories, {len(errors)} errors, {len(warnings)} warnings")
        return 1 if errors else 0
    if errors:
        print("Not writing outputs because of errors.")
        return 1

    hist_path = args.root / "docs/plan/history.csv"
    fc = release_forecast(main_sch, stories)
    previous_rows: list[dict] = []
    if hist_path.exists():
        with hist_path.open(encoding="utf-8", newline="") as f:
            previous_rows = list(csv.DictReader(f))
    done_after = str(max((a["sprint"] for a in plan.get("actual", [])), default=0))
    # compare with the last row recorded before the current review
    prev = next((r for r in reversed(previous_rows) if r.get("after sprint") != done_after), None)
    checks = replan_checks(main_sch, stories, plan, prev)
    for name, fired, detail in checks:
        if fired:
            print(f"TRIGGER {name}: {detail}")
    if args.record:
        active = [s for s in stories if s.status != "dropped"]
        row = {
            "date": fmt(dt.date.today()),
            "after sprint": str(max((a["sprint"] for a in plan.get("actual", [])), default=0)),
            "profile": profile,
            "velocity": f"{main_sch.velocity:.1f}",
            "done pts": str(sum(s.points for s in active if s.status == "done")),
            "total pts": str(sum(s.points for s in active)),
        }
        for r in fc:
            if r["cv"] or r["version"].startswith("v1.0"):
                row[r["version"]] = r["date"]
        existing = []
        if hist_path.exists():
            with hist_path.open(encoding="utf-8", newline="") as f:
                existing = list(csv.DictReader(f))
        if not existing or {k: existing[-1].get(k) for k in row} != row:
            fields = list(dict.fromkeys([*(existing[0].keys() if existing else []), *row.keys()]))
            with hist_path.open("w", encoding="utf-8", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=fields, restval="")
                wr.writeheader()
                wr.writerows(existing + [row])
            print(f"recorded forecast in {hist_path}")
    if hist_path.exists():
        with hist_path.open(encoding="utf-8", newline="") as f:
            plan["_history"] = list(csv.DictReader(f))
    for r in fc:
        print(f"{r['version']:>9}  {r['date']:>14}  sprint {str(r['sprint'] or '-'):>3}  {r['title']}")
    if not args.no_write:
        out_dir = args.root / "docs/plan"
        (out_dir / "PLAN.md").write_text(
            render_plan(main_sch, others, stories, epics, plan, description, warnings, checks), encoding="utf-8"
        )
        (out_dir / "BACKLOG.md").write_text(render_backlog(main_sch, stories, epics), encoding="utf-8")
        print(f"wrote {out_dir / 'PLAN.md'} and {out_dir / 'BACKLOG.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
