"""The London Underground, as data: stations, lines, travel times and routes.

Everything here is deterministic. The language model never chooses a station:
every station name that reaches a user comes from `tube_network.json`, a
snapshot of the TfL Unified API (Powered by TfL Open Data) taken on
2026-11-20 and shipped beside this file. `meta` inside it says exactly which
endpoints it came from.

Hop times are the minutes between adjacent stations in TfL's timetables. A
negative value marks one of the three hops TfL's timetables did not cover; it
is an estimate from straight-line distance, and its absolute value is used.
"""

from __future__ import annotations

import difflib
import heapq
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

DATA = json.loads(Path(__file__).with_name("tube_network.json").read_text("utf-8"))
STATIONS: dict = DATA["stations"]
LINES: dict = DATA["lines"]
HOPS: dict = DATA["hops"]
META: dict = DATA["meta"]

#: Minutes added for changing trains: walking between platforms and waiting.
INTERCHANGE_MINUTES = 5.0

#: When nothing names a line, these are tried in order. The Victoria line
#: comes first because most of its stations have a well-known landmark.
DEFAULT_LINES = ("victoria", "northern", "central", "piccadilly", "jubilee", "district")

#: TfL's disambiguating suffixes, rewritten for people.
_DISPLAY = {
    "(Dist&Picc Line)": "(District and Piccadilly)",
    "(H&C Line)": "(Hammersmith & City)",
    "(Circle Line)": "(Circle)",
}

#: Other rail services people call "lines", which this app does not cover.
NOT_TUBE = ("elizabeth", "overground", "dlr", "docklands", "tram", "thameslink",
            "crossrail", "national rail", "liberty", "lioness", "mildmay",
            "suffragette", "weaver", "windrush")

_LINE_ALIASES = {
    "hammersmith and city": "hammersmith-city",
    "hammersmith city": "hammersmith-city",
    "h and c": "hammersmith-city",
    "handc": "hammersmith-city",
    "waterloo and city": "waterloo-city",
    "waterloo city": "waterloo-city",
    "drain": "waterloo-city",
    "the drain": "waterloo-city",
    "met": "metropolitan",
}


class StationNotFound(LookupError):
    def __init__(self, query: str):
        super().__init__(query)
        self.query = query


class LineNotFound(LookupError):
    def __init__(self, query: str, not_tube: bool = False):
        super().__init__(query)
        self.query = query
        self.not_tube = not_tube


class RouteTooShort(ValueError):
    def __init__(self, available: int, needed: int):
        super().__init__(f"{available} stations for {needed} concepts")
        self.available = available
        self.needed = needed


class NoRoute(ValueError):
    pass


def norm(text: str) -> str:
    """Lower case, '&' as 'and', no apostrophes, punctuation as spaces."""
    text = (text or "").lower().replace("&", " and ")
    text = text.replace("'", "").replace("’", "")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _query_norm(text: str) -> str:
    words = norm(text).split()
    swaps = {"saint": "st", "rd": "road", "sq": "square", "stn": "station"}
    words = [swaps.get(w, w) for w in words]
    while words and words[-1] in ("station", "underground", "tube"):
        words.pop()
    while words and words[0] in ("the",):
        words.pop(0)
    return " ".join(words)


def display_name(sid: str) -> str:
    name = STATIONS[sid]["n"]
    for raw, nice in _DISPLAY.items():
        name = name.replace(raw, nice)
    return name


def base_name(sid: str) -> str:
    """The name without TfL's '(Central)'-style suffix."""
    return re.sub(r"\s*\([^)]*\)\s*$", "", STATIONS[sid]["n"]).strip()


def line_name(line_id: str) -> str:
    return LINES[line_id]["name"]


def anchor(sid: str) -> str:
    return STATIONS[sid].get("a", "")


def zone(sid: str) -> str:
    return STATIONS[sid].get("z", "")


def lines_at(sid: str) -> list[str]:
    return list(STATIONS[sid]["l"])


def hop_minutes(a: str, b: str) -> float:
    value = HOPS.get("|".join(sorted((a, b))))
    return abs(value) if value is not None else 2.0


# ── Lookups ─────────────────────────────────────────────────────────────────

_INDEX: dict[str, set[str]] = {}
for _sid in STATIONS:
    _INDEX.setdefault(norm(STATIONS[_sid]["n"]), set()).add(_sid)
    _INDEX.setdefault(norm(base_name(_sid)), set()).add(_sid)
_KEYS = sorted(_INDEX)


def _best(candidates: set[str], prefer_line: str | None) -> str:
    return sorted(candidates, key=lambda s: (
        0 if prefer_line and prefer_line in STATIONS[s]["l"] else 1,
        len(STATIONS[s]["n"]), s))[0]


def find_station(query: str, prefer_line: str | None = None) -> str:
    """The station id for what somebody typed, or StationNotFound."""
    q = _query_norm(query)
    if not q:
        raise StationNotFound(query)
    if q in _INDEX:
        return _best(_INDEX[q], prefer_line)
    starts = {s for k in _KEYS if k.startswith(q + " ") for s in _INDEX[k]}
    if starts:
        return _best(starts, prefer_line)
    pattern = re.compile(r"(^| )" + re.escape(q) + r"( |$)")
    within = {s for k in _KEYS if pattern.search(k) for s in _INDEX[k]}
    if within and len(q) >= 4:
        return _best(within, prefer_line)
    close = difflib.get_close_matches(q, _KEYS, n=3, cutoff=0.75)
    if close:
        return _best(_INDEX[close[0]], prefer_line)
    raise StationNotFound(query)


def find_line(query: str) -> str:
    """The line id for what somebody typed, or LineNotFound."""
    q = norm(query)
    q = re.sub(r"\b(line|lines|tube|underground)\b", " ", q)
    q = " ".join(w for w in q.split() if w != "the") or q
    if not q:
        raise LineNotFound(query)
    if any(word in q for word in NOT_TUBE):
        raise LineNotFound(query, not_tube=True)
    if q in _LINE_ALIASES:
        return _LINE_ALIASES[q]
    for line_id, line in LINES.items():
        if q in (norm(line_id), norm(line["name"])):
            return line_id
    close = difflib.get_close_matches(q, [norm(l["name"]) for l in LINES.values()], n=1, cutoff=0.7)
    if close:
        return next(i for i, l in LINES.items() if norm(l["name"]) == close[0])
    raise LineNotFound(query)


# ── Routes ──────────────────────────────────────────────────────────────────


@dataclass
class Stop:
    """One station on a route, and how the traveller got there."""
    sid: str
    line: str                      # the line ridden INTO this station (first: the line boarded)
    minutes: float = 0.0           # from the previous Stop
    passing: list[str] = field(default_factory=list)   # stations passed on the way, not stopped at


_ADJ: dict[tuple[str, str], set[str]] = {}
for _line_id, _line in LINES.items():
    for _route in _line["routes"]:
        for _a, _b in zip(_route["s"], _route["s"][1:]):
            _ADJ.setdefault((_a, _line_id), set()).add(_b)
            _ADJ.setdefault((_b, _line_id), set()).add(_a)


def shortest_path(start: str, goal: str, prefer_line: str | None = None) -> list[Stop]:
    """Every station from start to goal, fewest minutes first (Dijkstra).

    The state is (station, line), so a change of line costs
    INTERCHANGE_MINUTES. A named line is preferred: other lines cost a
    little more per hop, so a tie goes to the line the person asked for.
    """
    if start == goal:
        return [Stop(start, (prefer_line if prefer_line in lines_at(start) else lines_at(start)[0]))]
    heap: list = []
    best: dict = {}
    for line_id in lines_at(start):
        state = (start, line_id)
        best[state] = 0.0
        heapq.heappush(heap, (0.0, start, line_id, None))
    came: dict = {}
    while heap:
        cost, sid, line_id, prev = heapq.heappop(heap)
        state = (sid, line_id)
        if cost > best.get(state, float("inf")):
            continue
        if sid == goal:
            return _unwind(came, state)
        moves = [((nbr, line_id), hop_minutes(sid, nbr) + (0 if not prefer_line or line_id == prefer_line else 0.5))
                 for nbr in _ADJ.get(state, ())]
        moves += [((sid, other), INTERCHANGE_MINUTES) for other in lines_at(sid) if other != line_id]
        for nxt, step in moves:
            new = cost + step
            if new < best.get(nxt, float("inf")):
                best[nxt] = new
                came[nxt] = state
                heapq.heappush(heap, (new, nxt[0], nxt[1], state))
    raise NoRoute(f"{start} -> {goal}")


def _unwind(came: dict, state: tuple[str, str]) -> list[Stop]:
    chain = [state]
    while chain[-1] in came:
        chain.append(came[chain[-1]])
    chain.reverse()
    stops: list[Stop] = []
    for sid, line_id in chain:
        if stops and stops[-1].sid == sid:      # a change of line: same station
            continue
        minutes = hop_minutes(stops[-1].sid, sid) if stops else 0.0
        stops.append(Stop(sid, line_id, minutes))
    # The first stop's line is the line actually boarded.
    if len(stops) > 1:
        stops[0].line = stops[1].line
    # A change of line at a station costs time on the way to the next one.
    for prev, cur in zip(stops, stops[1:]):
        if cur.line != prev.line:
            cur.minutes += INTERCHANGE_MINUTES
    return stops


def continue_along(stops: list[Stop], line_id: str, count: int, backwards: bool = False) -> list[Stop]:
    """Up to `count` more stations past the end (or before the start) of a
    route, staying on one line and never revisiting a station."""
    if count <= 0:
        return []
    seen = {s.sid for s in stops}
    end = stops[0].sid if backwards else stops[-1].sid
    near = (stops[1].sid if len(stops) > 1 else None) if backwards else (stops[-2].sid if len(stops) > 1 else None)
    options: list[list[str]] = []
    for route in LINES[line_id]["routes"]:
        seq = route["s"]
        if end not in seq:
            continue
        i = seq.index(end)
        for direction in (seq[i + 1:], list(reversed(seq[:i]))):
            if near and direction and direction[0] == near:
                continue
            run = []
            for sid in direction:
                if sid in seen:
                    break
                run.append(sid)
            options.append(run)
    if not options:
        return []
    run = max(options, key=lambda r: (len(r), sum(1 for s in r if anchor(s))))[:count]
    extra, prev = [], end
    for sid in run:
        extra.append(Stop(sid, line_id, hop_minutes(prev, sid)))
        prev = sid
    return extra


def _zone_number(sid: str) -> float:
    numbers = [float(z) for z in re.findall(r"\d+", zone(sid))]
    return sum(numbers) / len(numbers) if numbers else 9.0


def central_window(line_id: str, count: int) -> list[Stop]:
    """`count` consecutive stations on a line, as central as possible and
    with as many landmark anchors as possible."""
    best_key, best_seq = None, None
    for route in LINES[line_id]["routes"]:
        seq = route["s"]
        for i in range(0, len(seq) - count + 1):
            window = seq[i:i + count]
            key = (sum(_zone_number(s) for s in window) - 0.75 * sum(1 for s in window if anchor(s)), route["name"], i)
            if best_key is None or key < best_key:
                best_key, best_seq = key, window
    if best_seq is None:
        longest = max(len(r["s"]) for r in LINES[line_id]["routes"])
        raise RouteTooShort(longest, count)
    return _as_stops(best_seq, line_id)


def from_station(start: str, count: int, line_id: str | None = None, reverse: bool = False) -> list[Stop]:
    """`count` stations starting (or, reversed, ending) at a station, on the
    named line or whichever line through it goes furthest."""
    candidates = [line_id] if line_id else lines_at(start)
    best = None
    for lid in candidates:
        seed = [Stop(start, lid)]
        run = continue_along(seed, lid, count - 1)
        key = (len(run) >= count - 1, lid == (line_id or ""), -DEFAULT_LINES.index(lid) if lid in DEFAULT_LINES else -99,
               sum(1 for s in run if anchor(s.sid)))
        if best is None or key > best[0]:
            best = (key, seed + run)
    stops = best[1] if best else [Stop(start, lines_at(start)[0])]
    if len(stops) < count:
        raise RouteTooShort(len(stops), count)
    if reverse:
        return _as_stops([s.sid for s in reversed(stops)], stops[-1].line)
    return stops


def _as_stops(sids: list[str], line_id: str) -> list[Stop]:
    stops, prev = [], None
    for sid in sids:
        stops.append(Stop(sid, line_id, hop_minutes(prev, sid) if prev else 0.0))
        prev = sid
    return stops


def pick_stations(path: list[Stop], count: int) -> list[Stop]:
    """Spread `count` stops evenly along a longer path, keeping both ends.

    The stations skipped over are kept on the next chosen stop as `passing`,
    with their minutes added in, so the journey still tells the truth about
    the line between stops.
    """
    if count >= len(path):
        return path
    if count == 1:
        indices = [0]
    else:
        indices = sorted({round(i * (len(path) - 1) / (count - 1)) for i in range(count)})
    chosen: list[Stop] = []
    for n, idx in enumerate(indices):
        stop = path[idx]
        if n == 0:
            chosen.append(Stop(stop.sid, stop.line, 0.0))
            continue
        between = path[indices[n - 1] + 1:idx + 1]
        minutes = sum(s.minutes for s in between)
        chosen.append(Stop(stop.sid, stop.line, minutes, [s.sid for s in between[:-1]]))
    return chosen


def plan(count: int, line_id: str | None = None, start: str | None = None,
         end: str | None = None) -> tuple[list[Stop], list[str]]:
    """A route with exactly `count` stops, and any notes about how it was
    bent to fit. Raises RouteTooShort or NoRoute."""
    notes: list[str] = []
    if start and end and start != end:
        path = shortest_path(start, end, line_id)
        if len(path) < count:
            more = continue_along(path, path[-1].line, count - len(path))
            if more:
                notes.append(f"That stretch had only {len(path)} stations, so the route carries on "
                             f"to {display_name(more[-1].sid)}.")
                path = path + more
        if len(path) < count:
            before = continue_along(path, path[0].line, count - len(path), backwards=True)
            if before:
                notes.append(f"The route starts earlier, at {display_name(before[-1].sid)}, to fit everything in.")
                head = _as_stops([s.sid for s in reversed(before)] + [path[0].sid], path[0].line)
                path = head + path[1:]
        if len(path) < count:
            raise RouteTooShort(len(path), count)
        return pick_stations(path, count), notes
    if start or end:
        return from_station(start or end, count, line_id, reverse=bool(end and not start)), notes
    for lid in ([line_id] if line_id else list(DEFAULT_LINES)):
        try:
            return central_window(lid, count), notes
        except RouteTooShort:
            if line_id:
                raise
    raise RouteTooShort(0, count)


def interchanges(sid: str, riding: str) -> list[str]:
    """The other lines you can change to here, by name."""
    return [line_name(l) for l in sorted(lines_at(sid)) if l != riding]


#: Station names that are also ordinary words: never treated as a sign that
#: generated text mentions a station.
_COMMON = {"bank", "angel", "temple", "oval", "borough", "monument", "victoria",
           "arsenal", "embankment", "kilburn", "morden", "richmond", "stratford",
           "waterloo", "westminster", "paddington", "barbican", "hampstead",
           "pimlico", "brixton", "euston", "wimbledon", "highgate", "aldgate",
           "moorgate", "canada water", "green park", "white city", "seven sisters",
           "old street", "blackfriars", "hounslow west", "north greenwich"}

_MENTIONABLE = sorted(
    {norm(base_name(s)) for s in STATIONS if norm(base_name(s)) not in _COMMON
     and len(norm(base_name(s)).split()) >= 2},
    key=len, reverse=True)


def stations_mentioned(text: str) -> set[str]:
    """Normalised multi-word station names that appear in some text."""
    t = " " + norm(text) + " "
    return {name for name in _MENTIONABLE if f" {name} " in t}
