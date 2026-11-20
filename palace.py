"""The memory palace itself: concepts, stations, scenes, journey, grading.

Everything in this file is deterministic. Given the same material and the
same route inputs it builds the same palace, which is what lets a stateless
app walk a journey, quiz it and strengthen it across separate calls: each
call rebuilds the palace from what the person said, and gets the same one.

Claude (see `app.py` and `llm.py`) is only ever asked for words: which
concepts a block of prose contains, a vivid scene for a concept at a station,
a verdict on an answer the rules here cannot settle. It never chooses a
station, an order or a score.
"""

from __future__ import annotations

import difflib
import re
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import tube

MAX_STOPS = 12
DEFAULT_STOPS = 10
MIN_CONCEPTS = 2
SCENE_WORD_LIMIT = 60

#: Days until the next review, by how many reviews have gone well so far.
REVIEW_DAYS = (1, 3, 7, 14, 30)
#: Below this share correct, practise again soon instead of spacing it out.
PASS_MARK = 0.6
RETRY_MINUTES = 20


@dataclass(frozen=True)
class Concept:
    term: str
    detail: str = ""
    aliases: tuple[str, ...] = ()


@dataclass
class Material:
    concepts: list[Concept]
    kind: str                       # "preset", "list", "prose" or "topic"
    title: str
    preset: str | None = None
    raw: str = ""
    notes: list[str] = field(default_factory=list)


# ── Presets: material that needs no typing ─────────────────────────────────

PRESETS: dict[str, dict] = {
    "bayes": {
        "title": "Bayes' theorem",
        "line": "victoria", "from": "Brixton", "to": "Victoria",
        "concepts": [
            Concept("Prior", "P(H), what you believed before the evidence: say 1 in 100 people have the condition",
                    ("p(h)", "prior probability", "base rate", "p of h")),
            Concept("Likelihood", "P(E|H), how probable the evidence is if the hypothesis is true: a 90% chance of a positive test",
                    ("p(e|h)", "p e given h", "likelihood function")),
            Concept("Evidence", "P(E), the total chance of the evidence: the sum of P(E|H)P(H) over every hypothesis, here 0.0585",
                    ("p(e)", "marginal likelihood", "marginal", "total probability")),
            Concept("Normalisation", "divide by P(E) so that the posteriors of all the hypotheses add up to 1",
                    ("normalization", "normalising", "normalizing", "normalising constant", "normalizing constant", "divide by p(e)")),
            Concept("Posterior", "P(H|E) = P(E|H)P(H)/P(E): your updated belief after the evidence, about 15%",
                    ("p(h|e)", "posterior probability", "p h given e")),
        ],
        # Hand-written scenes for the demo: used when Claude is unavailable,
        # and each uses only the landmark recorded for its station.
        "scenes": [
            "On Electric Avenue, before any evidence arrives, a busker under the Bowie mural sings your PRIOR on a loop: \"one in a hundred, one in a hundred\". P(H) is what you believed before boarding.",
            "Deep in Stockwell's wartime shelter a warden shines a torch at a test result: \"IF you have it, how LIKELY is this?\" The torch beam spells P(E|H): ninety per cent.",
            "Inside MI6 at Vauxhall Cross, agents pile up every report: true alarms from the sick, false alarms from the healthy. Their whiteboard: EVIDENCE, P(E), the sum of P(E|H)P(H), equals 0.0585.",
            "In Tate Britain a curator rescales every painting until the frames fill exactly one wall, no more, no less. NORMALISATION: divide by P(E) so the posteriors sum to one.",
            "Queen Victoria herself reads the verdict at the terminus: P(H|E) = P(E|H)P(H)/P(E), about fifteen per cent. Your POSTERIOR, and tomorrow's prior.",
        ],
    },
    "planets": {
        "title": "The planets, in order from the Sun",
        "concepts": [
            Concept("Mercury", "closest to the Sun and the smallest planet"),
            Concept("Venus", "the hottest planet, under thick clouds"),
            Concept("Earth", "our home, the only known planet with life"),
            Concept("Mars", "the red planet, with the tallest volcano"),
            Concept("Jupiter", "the largest planet, with the Great Red Spot"),
            Concept("Saturn", "famous for its bright rings"),
            Concept("Uranus", "an ice giant tipped on its side"),
            Concept("Neptune", "the farthest planet, with the fastest winds"),
        ],
    },
    "wives": {
        "title": "The six wives of Henry VIII",
        "concepts": [
            Concept("Catherine of Aragon", "divorced", ("aragon",)),
            Concept("Anne Boleyn", "beheaded", ("boleyn",)),
            Concept("Jane Seymour", "died, after giving birth to Edward", ("seymour",)),
            Concept("Anne of Cleves", "divorced", ("cleves",)),
            Concept("Catherine Howard", "beheaded", ("howard",)),
            Concept("Catherine Parr", "survived", ("parr",)),
        ],
    },
}


def preset_for(text: str) -> str | None:
    words = set(tube.norm(text).split())
    if len(words) > 10:
        return None
    if words & {"bayes", "bayesian", "bayess"}:
        return "bayes"
    if words & {"planet", "planets"} or {"solar", "system"} <= words:
        return "planets"
    if "wives" in words and ("henry" in words or "viii" in words or "8th" in words):
        return "wives"
    return None


# ── Reading the material ────────────────────────────────────────────────────

_BULLET = re.compile(r"^\s*(?:[-*•·>]+|\(?\d{1,2}[.)]|\(?[a-z][.)](?=\s))\s*", re.IGNORECASE)
_SPLIT_TERM = re.compile(r"\s*(?::|\s[-–—=]\s|=)\s*")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _clip(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(",;:") + "…"


def _concept_from(item: str) -> Concept | None:
    item = _BULLET.sub("", item).strip().strip(".;,")
    if not item:
        return None
    parts = _SPLIT_TERM.split(item, maxsplit=1)
    if len(parts) == 2 and parts[0] and len(parts[0].split()) <= 8:
        return Concept(_clip(parts[0], 80), _clip(parts[1], 200))
    return Concept(_clip(item, 80))


def _list_items(text: str) -> list[str] | None:
    lines = [line for line in re.split(r"[\r\n]+", text) if line.strip()]
    if len(lines) >= 2:
        return lines
    one = lines[0] if lines else ""
    numbered = [p for p in re.split(r"(?:^|\s)\(?\d{1,2}[.)]\s+", one) if p.strip()]
    if len(numbered) >= 3:
        return numbered
    if one.count(";") >= 1:
        return [p for p in one.split(";") if p.strip()]
    commas = [p.strip() for p in one.split(",") if p.strip()]
    if len(commas) >= 3 and all(len(p.split()) <= 10 for p in commas):
        last = re.split(r"\s+(?:and|&)\s+", commas[-1])
        return commas[:-1] + last if all(last) else commas
    arrows = [p for p in re.split(r"\s*(?:->|→|\bthen\b)\s*", one) if p.strip()]
    if len(arrows) >= 3:
        return arrows
    return None


def read_material(text: str, limit: int) -> Material:
    """Concepts from what the person typed, or a Material of kind 'prose' or
    'topic' with no concepts yet, which needs Claude to read it."""
    raw = (text or "").strip()
    key = preset_for(raw)
    if key:
        preset = PRESETS[key]
        concepts = list(preset["concepts"])[:limit]
        notes = [] if len(preset["concepts"]) <= limit else \
            [f"Only the first {limit} of {len(preset['concepts'])} items fit on the route."]
        return Material(concepts, "preset", preset["title"], key, raw, notes)
    items = _list_items(raw)
    if items:
        concepts, seen = [], set()
        for item in items:
            concept = _concept_from(item)
            if concept and tube.norm(concept.term) not in seen:
                seen.add(tube.norm(concept.term))
                concepts.append(concept)
        notes = []
        if len(concepts) > limit:
            notes.append(f"Only the first {limit} of {len(concepts)} items fit on the route; "
                         "build a second palace for the rest.")
            concepts = concepts[:limit]
        return Material(concepts, "list", _title_from(concepts), None, raw, notes)
    if len(raw.split()) <= 8:
        return Material([], "topic", _clip(raw, 60), None, raw)
    return Material([], "prose", "Your notes", None, raw)


def sentences_as_concepts(text: str, limit: int) -> list[Concept]:
    """The fallback for prose without Claude: one sentence per station."""
    out = []
    for sentence in _SENTENCE.split(" ".join(text.split())):
        sentence = sentence.strip()
        if len(sentence.split()) < 3:
            continue
        words = sentence.split()
        out.append(Concept(_clip(" ".join(words[:6]).strip(",;:."), 80), _clip(sentence, 200)))
        if len(out) == limit:
            break
    return out


def _title_from(concepts: list[Concept]) -> str:
    if not concepts:
        return "Your list"
    names = ", ".join(c.term for c in concepts[:3])
    return _clip(f"{names}{' and more' if len(concepts) > 3 else ''}", 80)


def resume_material(material: Material) -> str:
    """Text that, given back as `material`, rebuilds exactly these concepts."""
    if material.preset:
        return material.preset if material.preset != "wives" else "wives of henry viii"
    if material.kind == "list":
        return material.raw
    return "; ".join(f"{c.term}: {c.detail}" if c.detail else c.term for c in material.concepts)


# ── Scenes ──────────────────────────────────────────────────────────────────

_ACTIONS = ("is juggling flaming torches", "is painted twenty feet high", "is doing star jumps",
            "is riding a penny-farthing in circles", "is bellowing through a megaphone",
            "is balanced on a stack of suitcases", "is tangled in streamers", "is glowing like a neon sign")
_SOUNDS = ("clanging like a fire bell", "humming the Tube's door chime", "roaring like a jet engine",
           "squeaking like wet trainers")
_SMELLS = ("smelling of burnt toast", "smelling of fresh paint", "reeking of vinegar", "smelling of oranges")
_TOUCH = ("ice-cold to touch", "sticky as syrup", "buzzing under your hand", "as heavy as an anvil")


def _pick(options: tuple[str, ...], *seed: str) -> str:
    return options[zlib.crc32("|".join(seed).encode("utf-8")) % len(options)]


def template_scene(concept: Concept, sid: str, variant: int = 0) -> str:
    """A station-specific scene from rules alone. Variant 1 is the stronger,
    multi-sensory rewrite used when a station was not recalled."""
    station = tube.display_name(sid)
    landmark = tube.anchor(sid)
    term = concept.term
    detail = _clip(concept.detail, 140)
    if variant == 0:
        action = _pick(_ACTIONS, term, sid)
        place = f"Right beside {landmark}" if landmark else f"Under the {station} roundel"
        tail = f" It is there to tell you: {detail}." if detail else f" Say it: {station} means {term}."
        return f"Step off at {station}. {place}, {term} {action}.{tail}"
    sound, smell, touch = _pick(_SOUNDS, term, sid, "s"), _pick(_SMELLS, term, sid, "m"), _pick(_TOUCH, term, sid, "t")
    place = landmark or f"the {station} platform"
    link = f"{station}, {term}" + (f": {detail}" if detail else "")
    return (f"Make it impossible to miss. At {station}, {place} bursts open and out marches {term}, "
            f"{sound}, {smell}, {touch}. Shake its hand and say aloud: \"{link}.\"")


def check_scene(text: str, sid: str, concept: Concept) -> str | None:
    """A generated scene if it is safe to show, else None.

    Refused: empty, far too long, or naming a station that is not this one
    (unless the concept itself names it). This is the guard that keeps a
    model from inventing a different route inside the imagery."""
    scene = " ".join((text or "").split())
    if not scene or len(scene.split()) > SCENE_WORD_LIMIT:
        return None
    allowed = tube.stations_mentioned(" ".join((tube.display_name(sid), tube.anchor(sid),
                                               concept.term, concept.detail)))
    if tube.stations_mentioned(scene) - allowed:
        return None
    station = tube.base_name(sid)
    if tube.norm(station) not in tube.norm(scene):
        scene = f"At {station}: {scene}"
    return scene


# ── The palace ──────────────────────────────────────────────────────────────


def build(material: Material, stops: list[tube.Stop], scenes: list[str], sources: list[str],
          route_inputs: dict) -> dict:
    """The palace as plain data: what `palace` holds in every output."""
    out_stops = []
    for i, (stop, concept) in enumerate(zip(stops, material.concepts)):
        nxt = stops[i + 1] if i + 1 < len(stops) else None
        out_stops.append({
            "n": i + 1,
            "station": tube.display_name(stop.sid),
            "station_id": stop.sid,
            "line": tube.line_name(stop.line),
            "line_id": stop.line,
            "zone": tube.zone(stop.sid),
            "landmark": tube.anchor(stop.sid),
            "concept": concept.term,
            "detail": concept.detail,
            "scene": scenes[i],
            "scene_source": sources[i],
            "minutes_from_previous": int(round(stop.minutes)),
            "passing": [tube.display_name(s) for s in stop.passing],
            "interchanges": tube.interchanges(stop.sid, stop.line),
            "change_to": tube.line_name(nxt.line) if nxt and nxt.line != stop.line else "",
        })
    line_names = []
    for s in out_stops:
        if s["line"] not in line_names:
            line_names.append(s["line"])
    return {
        "version": 1,
        "title": material.title,
        "material": resume_material(material),
        "route": route_inputs,
        "lines": line_names,
        "minutes": sum(s["minutes_from_previous"] for s in out_stops),
        "stops": out_stops,
    }


def from_input(value) -> dict | None:
    """A palace handed back by a caller, checked against the network.

    Station names and lines are never taken from the caller: they are
    rebuilt from each `station_id`. Anything that does not check out makes
    the whole palace unusable, and the caller's material is used instead."""
    if not isinstance(value, dict) or not isinstance(value.get("stops"), list):
        return None
    stops = value["stops"]
    if not MIN_CONCEPTS <= len(stops) <= MAX_STOPS:
        return None
    clean = []
    prev_sid = None
    for i, stop in enumerate(stops):
        if not isinstance(stop, dict):
            return None
        sid = stop.get("station_id")
        concept = stop.get("concept")
        if sid not in tube.STATIONS or not isinstance(concept, str) or not concept.strip():
            return None
        line_id = stop.get("line_id")
        if line_id not in tube.LINES or line_id not in tube.lines_at(sid):
            line_id = tube.lines_at(sid)[0]
        scene = stop.get("scene") if isinstance(stop.get("scene"), str) else ""
        concept_obj = Concept(_clip(concept, 80), _clip(str(stop.get("detail") or ""), 200))
        scene = check_scene(_clip(scene, 600), sid, concept_obj) or template_scene(concept_obj, sid)
        clean.append({
            "n": i + 1, "station": tube.display_name(sid), "station_id": sid,
            "line": tube.line_name(line_id), "line_id": line_id, "zone": tube.zone(sid),
            "landmark": tube.anchor(sid), "concept": concept_obj.term, "detail": concept_obj.detail,
            "scene": scene, "scene_source": "given",
            "minutes_from_previous": int(round(tube.hop_minutes(prev_sid, sid))) if prev_sid else 0,
            "passing": [], "interchanges": tube.interchanges(sid, line_id), "change_to": "",
        })
        prev_sid = sid
    for cur, nxt in zip(clean, clean[1:]):
        if nxt["line_id"] != cur["line_id"]:
            cur["change_to"] = nxt["line"]
    title = value.get("title") if isinstance(value.get("title"), str) else "Your memory palace"
    lines = []
    for s in clean:
        if s["line"] not in lines:
            lines.append(s["line"])
    return {"version": 1, "title": _clip(title, 80), "material": "", "route": {}, "lines": lines,
            "minutes": sum(s["minutes_from_previous"] for s in clean), "stops": clean}


def concepts_of(palace: dict) -> list[Concept]:
    known = {}
    for preset in PRESETS.values():
        for c in preset["concepts"]:
            known[c.term] = c
    return [known.get(s["concept"]) or Concept(s["concept"], s["detail"]) for s in palace["stops"]]


def route_summary(palace: dict) -> str:
    stops = palace["stops"]
    lines = " then ".join(palace["lines"])
    return (f"{lines} line{'s' if len(palace['lines']) > 1 else ''}: {stops[0]['station']} to "
            f"{stops[-1]['station']}, {len(stops)} stops, about {palace['minutes']} minutes")


def narrative(palace: dict) -> str:
    stops = palace["stops"]
    parts = [f"Board the {stops[0]['line']} line at {stops[0]['station']}, where {stops[0]['concept']} waits."]
    for prev, s in zip(stops, stops[1:]):
        if prev["change_to"]:
            parts.append(f"Change to the {prev['change_to']} line.")
        parts.append(f"{s['station']}: {s['concept']}.")
    parts.append(f"Alight at {stops[-1]['station']}. Mind the gap.")
    return " ".join(parts)


# ── The journey ─────────────────────────────────────────────────────────────

_ACTION_WORDS = (
    ("previous", ("previous", "prev", "back", "before", "last one", "go back", "earlier")),
    ("repeat", ("repeat", "again", "where am i", "this station", "current", "same", "stay", "once more")),
    ("skip", ("skip", "pass", "miss out", "jump")),
    ("start", ("start", "begin", "first", "restart", "from the top", "beginning")),
    ("next", ("next", "forward", "continue", "go on", "onward", "on", "carry on", "advance")),
)


def action_of(text: str | None) -> str:
    t = " " + tube.norm(text or "") + " "
    for action, words in _ACTION_WORDS:
        if any(f" {w} " in t for w in words):
            return action
    return ""


def find_stop(palace: dict, station_text: str) -> int | None:
    """1-based position of a station on this palace, or None."""
    q = tube.norm(station_text)
    if not q:
        return None
    if q.isdigit():
        n = int(q)
        return n if 1 <= n <= len(palace["stops"]) else None
    for s in palace["stops"]:
        name = tube.norm(s["station"])
        if q == name or name.startswith(q + " ") or q.startswith(name):
            return s["n"]
    try:
        sid = tube.find_station(station_text)
    except tube.StationNotFound:
        return None
    return next((s["n"] for s in palace["stops"] if s["station_id"] == sid), None)


def step(palace: dict, position: int | None, action: str) -> dict:
    """Where the journey goes next, and what the announcer says."""
    stops = palace["stops"]
    total = len(stops)
    current = position if position and 1 <= position <= total else None
    if not action:
        action = "next" if current else "start"
    if action == "start":
        target = 1
    elif action == "previous":
        target = max(1, (current or 1) - 1)
    elif action == "repeat":
        target = current or 1
    elif action == "skip":
        target = (current or 1) + 1 if current else 2
    else:
        target = (current or 0) + 1
    beyond = target > total
    target = min(target, total)
    s = stops[target - 1]
    prev = stops[target - 2] if target > 1 else None
    nxt = stops[target] if target < total else None

    said = []
    if beyond:
        said.append(f"That was the last stop. This is still {s['station']}.")
    elif action == "start" or target == 1 and not current:
        said.append(f"Welcome aboard this {s['line']} line memory palace, {palace['title']}: "
                    f"{total} stops from {stops[0]['station']} to {stops[-1]['station']}, "
                    f"about {palace['minutes']} minutes. This is {s['station']}.")
    elif action == "previous":
        said.append(f"Going back one stop. This is {s['station']}.")
    elif action == "repeat":
        said.append(f"Once more: this is {s['station']}.")
    elif action == "skip" and current:
        said.append(f"Skipping {stops[current - 1]['station'] if current < total else ''}… "
                    f"this is {s['station']}.".replace("Skipping … ", ""))
    else:
        said.append(f"This is {s['station']}.")
    if s["interchanges"] and not beyond:
        said.append(f"Change here for the {_and(s['interchanges'])} line{'s' if len(s['interchanges']) > 1 else ''}.")
    if nxt:
        if s["change_to"]:
            said.append(f"Change here to the {s['change_to']} line for {nxt['station']}.")
        said.append(f"The next station is {nxt['station']}.")
    else:
        said.append("This train terminates here. All change, please, and mind the gap. "
                    "When you are ready, ask me to quiz you on this memory palace.")
    return {
        "position": target,
        "total": total,
        "stop": s,
        "previous": prev,
        "next": nxt,
        "at_end": nxt is None,
        "announcement": " ".join(said),
        "action": action,
    }


def _and(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


# ── Grading recall ──────────────────────────────────────────────────────────

_FILLER = {"a", "an", "the", "at", "was", "is", "it", "i", "think", "then", "there", "station",
           "stop", "had", "we", "saw", "found", "and", "of", "to", "in", "on", "with", "for",
           "next", "first", "after", "that", "this", "s", "my", "answer", "remembered", "recalled"}
_BLANK = {"", "dont know", "i dont know", "no idea", "pass", "blank", "nothing", "unknown", "forgot",
          "not sure", "cant remember", "idk", "skip"}


def _tokens(text: str) -> list[str]:
    return [w for w in tube.norm(text).split() if w not in _FILLER]


def _same_word(a: str, b: str) -> bool:
    if a == b:
        return True
    if len(a) >= 4 and len(b) >= 4 and difflib.SequenceMatcher(None, a, b).ratio() >= 0.84:
        return True
    return len(a) >= 5 and len(b) >= 5 and a[:5] == b[:5]


def match_score(answer: str, concept: Concept) -> float:
    """How well an answer names a concept, from 0 to 1."""
    a = tube.norm(answer)
    if not a or a in _BLANK:
        return 0.0
    padded = f" {a} "
    for name in (concept.term, *concept.aliases):
        n = tube.norm(name)
        if n and f" {n} " in padded:
            return 1.0
    term = tube.norm(concept.term)
    t_tokens = _tokens(concept.term) or term.split()
    a_tokens = _tokens(answer) or a.split()
    hit = sum(1 for t in t_tokens if any(_same_word(t, w) for w in a_tokens))
    token_score = hit / len(t_tokens) if t_tokens else 0.0
    ratio = difflib.SequenceMatcher(None, a, term).ratio() if len(a) <= 2 * len(term) + 6 else 0.0
    d_tokens = [t for t in _tokens(concept.detail) if len(t) > 3][:12]
    d_hit = sum(1 for t in d_tokens if any(_same_word(t, w) for w in a_tokens))
    detail_score = min(1.0, d_hit / 3) * 0.6 if d_tokens else 0.0
    return round(min(1.0, max(token_score, ratio, detail_score)), 3)


CORRECT, PARTIAL = 0.8, 0.45


def split_recall(palace: dict, recall: str) -> list[str | None]:
    """One answer per stop, from free text.

    If the text names the stations, each answer is the text beside its
    station ("brixton prior, stockwell likelihood" or "prior at brixton,
    likelihood at stockwell"). Otherwise the answers are taken in order."""
    stops = palace["stops"]
    text = " " + tube.norm(recall.replace("\n", " , ")) + " "
    found: list[tuple[int, int, int]] = []   # (start, end, stop index)
    for i, s in enumerate(stops):
        names = {tube.norm(s["station"]), tube.norm(re.sub(r"\s*\(.*\)$", "", s["station"]))}
        first = tube.norm(s["station"]).split()[0]
        if len(first) >= 5 and sum(1 for o in stops if tube.norm(o["station"]).split()[0] == first) == 1:
            names.add(first)
        best = None
        for name in sorted(names, key=len, reverse=True):
            m = re.search(rf" {re.escape(name)} ", text)
            if m and (best is None or m.start() < best[0]):
                best = (m.start(), m.end())
        if best:
            found.append((best[0], best[1], i))
    answers: list[str | None] = [None] * len(stops)
    if found:
        found.sort()
        lead = text[:found[0][0]].strip()
        after = not lead or len(lead.split()) <= 2     # "brixton: prior, ..."  vs  "prior at brixton, ..."
        for k, (start, end, i) in enumerate(found):
            if after:
                stop_at = found[k + 1][0] if k + 1 < len(found) else len(text)
                answers[i] = text[end:stop_at].strip()
            else:
                begin = found[k - 1][1] if k > 0 else 0
                answers[i] = text[begin:start].strip()
        return [a if a and _tokens(a) else None for a in answers]
    parts = [p.strip() for p in re.split(r"[\n;,]+|\bthen\b|->|→|\bnext\b", recall) if p.strip()]
    if len(parts) == 1 and len(stops) > 1:
        parts = [p for p in re.split(r"\s+(?:and|&)\s+", parts[0]) if p.strip()]
    for i in range(min(len(parts), len(stops))):
        cleaned = _BULLET.sub("", parts[i]).strip()
        answers[i] = cleaned if tube.norm(cleaned) not in _BLANK else None
    return answers


def grade(palace: dict, recall: str) -> dict:
    """Deterministic marking of a recall attempt."""
    concepts = concepts_of(palace)
    answers = split_recall(palace, recall)
    items, sequence = [], []
    for i, (s, concept, answer) in enumerate(zip(palace["stops"], concepts, answers)):
        if not answer:
            items.append({"n": i + 1, "station": s["station"], "expected": concept.term, "answer": "",
                          "status": "missing", "score": 0.0})
            continue
        score = match_score(answer, concept)
        status = "correct" if score >= CORRECT else "partial" if score >= PARTIAL else "wrong"
        belongs = ""
        if status != "correct":
            others = [(match_score(answer, c), j) for j, c in enumerate(concepts) if j != i]
            top = max(others) if others else (0.0, -1)
            if top[0] >= CORRECT:
                status, belongs = "misplaced", palace["stops"][top[1]]["station"]
                sequence.append(top[1])
        else:
            sequence.append(i)
        items.append({"n": i + 1, "station": s["station"], "expected": concept.term,
                      "answer": _clip(answer, 80), "status": status, "score": score,
                      "belongs_at": belongs})
    return {"items": items, "sequence": sequence}


def score_of(items: list[dict]) -> float:
    if not items:
        return 0.0
    points = sum(1.0 if i["status"] == "correct" else 0.5 if i["status"] in ("partial", "misplaced") else 0.0
                 for i in items)
    return round(points / len(items), 2)


def order_correct(graded: dict) -> bool:
    seq = graded["sequence"]
    return bool(seq) and seq == sorted(seq) and len(set(seq)) == len(seq) and \
        all(i["status"] == "correct" for i in graded["items"])


def result_lines(items: list[dict]) -> list[str]:
    words = {"correct": "correct", "partial": "nearly", "wrong": "not quite",
             "missing": "missed", "misplaced": "right idea, wrong station"}
    out = []
    for i in items:
        line = f"{i['n']}. {i['station']}: {i['expected']}, {words[i['status']]}"
        if i["answer"] and i["status"] != "correct":
            line += f" (you said \"{i['answer']}\""
            line += f", which lives at {i['belongs_at']})" if i.get("belongs_at") else ")"
        out.append(line)
    return out


def questions(palace: dict) -> list[str]:
    stops = palace["stops"]
    out = [f"Stop {s['n']}, {s['station']}: what is waiting for you there?" for s in stops]
    out.append(f"Bonus: name all {len(stops)} in order, from {stops[0]['station']} to {stops[-1]['station']}.")
    return out


# ── Adapting ────────────────────────────────────────────────────────────────


def weak_from_list(palace: dict, struggled: list[str]) -> tuple[set[int], list[str]]:
    """Stop numbers named in a list of stations, concepts or numbers."""
    concepts = concepts_of(palace)
    found, unknown = set(), []
    for raw in struggled:
        if not isinstance(raw, str) or not raw.strip():
            continue
        text = re.sub(r"^\s*(stop|station|number)\s+", "", raw.strip(), flags=re.IGNORECASE)
        n = find_stop(palace, text)
        if n is None:
            scores = [(match_score(text, c), i + 1) for i, c in enumerate(concepts)]
            top = max(scores)
            n = top[1] if top[0] >= CORRECT else None
        if n is None:
            unknown.append(_clip(raw, 40))
        else:
            found.add(n)
    return found, unknown


def weak_from_results(palace: dict, results) -> set[int]:
    """Stop numbers that a previous `palace.test` marked as not correct."""
    found = set()
    if not isinstance(results, dict):
        return found
    for item in results.get("stations", []) if isinstance(results.get("stations"), list) else []:
        if isinstance(item, dict) and item.get("status") != "correct":
            n = find_stop(palace, str(item.get("station", "")))
            if n:
                found.add(n)
    return found


def schedule(now: datetime, score: float, review_count: int) -> dict:
    """Expanding intervals: 1, 3, 7, 14, then 30 days; 20 minutes after a
    poor score, without counting it as a review."""
    count = max(0, review_count)
    if score < PASS_MARK:
        delta, duration, next_count = timedelta(minutes=RETRY_MINUTES), f"PT{RETRY_MINUTES}M", count
        words = f"{RETRY_MINUTES} minutes"
    else:
        days = REVIEW_DAYS[min(count, len(REVIEW_DAYS) - 1)]
        delta, duration, next_count = timedelta(days=days), f"P{days}D", count + 1
        words = "1 day" if days == 1 else f"{days} days"
    return {"at": (now + delta).isoformat(timespec="seconds"), "duration": duration,
            "next_count": next_count, "words": words, "when": now + delta}


def review_plan(palace: dict, weak: list[int], plan: dict, score: float) -> list[str]:
    stops = palace["stops"]
    out = []
    if weak:
        names = _and([stops[n - 1]["station"] for n in weak])
        out.append(f"Now: walk {names} once more, holding each new scene in mind for five seconds.")
    else:
        out.append("Now: nothing to fix. Every station held.")
    if score < PASS_MARK:
        out.append(f"In {plan['words']}: walk the whole route again, then ask me to quiz you. "
                   f"Under {int(PASS_MARK * 100)}% correct means practise again soon before spacing it out.")
    else:
        out.append(f"In {plan['words']} ({plan['when'].strftime('%a %d %b, %H:%M')}): ask me to quiz you on "
                   f"this memory palace, without looking first.")
    out.append("Each time every station holds, the gap grows: 1, 3, 7, 14, then 30 days.")
    return out
