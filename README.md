# Mind the Gap 🚇🧠

**A memory palace built on real London Underground routes.** Give Pippa
something to learn and Mind the Gap puts one idea at each station of a real
Tube line. You get a vivid scene at every stop, a stop-by-stop journey, a
quiz, and stronger images for whatever you forgot, with a spaced-review
schedule.

> *"@Mind the Gap build a memory palace for bayes theorem"*
> → Victoria line, Brixton to Victoria, 5 stops, about 7 minutes:
> **Prior** at Brixton, **Likelihood** at Stockwell, **Evidence** at
> Vauxhall, **Normalisation** at Pimlico, **Posterior** at Victoria.

It follows the five stages of the method of loci:

| Stage | Outcome | What happens |
|---|---|---|
| UNDERSTAND | `palace.create` | Reads the material: a preset, a list, or prose (Claude picks out the concepts) |
| CONSTRUCT | `palace.create` | Picks a real route and gives each concept a station and a scene |
| EXPERIENCE | `palace.journey` | Walks the route one stop at a time: next, previous, repeat, skip |
| RETRIEVE | `palace.test` | Quiz mode, or marks what you recalled at each station |
| ADAPT | `palace.adapt` | Writes stronger scenes for weak stations and schedules the next review |

## Spec tools → Pippa outcomes

Pippa apps are not MCP servers, and they do not register custom function
definitions. An app is a `manifest.yaml` of **outcomes** with typed inputs
and outputs, plus `App.handle(call)` in `app.py` (see
[docs/HOW_PIPPA_APPS_WORK.md](docs/HOW_PIPPA_APPS_WORK.md)). So the four tools
from the spec are four outcomes. Pippa chooses between them from their
example phrases:

| Spec tool | Outcome |
|---|---|
| `create_memory_palace` | `palace.create` |
| `get_memory_palace_journey` | `palace.journey` |
| `test_memory_palace` | `palace.test` |
| `adapt_memory_palace` | `palace.adapt` |

## Who does what: deterministic code vs. the LLM

| Deterministic, in code (`tube.py`, `palace.py`) | Claude, optional (`llm.py`) |
|---|---|
| Every station, line, route and travel time | Picking concepts out of a paragraph of notes |
| Routing (Dijkstra over station × line, 5 min per change) | Writing a vivid scene for each stop |
| The order of stops, the journey, announcements | A second opinion on answers the rules mark as close |
| Marking recall (aliases, fuzzy match, misplaced answers) | |
| Spaced-review schedule (1 → 3 → 7 → 14 → 30 days) | |

**The LLM never chooses a station.** Stations come only from
`tube_network.json`, a snapshot of the TfL Unified API: 11 lines, 272
stations and 313 hops, with times taken from TfL's timetables. Every scene
Claude writes is checked, and a scene that names any station other than its
own is thrown away and replaced with a template. Claude gets no tools. Its
replies are JSON constrained to a schema, and the notes are passed to it as
data, not as instructions.

**Without a key, everything still works.** Lists and presets need no LLM.
Imagery falls back to deterministic templates, keyed on a hash of the concept
and the station, so the same input always gives the same scene. Prose falls
back to one sentence per stop. Each fallback adds a warning that says what
happened.

## Inputs (shared by all outcomes, all optional)

| Input | Type | Meaning |
|---|---|---|
| `material` | string | What to memorise: a list (lines, commas, `;`, numbers), `term: detail` pairs, prose, or a preset (`bayes`, `planets`, `wives of henry viii`) |
| `line` | string | Tube line, e.g. `Victoria`. The Elizabeth line, DLR and Overground are refused politely |
| `from_station` / `to_station` | string | Route ends. Fuzzy matched (`st pauls`, `kings cross`) |
| `max_stations` | integer | 2–12 (default 10) |
| `style` | string | Imagery style: `funny`, `spooky`, `for a ten year old` |
| `position` / `current_station` / `action` | int / string / string | Where you are and what to do next on the journey |
| `recall` | string | What you remembered: `brixton prior, stockwell likelihood…` or just `prior, likelihood, …` |
| `struggled_with` | array of string | Stations or concepts you got wrong |
| `review_count` | integer | Successful reviews so far |
| `palace` / `retrieval_results` | object | Earlier outputs, passed back if the caller kept them |

### Why every call takes the material again

Pippa apps are **stateless**: they get no conversation, no user id and no
storage. So every call rebuilds the palace from the material and route
inputs, deterministically, and the same words give the same palace,
station for station. `palace.create` returns `resume_inputs`, which are
exactly what rebuilds it. A caller that kept the `palace` object can pass it
back instead. Its station names are never trusted: each stop is rebuilt from
its `station_id`, which is checked against the network.

## Outputs

**`palace.create`**: `title`, `route_summary`, `route_source`,
`imagery_source` (`curated` / `template` / `claude`), `stations` (`"1.
Brixton (Victoria): Prior"`), `story`, `palace` (every stop with station,
line, zone, landmark, concept, scene, minutes, stations passed, interchanges),
`resume_hint`, `resume_inputs`, `notes`, and `live_status` (live TfL line
status, when reachable).

**`palace.journey`**: `position`, `total`, `station`, `line`, `concept`,
`scene`, `announcement` (*"This is Stockwell. Change here for the Northern
line. The next station is Vauxhall."*), `at_end`, `action`,
`route_summary`, plus `next_station`, `previous_station` and `passing`.

**`palace.test`**: `mode` (`quiz` / `graded`), `total`, `questions`,
`results`, `weak_stations`, `grading_method`, `route_summary`, plus `score`,
`correct`, `order_correct` and `retrieval_results`.

**`palace.adapt`**: `weak_stations`, `strengthened` (new multi-sensory
scenes), `unchanged` (stations you recalled, whose scenes are kept),
`score`, `next_review_at` (datetime), `review_interval` (duration, e.g.
`P3D` or `PT20M`), `review_count`, `review_plan`, `imagery_source`, `palace`.

## Example conversation

```
you    @Mind the Gap build a memory palace for bayes theorem
pippa  Victoria line, Brixton → Victoria, 5 stops. Stop 1, Brixton: on Electric
       Avenue a busker under the Bowie mural sings your PRIOR on a loop…
you    @Mind the Gap next stop on my bayes palace, I'm at stop 1
pippa  This is Stockwell. Change here for the Northern line. The next station
       is Vauxhall. Deep in Stockwell's wartime shelter a warden…
you    @Mind the Gap quiz me on my bayes palace
you    @Mind the Gap here is what i recalled for bayes: brixton prior, stockwell
       likelihood, vauxhall evidence, pimlico posterior, victoria normalisation
pippa  3 of 5 stations held (80%). Weak: Pimlico, Victoria (right idea, wrong
       station).
you    @Mind the Gap bayes: i struggled with pimlico and victoria
pippa  Strengthened 2 stops. Review again in 1 day.
```

## The Bayes demo

The preset runs along the real Victoria line, Brixton → Victoria, using a
diagnostic test: 1% of people have the condition, the test catches 90% of
them, and 5% of healthy people test positive anyway.

| Stop | Concept | Number |
|---|---|---|
| Brixton | **Prior** P(H) | 0.01 |
| Stockwell | **Likelihood** P(E\|H) | 0.90 |
| Vauxhall | **Evidence** P(E) = 0.9×0.01 + 0.05×0.99 | 0.0585 |
| Pimlico | **Normalisation**: divide by P(E) so the posteriors sum to 1 | ÷ 0.0585 |
| Victoria | **Posterior** P(H\|E) = 0.009 / 0.0585 | ≈ 0.154 |

The curated scenes use each station's real landmark: Electric Avenue and the
Bowie mural, the wartime deep shelter, MI6 at Vauxhall Cross, Tate Britain,
and Queen Victoria herself.

## Setting up the Claude key (optional)

In Pippa: **Apps → Created by me → Mind the Gap → Keys**, then add
`ANTHROPIC_API_KEY`. Never put it in any file in this repository. The key is
sent only to `https://api.anthropic.com/v1/messages`, in an unredirected
header, and redirects are refused. The model is `claude-opus-5-5`, called at
low effort with structured JSON output. Imagery runs in parallel batches of 4
stops, inside an 18-second budget.

## Files

| File | What it is |
|---|---|
| `app.py` | `App`: the four outcomes, the network seam, Claude and live-status calls |
| `palace.py` | Material parsing, presets, templates, scene checks, journey, grading, schedule |
| `tube.py` | The network: lookups, routing, route fitting, station-mention detection |
| `llm.py` | The Claude Messages API over `urllib`, failing safe to fixed reasons |
| `tube_network.json` | TfL snapshot (54 KB) |
| `test_app.py` | Tests through `LocalRunner`, with no network |

## Limits, honestly

- **Routes are from a snapshot** taken on 2026-11-20. Live TfL line status is
  fetched on `create` (2.5 s timeout, best effort), but routing does not
  react to closures.
- **Journey and test calls use template scenes** unless the caller passes the
  `palace` back. Without state, the Claude scenes from `create` cannot be
  remembered, and regenerating them every call would make them change.
- **Hop times**: 3 of 313 come from straight-line distance, because TfL's
  timetables did not cover them.
- Station landmarks are short, well-known facts written for this app. They
  are not TfL data.

## What we'd have liked to build (some of it impossible today)

- **A real ride-along mode.** Scenes timed to your actual train using TfL
  arrival predictions, so the scene for Vauxhall arrives just as the doors
  open at Vauxhall.
- **Sound.** Station announcements in the Tube's own voice, and an audio
  palace for commuting with no screen.
- **Memory across sessions.** A palace that remembers what you forgot, and
  real spaced repetition that nudges you on day 3. Pippa apps are stateless
  by design, so this would need a Pippa-side store.
- **Every rail network.** The Elizabeth line, the Overground, the DLR, and
  then the Paris Métro, the New York subway and Tokyo. Same code, different
  snapshot.
- **Your own commute as the palace.** Learn on the route you already ride
  every day.
- **Generated station art**, an image per scene, and AR overlays on the
  platform itself.
- **Bayesian mastery tracking.** A Beta prior on each station's recall
  probability, updated after each quiz, scheduling the next review where the
  posterior drops below 90%. It would be the Bayes demo, applied to itself.
- **Multiplayer palaces.** A class shares one route, and each person adds
  scenes.
- **Personalised landmarks.** "The café where we met" at Angel. That needs
  memory, and consent.

---

Powered by TfL Open Data. Contains OS data © Crown copyright and database
rights. Built on Pippa, with Claude.
