"""Tests for Mind the Gap, through LocalRunner so the manifest is enforced.

The network is never used: every test replaces `App._fetch` on the
instance. Claude is stubbed with canned Messages API replies.
"""

from __future__ import annotations

import json
import urllib.error
from datetime import datetime, timezone

import pytest
from pippa_agentapp_sdk import LocalRunner, OutputShapeViolation

import tube
from app import App, NoMaterial, NothingToAdapt, UnknownLine, UnknownStation

NOW = datetime(2026, 11, 20, 18, 0, tzinfo=timezone.utc)
FAKE_KEY = "test-key-not-real"  # pragma: allowlist secret
BAYES_ROUTE = ["Brixton", "Stockwell", "Vauxhall", "Pimlico", "Victoria"]
CELLS = "mitochondria: powerhouse of the cell\nribosome: makes proteins\nnucleus: holds the DNA\ngolgi: packages proteins"


def offline(url, timeout=6.0, data=None, headers=None, key_header=None):
    raise urllib.error.URLError("offline in tests")


def tfl_ok(url, timeout=6.0, data=None, headers=None, key_header=None):
    assert url.startswith("https://api.tfl.gov.uk/Line/")
    return json.dumps([{"id": "victoria", "lineStatuses": [{"statusSeverityDescription": "Good Service"}]}]).encode()


def claude_reply(payload: dict) -> bytes:
    return json.dumps({"stop_reason": "end_turn",
                       "content": [{"type": "text", "text": json.dumps(payload)}]}).encode()


def run(inputs: dict, outcome: str = "palace.create", fetch=offline, explicit: bool = False):
    app = App()
    app._fetch = fetch
    return LocalRunner(app).call(outcome, inputs, granted_secrets={}, now=NOW, request_id="t1",
                                 explicit=explicit).result


@pytest.fixture(autouse=True)
def no_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def test_manifest_is_valid():
    assert App().manifest is not None


# ── CONSTRUCT ───────────────────────────────────────────────────────────────


def test_bayes_demo_uses_the_real_victoria_line_in_order():
    r = run({"material": "bayes"}, fetch=tfl_ok)
    stops = r.data["palace"]["stops"]
    assert [s["station"] for s in stops] == BAYES_ROUTE
    assert [s["concept"] for s in stops] == ["Prior", "Likelihood", "Evidence", "Normalisation", "Posterior"]
    assert r.data["imagery_source"] == "curated"
    assert r.data["live_status"] == ["Victoria: Good Service"]
    assert "live line status" in r.data["route_source"]
    assert "0.0585" in stops[2]["scene"]


def test_every_station_is_real_and_adjacent_on_its_line():
    r = run({"material": CELLS, "line": "central"})
    stops = r.data["palace"]["stops"]
    assert len(stops) == 4
    for a, b in zip(stops, stops[1:]):
        assert a["station_id"] in tube.STATIONS and b["station_id"] in tube.STATIONS
        assert b["station_id"] in tube._ADJ[(a["station_id"], "central")]


def test_list_without_key_falls_back_to_templates_with_a_warning():
    r = run({"material": CELLS})
    assert r.data["imagery_source"] == "template"
    assert any("template" in w for w in r.warnings)
    assert any("Live TfL" in n for n in r.data["notes"])


def test_named_route_between_stations():
    r = run({"material": CELLS, "from_station": "Oxford Circus", "to_station": "Victoria"})
    stations = [s["station"] for s in r.data["palace"]["stops"]]
    assert stations[0] == "Oxford Circus"
    assert len(stations) == 4


def test_the_same_inputs_build_the_same_palace():
    assert run({"material": CELLS}).data["palace"] == run({"material": CELLS}).data["palace"]


def test_no_material_says_what_to_type():
    with pytest.raises(NoMaterial):
        run({})


def test_unknown_station_is_named():
    with pytest.raises(UnknownStation) as info:
        run({"material": CELLS, "from_station": "Narnia"})
    assert "Narnia" in info.value.user_message


def test_elizabeth_line_is_not_tube():
    with pytest.raises(UnknownLine) as info:
        run({"material": CELLS, "line": "Elizabeth line"})
    assert "isn't part of the Underground" in info.value.user_message


def test_claude_scenes_are_used_and_invented_stations_rejected(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    seen = {}

    def fetch(url, timeout=6.0, data=None, headers=None, key_header=None):
        if "tfl" in url:
            return tfl_ok(url)
        assert url == "https://api.anthropic.com/v1/messages"
        assert key_header == ("x-api-key", FAKE_KEY)
        assert FAKE_KEY not in data.decode()
        seen["body"] = json.loads(data)
        return claude_reply({"scenes": [
            {"index": 0, "scene": "A giant MITOCHONDRIA battery glows at the station."},
            {"index": 1, "scene": "A RIBOSOME waves from Baker Street."},   # another station: rejected
            {"index": 2, "scene": "The NUCLEUS sits in a vault of DNA."},
            {"index": 3, "scene": "GOLGI wraps parcels with ribbon."},
        ]})

    r = run({"material": CELLS}, fetch=fetch)
    stops = r.data["palace"]["stops"]
    assert [s["scene_source"] for s in stops] == ["claude", "template", "claude", "claude"]
    assert "Baker Street" not in stops[1]["scene"]
    assert seen["body"]["model"] == "claude-opus-5-5"
    assert any("station check" in w for w in r.warnings)


def test_claude_rate_limit_falls_back(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)

    def fetch(url, timeout=6.0, data=None, headers=None, key_header=None):
        raise urllib.error.HTTPError(url, 429, "busy", {}, None)

    r = run({"material": CELLS}, fetch=fetch)
    assert r.data["imagery_source"] == "template"
    assert any("busy" in w for w in r.warnings)
    assert FAKE_KEY not in json.dumps(r.data)


def test_claude_timeout_falls_back(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)

    def fetch(url, timeout=6.0, data=None, headers=None, key_header=None):
        raise TimeoutError("slow")

    assert run({"material": CELLS}, fetch=fetch).data["imagery_source"] == "template"


def test_prose_without_key_becomes_one_stop_per_sentence():
    text = ("Photosynthesis turns light into chemical energy. It happens in the chloroplasts of plant cells. "
            "Carbon dioxide and water go in. Glucose and oxygen come out.")
    r = run({"material": text})
    assert len(r.data["stations"]) == 4


# ── EXPERIENCE ──────────────────────────────────────────────────────────────


def test_journey_starts_then_moves_next_back_and_ends():
    first = run({"material": "bayes"}, "palace.journey")
    assert first.data["position"] == 1 and first.data["station"] == "Brixton"
    assert "Welcome aboard" in first.data["announcement"]
    nxt = run({"material": "bayes", "position": 1, "action": "next"}, "palace.journey")
    assert nxt.data["station"] == "Stockwell" and nxt.data["previous_station"] == "Brixton"
    back = run({"material": "bayes", "current_station": "Vauxhall", "action": "previous"}, "palace.journey")
    assert back.data["station"] == "Stockwell"
    skip = run({"material": "bayes", "position": 2, "action": "skip"}, "palace.journey")
    assert skip.data["position"] == 3
    end = run({"material": "bayes", "position": 4, "action": "next"}, "palace.journey")
    assert end.data["at_end"] is True and "mind the gap" in end.data["announcement"]


def test_journey_picked_by_name_still_answers():
    r = run({"material": "bayes", "action": "repeat", "position": 3}, "palace.journey", explicit=True)
    assert r.data["station"] == "Vauxhall"


def test_station_not_on_palace_lists_the_stops():
    with pytest.raises(UnknownStation) as info:
        run({"material": "bayes", "current_station": "Morden"}, "palace.journey")
    assert "Brixton" in info.value.user_message


# ── RETRIEVE ────────────────────────────────────────────────────────────────


def test_quiz_mode_asks_questions():
    r = run({"material": "bayes"}, "palace.test")
    assert r.data["mode"] == "quiz" and len(r.data["questions"]) == 6


def test_perfect_recall_in_order():
    r = run({"material": "bayes", "recall": "prior, likelihood, evidence, normalization, posterior"}, "palace.test")
    assert r.data["score"] == 1.0 and r.data["order_correct"] is True and r.data["weak_stations"] == []


def test_swapped_answers_are_misplaced():
    recall = "brixton prior, stockwell likelihood, vauxhall evidence, pimlico posterior, victoria normalisation"
    r = run({"material": "bayes", "recall": recall}, "palace.test")
    assert r.data["weak_stations"] == ["Pimlico", "Victoria"]
    assert r.data["order_correct"] is False
    assert r.data["score"] == 0.8


def test_missing_answers_count_as_missed():
    r = run({"material": "bayes", "recall": "prior, likelihood"}, "palace.test")
    assert r.data["correct"] == 2 and len(r.data["weak_stations"]) == 3


# ── ADAPT ───────────────────────────────────────────────────────────────────


def test_adapt_strengthens_only_the_weak_stations():
    before = run({"material": "bayes"}).data["palace"]["stops"]
    r = run({"material": "bayes", "struggled_with": ["Pimlico", "Vauxhall"]}, "palace.adapt")
    after = r.data["palace"]["stops"]
    assert r.data["weak_stations"] == ["Vauxhall", "Pimlico"]
    for b, a in zip(before, after):
        if b["station"] in ("Vauxhall", "Pimlico"):
            assert a["scene"] != b["scene"]
        else:
            assert a["scene"] == b["scene"]
    assert r.data["review_interval"] == "P1D"
    assert r.data["next_review_at"].startswith("2026-11-21T18:00")


def test_poor_score_reviews_again_in_twenty_minutes():
    r = run({"material": "bayes", "recall": "prior"}, "palace.adapt")
    assert r.data["review_interval"] == "PT20M" and r.data["review_count"] == 0


def test_spacing_grows_with_reviews():
    r = run({"material": "bayes", "recall": "prior, likelihood, evidence, normalisation, posterior",
             "review_count": 2}, "palace.adapt")
    assert r.data["review_interval"] == "P7D" and r.data["review_count"] == 3


def test_adapt_needs_something_to_work_on():
    with pytest.raises(NothingToAdapt):
        run({"material": "bayes"}, "palace.adapt")


def test_a_passed_back_palace_is_rebuilt_from_station_ids():
    palace = run({"material": CELLS}).data["palace"]
    palace["stops"][0]["station"] = "Hogwarts"
    r = run({"palace": palace, "position": 1, "action": "repeat"}, "palace.journey")
    assert r.data["station"] != "Hogwarts"


def test_wrong_output_shape_is_refused(monkeypatch):
    from pippa_agentapp_sdk import Result
    monkeypatch.setattr(App, "handle", lambda self, call: Result(data={"title": "x"}))
    with pytest.raises(OutputShapeViolation):
        run({"material": "bayes"})
