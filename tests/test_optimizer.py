"""Offline tests: the loop runs end to end against a fake writer and a fake scorer."""

import json

import pytest
from PIL import Image

import optimizer as o


def test_scale_round_trips():
    for raw in (0.3, 0.45, 0.508, 0.6):
        assert abs(o.from_pct(o.to_pct(raw)) - raw) < 1e-9
    assert o.to_pct(-5) == 0 and o.to_pct(5) == 100


def test_clean_strips_openers_dupes_and_own_brand(monkeypatch):
    monkeypatch.setattr(o.config, "OWN_BRANDS", ("acme florals",))
    out = o.clean([
        "The image shows pink tulips in a vase",
        "pink tulips in a vase",
        "Acme Florals pink tulips",
        "  An image of  a  gift box ",
    ])
    assert out == ["Pink tulips in a vase", "A gift box"]


class FakeScorer:
    name, model_id = "fake", "fake"

    def score(self, path, texts):
        # More precise words score higher: the count of known-good nouns.
        good = ("pink", "tulips", "glass", "vase", "twine")
        return [{"cos": s, "match": s} for s in (0.30 + 0.05 * sum(w in t.lower() for w in good) for t in texts)]


@pytest.fixture
def product(tmp_path, monkeypatch):
    monkeypatch.setattr(o, "DATA", tmp_path)
    (tmp_path / "images").mkdir()
    Image.new("RGB", (64, 64), "pink").save(tmp_path / "images" / "t.jpg")
    monkeypatch.setitem(o._scorers, "fake", FakeScorer())
    return {"id": "t", "name": "Pink Tulips", "image_file": "images/t.jpg",
            "current_alt": "Pink Tulips - Standard", "copy": "Pink tulips tied with twine in a glass vase."}


def test_loop_rewrites_until_target(product, tmp_path, monkeypatch):
    replies = iter([
        {"description": "pink tulips", "candidates": ["Tulips", "Pink tulips"]},
        {"candidates": ["Pink tulips tied with twine in a glass vase"]},
    ])
    monkeypatch.setattr(o, "call_writer", lambda w, p, i: (next(replies), {"input": 100, "output": 50}))
    r = o.optimize(product, out_dir=tmp_path / "res", scorer="fake", target=0.50, max_rounds=3)
    assert r["best"]["text"] == "Pink tulips tied with twine in a glass vase"
    assert r["reached_target"] and r["rewrite_rounds"] == 1 and not r["kept_original"]
    assert json.loads((tmp_path / "res" / "t.json").read_text())["best"]["match"] == r["best"]["match"]


def test_good_current_alt_is_kept(product, tmp_path, monkeypatch):
    product["current_alt"] = "Pink tulips tied with twine in a glass vase"
    monkeypatch.setattr(o, "call_writer", lambda w, p, i: ({"description": "", "candidates": ["Flowers"]}, {"input": 1, "output": 1}))
    r = o.optimize(product, out_dir=tmp_path / "res", scorer="fake", target=0.50)
    assert r["kept_original"] and r["already_met"]
