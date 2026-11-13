"""Gazetteer / country matcher / origin / map geometry checks (offline; uses the built gazetteer)."""
import json

import pytest

from fnp.config import DATA_DIR
from fnp.geo.countries import GAZ_PATH, CountryMatcher, domain_country

pytestmark = pytest.mark.skipif(not GAZ_PATH.exists(), reason="gazetteer not built")


@pytest.fixture(scope="module")
def m():
    return CountryMatcher()


def isos(m, t):
    return sorted(h["iso3"] for h in m.match(t))


@pytest.mark.parametrize("text,expected", [
    ("Modi meets Biden at the White House; India-US ties deepen", ["IND", "USA"]),
    ("Pakistan army shells across LoC in Kashmir", ["IND", "PAK"]),
    ("चीन ने भारत की सीमा पर सैनिक भेजे", ["CHN", "IND"]),
    ("Россия и Украина: переговоры", ["RUS", "UKR"]),
    ("以色列和伊朗", ["IRN", "ISR"]),
    ("🇧🇷 Lula's victory blow to Trump in Latin America", ["BRA"]),
    ("Georgia's Tbilisi protests", ["GEO"]),
    ("Turkey's Erdogan visits Ankara", ["TUR"]),
])
def test_known_posts(m, text, expected):
    assert isos(m, text) == expected


@pytest.mark.parametrize("text", ["I love turkey sandwiches", "Let us go to Georgia, Atlanta is hot",
                                  "Michael Jordan returns", "Native American tribes protest",
                                  "nothing geographic here", "RSS feed updated"])
def test_ambiguous_and_blocked(m, text):
    assert m.match(text) == []


def test_weights_sum_to_one(m):
    h = m.match("India, Pakistan and China held talks in Beijing")
    assert abs(sum(x["weight"] for x in h) - 1) < 1e-9 and len(h) == 3


def test_domain_country():
    assert domain_country("www.dawn.com.pk")[0] == "PAK"
    assert domain_country("timesofindia.indiatimes.com", {"indiatimes.com": "IND"}) == ("IND", "outlet")
    assert domain_country("example.io") is None          # vanity ccTLD is not a country


def test_world_paths_india_worldview():
    p = DATA_DIR / "geo" / "world_paths_v1.json"
    if not p.exists():
        pytest.skip("world paths not built")
    w = json.loads(p.read_text())
    assert {"IND", "PAK", "CHN", "USA"} <= set(w["countries"]) and "ATA" not in w["countries"]
    assert len(w["viewBox"].split()) == 4
