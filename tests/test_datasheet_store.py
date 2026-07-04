"""DatasheetStore: schema, per-faction replace scoping, unit/weapon search.

Phase 1 tests run against hand-built UnitProfile rows — no PDF ingest yet.
All content is original synthetic data, never GW material.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rulehound.config import SearchConfig
from rulehound.ingest.embed import HashingEmbedder
from rulehound.models import UnitProfile, WeaponProfile
from rulehound.search.hybrid import hybrid_search
from rulehound.search.spell import SpellCorrector
from rulehound.store.datasheet_store import DatasheetStore, faction_hash_key

DIM = 64


def _embed_text(u: UnitProfile) -> str:
    weapon_names = " ".join(w.name for w in u.weapons)
    return f"{u.name} {' '.join(u.keywords)} {u.abilities_text} {weapon_names}"


def iron_wardens() -> list[UnitProfile]:
    return [
        UnitProfile(
            unit_id="iron-wardens--shield-captain",
            faction="Iron Wardens",
            name="Shield Captain",
            movement='6"', toughness="5", save="2+", wounds="6",
            leadership="6+", oc="2",
            keywords=["Infantry", "Character", "Imperium"],
            abilities_text=(
                "Aegis of Command: While this model leads a unit, models in "
                "that unit have a 4+ invulnerable save. Once per battle it "
                "can reroll a failed charge."
            ),
            points="120",
            weapons=[
                WeaponProfile(
                    name="Sentinel Blade", range="Melee", attacks="5",
                    skill="2+", strength="5", ap="-2", damage="2",
                    keywords=["Lethal Hits"],
                ),
                WeaponProfile(
                    name="Guardian Pistol", range='12"', attacks="2",
                    skill="2+", strength="4", ap="-1", damage="1",
                    keywords=["Pistol"],
                ),
            ],
            raw_text="Shield Captain M 6 T 5 Sv 2+ W 6 Ld 6+ OC 2 Aegis of Command",
        ),
        UnitProfile(
            unit_id="iron-wardens--bulwark-squad",
            faction="Iron Wardens",
            name="Bulwark Squad",
            movement='5"', toughness="5", save="2+", wounds="3",
            leadership="6+", oc="2",
            keywords=["Infantry", "Battleline", "Imperium"],
            abilities_text=(
                "Shield Wall: Each time a ranged attack targets this unit, "
                "subtract 1 from the wound roll if it is wholly within your "
                "deployment zone."
            ),
            points="200",
            weapons=[
                WeaponProfile(
                    name="Bulwark Spear", range='18"', attacks="2",
                    skill="3+", strength="5", ap="-1", damage="2",
                    keywords=["Assault"],
                ),
            ],
            raw_text="Bulwark Squad M 5 T 5 Sv 2+ W 3 Ld 6+ OC 2 Shield Wall",
        ),
        # fallback-path record: stat parsing failed, only raw text is known
        UnitProfile(
            unit_id="iron-wardens--gyrfalcon-lander",
            faction="Iron Wardens",
            name="Gyrfalcon Lander",
            parse_confidence="fallback",
            raw_text=(
                "Gyrfalcon Lander M 20 T 9 Sv 3+ W 14 Ld 7+ OC 0 hover "
                "transport deep strike gyrfalcon missile battery"
            ),
        ),
    ]


def verdant_host() -> list[UnitProfile]:
    return [
        UnitProfile(
            unit_id="verdant-host--thorn-stalkers",
            faction="Verdant Host",
            name="Thorn Stalkers",
            movement='8"', toughness="4", save="4+", wounds="2",
            leadership="7+", oc="1",
            keywords=["Infantry", "Skirmisher"],
            abilities_text=(
                "Creeping Vines: This unit can move through terrain features "
                "as if they were not there, slipping between the branches."
            ),
            points="85",
            weapons=[
                WeaponProfile(
                    name="Thorn Launcher", range='24"', attacks="D6",
                    skill="4+", strength="4", ap="0", damage="1",
                    keywords=["Blast"],
                ),
            ],
            raw_text="Thorn Stalkers M 8 T 4 Sv 4+ W 2 Ld 7+ OC 1 Creeping Vines",
        ),
    ]


@pytest.fixture()
def dstore(tmp_path: Path) -> DatasheetStore:
    s = DatasheetStore(tmp_path / "datasheets.db", vector_dim=DIM)
    yield s
    s.close()


@pytest.fixture()
def populated(dstore: DatasheetStore) -> DatasheetStore:
    dstore.replace_faction(iron_wardens(), "Iron Wardens", "hash-iw-1")
    dstore.replace_faction(verdant_host(), "Verdant Host", "hash-vh-1")
    if dstore.vector_enabled:
        emb = HashingEmbedder(dimension=DIM)
        units = iron_wardens() + verdant_host()
        vectors = {
            u.unit_id: emb.encode([_embed_text(u)])[0] for u in units
        }
        dstore.store_vectors(vectors, emb.name, DIM)
    return dstore


# --- storage round-trip --------------------------------------------------


def test_unit_round_trip(populated: DatasheetStore):
    unit = populated.get_unit("iron-wardens--shield-captain")
    assert unit is not None
    assert unit.faction == "Iron Wardens"
    assert unit.toughness == "5"
    assert unit.save == "2+"
    assert unit.keywords == ["Infantry", "Character", "Imperium"]
    assert unit.points == "120"
    assert [w.name for w in unit.weapons] == ["Sentinel Blade", "Guardian Pistol"]
    assert unit.weapons[0].ap == "-2"
    assert unit.weapons[0].keywords == ["Lethal Hits"]


def test_weapon_ids_deterministic(populated: DatasheetStore):
    unit = populated.get_unit("iron-wardens--shield-captain")
    assert unit.weapons[0].weapon_id == "iron-wardens--shield-captain--sentinel-blade"
    got = populated.get_weapon("iron-wardens--shield-captain--sentinel-blade")
    assert got is not None
    weapon, unit_id = got
    assert weapon.name == "Sentinel Blade"
    assert unit_id == "iron-wardens--shield-captain"


def test_duplicate_weapon_names_get_suffixed(dstore: DatasheetStore):
    unit = UnitProfile(
        unit_id="f--u", faction="F", name="U", raw_text="u",
        weapons=[
            WeaponProfile(name="Twin Talon", range="Melee"),
            WeaponProfile(name="Twin Talon", range='12"'),
        ],
    )
    dstore.replace_faction([unit], "F", "h1")
    got = dstore.get_unit("f--u")
    assert [w.weapon_id for w in got.weapons] == ["f--u--twin-talon", "f--u--twin-talon-2"]


def test_fallback_unit_is_stored_and_searchable(populated: DatasheetStore):
    unit = populated.get_unit("iron-wardens--gyrfalcon-lander")
    assert unit is not None
    assert unit.parse_confidence == "fallback"
    assert unit.movement == "" and unit.weapons == []
    # still findable through FTS via raw_text
    hits = populated.keyword_search_units("gyrfalcon lander", k=5)
    assert hits and hits[0].rule_id == "iron-wardens--gyrfalcon-lander"


def test_counts_and_factions(populated: DatasheetStore):
    assert populated.unit_count() == 4
    assert populated.weapon_count() == 4
    assert populated.factions() == ["Iron Wardens", "Verdant Host"]
    assert populated.get_meta(faction_hash_key("Iron Wardens")) == "hash-iw-1"
    assert populated.get_meta(faction_hash_key("Verdant Host")) == "hash-vh-1"


# --- per-faction replace scoping ------------------------------------------


def test_replace_faction_is_additive(populated: DatasheetStore):
    """Re-ingesting one faction must not touch the other."""
    updated = iron_wardens()[:2]  # drop the lander, tweak a stat
    updated[0].points = "130"
    populated.replace_faction(updated, "Iron Wardens", "hash-iw-2")

    # Verdant Host untouched
    assert populated.get_unit("verdant-host--thorn-stalkers") is not None
    hits = populated.keyword_search_units("thorn stalkers", k=5)
    assert hits and hits[0].rule_id == "verdant-host--thorn-stalkers"

    # Iron Wardens replaced: lander gone, points updated, hash updated
    assert populated.get_unit("iron-wardens--gyrfalcon-lander") is None
    assert populated.get_unit("iron-wardens--shield-captain").points == "130"
    assert populated.get_meta(faction_hash_key("Iron Wardens")) == "hash-iw-2"
    assert populated.unit_count() == 3


def test_replace_faction_purges_stale_fts(populated: DatasheetStore):
    populated.replace_faction(iron_wardens()[:2], "Iron Wardens", "hash-iw-2")
    hits = populated.keyword_search_units("gyrfalcon", k=5)
    assert not any(h.rule_id == "iron-wardens--gyrfalcon-lander" for h in hits)
    whits = populated.keyword_search_weapons("thorn launcher", k=5)
    assert whits and whits[0].rule_id == "verdant-host--thorn-stalkers--thorn-launcher"


def test_replace_faction_purges_stale_vectors(populated: DatasheetStore):
    if not populated.vector_enabled:
        pytest.skip("sqlite-vec unavailable")
    emb = HashingEmbedder(dimension=DIM)
    # replace Iron Wardens with a single unit and store only its vector
    solo = iron_wardens()[:1]
    populated.replace_faction(solo, "Iron Wardens", "hash-iw-2")
    populated.store_vectors(
        {solo[0].unit_id: emb.encode([_embed_text(solo[0])])[0]}, emb.name, DIM
    )
    # every vector hit must resolve to a live unit (no stale rowid reuse)
    query_vec = emb.encode(["shield captain aegis of command"])[0]
    for hit in populated.vector_search_units(query_vec, k=10):
        assert populated.get_unit(hit.rule_id) is not None
    # Verdant Host's vector survived the Iron Wardens replace
    vh_vec = emb.encode([_embed_text(verdant_host()[0])])[0]
    hits = populated.vector_search_units(vh_vec, k=3)
    assert any(h.rule_id == "verdant-host--thorn-stalkers" for h in hits)


# --- search ---------------------------------------------------------------


def test_keyword_search_units_by_name(populated: DatasheetStore):
    hits = populated.keyword_search_units("bulwark squad", k=5)
    assert hits and hits[0].rule_id == "iron-wardens--bulwark-squad"


def test_keyword_search_units_by_keyword(populated: DatasheetStore):
    hits = populated.keyword_search_units("battleline", k=5)
    assert [h.rule_id for h in hits] == ["iron-wardens--bulwark-squad"]


def test_keyword_search_units_prefix_typing(populated: DatasheetStore):
    """search-as-you-type: last token is prefix-matched."""
    hits = populated.keyword_search_units("bulw", k=5)
    assert hits and hits[0].rule_id == "iron-wardens--bulwark-squad"


def test_keyword_search_weapons(populated: DatasheetStore):
    hits = populated.keyword_search_weapons("sentinel blade", k=5)
    assert hits and hits[0].rule_id == "iron-wardens--shield-captain--sentinel-blade"
    assert hits[0].title == "Sentinel Blade"


def test_keyword_search_weapons_by_ability_keyword(populated: DatasheetStore):
    hits = populated.keyword_search_weapons("lethal hits", k=5)
    assert [h.rule_id for h in hits] == ["iron-wardens--shield-captain--sentinel-blade"]


def test_hybrid_search_over_units_surface(populated: DatasheetStore):
    """hybrid_search runs unchanged against the units surface."""
    embedder = HashingEmbedder(dimension=DIM) if populated.vector_enabled else None
    cfg = SearchConfig()
    results, timings, _ = hybrid_search(populated.units, embedder, "shield captain", cfg)
    assert results and results[0].rule_id == "iron-wardens--shield-captain"
    assert timings.total_ms < 300


def test_hybrid_search_over_weapons_surface(populated: DatasheetStore):
    results, _, _ = hybrid_search(populated.weapons, None, "thorn launcher", SearchConfig())
    assert results and results[0].rule_id == "verdant-host--thorn-stalkers--thorn-launcher"


def test_hybrid_search_with_datasheet_spell_correction(populated: DatasheetStore):
    """Typo in a unit name is corrected from the datasheet vocab."""
    corrector = SpellCorrector(populated.load_vocab())
    results, _, correction = hybrid_search(
        populated.units, None, "bulwerk squad", SearchConfig(), corrector=corrector
    )
    assert correction is not None and correction.changed
    assert results and results[0].rule_id == "iron-wardens--bulwark-squad"


def test_vocab_built_from_datasheet_corpus(populated: DatasheetStore):
    vocab = populated.load_vocab()
    assert "bulwark" in vocab
    assert "sentinel" in vocab  # weapon names included
    assert "gyrfalcon" in vocab  # fallback units' raw_text included


def test_empty_store_searches_cleanly(dstore: DatasheetStore):
    assert dstore.keyword_search_units("anything", k=5) == []
    assert dstore.keyword_search_weapons("anything", k=5) == []
    assert dstore.unit_count() == 0
    assert dstore.factions() == []
