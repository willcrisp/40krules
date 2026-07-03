"""API contract tests (§7): response shapes for /search, /rule/{id}, /health,
plus crop serving and the ingest upload endpoint."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rulehound.api.app import create_app


@pytest.fixture()
def client(ingested):
    cfg, _ = ingested
    app = create_app(cfg)
    with TestClient(app) as c:
        yield c


def test_search_response_shape(client) -> None:
    res = client.get("/search", params={"q": "disembark"})
    assert res.status_code == 200
    body = res.json()
    assert body["query"] == "disembark"
    assert set(body["latency_ms"]) == {"embed", "keyword", "vector", "total"}
    assert body["results"]
    top = body["results"][0]
    for key in ("rule_id", "title", "section_path", "text", "pages", "crops", "score", "related"):
        assert key in top, f"missing {key}"
    assert top["rule_id"].endswith("disembark")
    assert top["text"]  # verbatim text present
    # only the top result carries `related`
    for other in body["results"][1:]:
        assert "related" not in other


def test_search_reports_spell_correction(client) -> None:
    body = client.get("/search", params={"q": "dismbark rules"}).json()
    assert body["corrected_query"] == "disembark rules"
    assert body["results"][0]["rule_id"].endswith("disembark")
    # clean queries carry no correction field
    clean = client.get("/search", params={"q": "disembark rules"}).json()
    assert "corrected_query" not in clean


def test_crops_are_served(client) -> None:
    res = client.get("/search", params={"q": "disembark"})
    crops = res.json()["results"][0]["crops"]
    assert crops, "top result has no crop images"
    img = client.get(crops[0])
    assert img.status_code == 200
    assert img.headers["content-type"] == "image/png"


def test_rule_endpoint_with_neighbours(client) -> None:
    rid = client.get("/search", params={"q": "disembark"}).json()["results"][0]["rule_id"]
    res = client.get(f"/rule/{rid}")
    assert res.status_code == 200
    body = res.json()
    assert body["rule_id"] == rid
    assert "outgoing" in body["neighbours"] and "incoming" in body["neighbours"]
    assert any(n["rule_id"].endswith("reserves") for n in body["neighbours"]["outgoing"])


def test_rule_404(client) -> None:
    assert client.get("/rule/not-a-rule").status_code == 404


def test_health_reports_meta(client, ingested) -> None:
    _, summary = ingested
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["rules"] == summary["chunks"]
    assert body["db_meta"]["doc_hash"] == summary["doc_hash"]
    assert body["db_meta"]["embedding_model"] == "hash"


def test_ingest_upload_and_status(tmp_path, fixture_pdf) -> None:
    from .conftest import make_config

    app = create_app(make_config(tmp_path))
    with TestClient(app) as c:
        assert c.get("/health").json()["rules"] == 0
        with open(fixture_pdf, "rb") as f:
            res = c.post("/ingest", files={"file": ("fixture.pdf", f, "application/pdf")})
        assert res.status_code == 200

        import time

        for _ in range(100):
            status = c.get("/ingest/status").json()
            if status["state"] in ("done", "error"):
                break
            time.sleep(0.1)
        assert status["state"] == "done", status
        assert c.get("/health").json()["rules"] > 0
        assert c.get("/search", params={"q": "disembark"}).json()["results"]


def test_ingest_rejects_non_pdf(client) -> None:
    res = client.post("/ingest", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert res.status_code == 400
