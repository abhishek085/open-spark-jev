"""CPU-only checks for the multi-model gateway and the bundled UI (no weights loaded)."""

from fastapi.testclient import TestClient

from open_spark_jev.serve import gateway


def test_ui_and_model_listing(monkeypatch):
    monkeypatch.setattr(gateway, "BACKEND_KIND", "hf")
    monkeypatch.setattr(gateway, "DEFAULT_MODEL", "sft-qwen3-1.7b")
    c = TestClient(gateway.app)
    r = c.get("/")
    assert r.status_code == 200 and "spark-s1" in r.text
    j = c.get("/v1/models").json()
    assert j["default"] == "sft-qwen3-1.7b" and isinstance(j["data"], list)


def test_boolean_alias_and_unknown_model(monkeypatch):
    q = gateway._from_jev("x", gateway.JevQuestion(type="boolean", instructions="It is raining."))
    assert q.type == "noul"
    monkeypatch.setattr(gateway, "BACKEND_KIND", "hf")
    monkeypatch.setattr(gateway, "DEFAULT_MODEL", "nope")
    r = TestClient(gateway.app).post("/v1/evaluate", json={"state": "s", "model": "nope", "questions": {"q": {"type": "boolean", "instructions": "x"}}})
    assert r.status_code == 404
