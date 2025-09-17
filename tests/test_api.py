# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status":"ok"}

def test_optimize_empty(monkeypatch):
    # Simulate no input data directory
    monkeypatch.delenv("INPUT_DIR", raising=False)
    resp = client.get("/optimize")
    assert resp.status_code == 500 or resp.status_code == 400
