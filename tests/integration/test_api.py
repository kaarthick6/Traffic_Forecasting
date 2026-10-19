import pytest
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("pydantic_settings")
from fastapi.testclient import TestClient
from app.main import app

def test_health_and_api_contract():
    with TestClient(app) as client:
        assert client.get("/health").json()=={"status":"ok"}
        schema=client.get("/openapi.json").json()
        assert "/api/v1/travel-time/estimate" in schema["paths"]
        assert "/api/v1/predictions" in schema["paths"]

