from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data

def test_predict_validation_error():
    # Missing required fields
    response = client.post("/predict", json={"tenure": 5})
    assert response.status_code == 422
