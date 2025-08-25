def test_root_endpoint(backend_client):
    r = backend_client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Successfully Running the server"}

def test_health_endpoint(backend_client):
    r = backend_client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "healthy"}

def test_predict_endpoint(backend_client):
    payload = {"alcohol": 10.0}
    r = backend_client.post("/predict", json=payload)
    body = r.json()
    assert r.status_code == 200
    assert "predicted_quality" in body

def test_predict_validation_error_on_extra_field(backend_client):
    payload = [{"alcohol": 15, "country": "DE", "junk": 1}]
    r = backend_client.post("/predict", json=payload)
    assert r.status_code == 422  # FastAPI returns 422 on Pydantic validation errors
