from pathlib import Path

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_classify_returns_valid_distribution():
    sample_path = next(Path("data/raw/test").glob("*/*.jpeg"))
    with open(sample_path, "rb") as f:
        resp = client.post(
            "/v1/sar/classify",
            files={"file": (sample_path.name, f, "image/jpeg")},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_class"] in body["class_probabilities"]
    assert len(body["class_probabilities"]) == 10
    total_prob = sum(body["class_probabilities"].values())
    assert 0.95 <= total_prob <= 1.05  # 확률 분포이므로 합이 대략 1


def test_classify_rejects_non_image():
    resp = client.post(
        "/v1/sar/classify",
        files={"file": ("not_an_image.txt", b"hello world", "text/plain")},
    )
    assert resp.status_code == 400
