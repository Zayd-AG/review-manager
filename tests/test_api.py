"""FastAPI route tests that do not load the fine-tuned model."""

from __future__ import annotations

import os
import unittest

# Lets health and request-validation tests run even when Postgres is not local.
os.environ.setdefault(
    "DATABASE_URL", "postgresql://feedbacklens:localdevpassword@localhost:5432/feedbacklens"
)

from fastapi.testclient import TestClient

from backend.app.main import app


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_health_is_public_and_successful(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_classification_rejects_overlong_review_before_inference(self) -> None:
        response = self.client.post("/classify", json={"text": "x" * 10_001})
        self.assertEqual(response.status_code, 422)

    def test_request_body_limit_rejects_large_payload(self) -> None:
        response = self.client.post("/classify", json={"text": "x" * 70_000})
        self.assertEqual(response.status_code, 413)

    @unittest.skipUnless(
        os.getenv("RUN_DATABASE_TESTS") == "1", "Database tests are enabled in CI"
    )
    def test_ready_can_connect_to_postgres(self) -> None:
        response = self.client.get("/ready")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ready"})


if __name__ == "__main__":
    unittest.main()
