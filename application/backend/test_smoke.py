import unittest

from fastapi.testclient import TestClient

from application.backend.main import app


class BackendSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)

        payload = response.json()
        self.assertIn("model_loaded", payload)
        self.assertIn("model_status", payload)
        self.assertIn("features", payload)
        self.assertEqual(payload["features"], [
            "bulk_density",
            "organic_matter_pct",
            "cation_exchange_capacity",
            "salinity_ec",
        ])

    def test_predict_endpoint_with_valid_payload(self):
        payload = {
            "bulk_density": 1.2,
            "organic_matter_pct": 3.0,
            "cation_exchange_capacity": 15.0,
            "salinity_ec": 0.5,
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)

        result = response.json()
        self.assertIn("prediction", result)
        self.assertIn("interpretation", result)
        self.assertIn(result["prediction"], [0, 1])

    def test_predict_endpoint_rejects_invalid_input(self):
        payload = {
            "bulk_density": -1.0,
            "organic_matter_pct": 3.0,
            "cation_exchange_capacity": 15.0,
            "salinity_ec": 0.5,
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
