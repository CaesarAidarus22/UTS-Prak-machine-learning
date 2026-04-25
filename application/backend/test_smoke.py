import unittest

from fastapi.testclient import TestClient

from application.backend.main import ALL_FEATURES, app


VALID_PAYLOAD = {
    "bulk_density": 1.2,
    "organic_matter_pct": 3.0,
    "cation_exchange_capacity": 15.0,
    "salinity_ec": 0.5,
    "buffering_capacity": 0.7,
    "soil_moisture_pct": 35.0,
    "moisture_limit_dry": 16.0,
    "moisture_limit_wet": 42.0,
    "soil_temp_c": 25.0,
    "air_temp_c": 28.0,
    "light_intensity_par": 700.0,
    "soil_ph": 6.5,
    "ph_stress_flag": 0,
    "nitrogen_ppm": 100.0,
    "phosphorus_ppm": 50.0,
    "potassium_ppm": 110.0,
    "soil_type": "Loamy",
    "moisture_regime": "optimal",
    "thermal_regime": "optimal",
    "nutrient_balance": "optimal",
    "plant_category": "vegetable",
}


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
        self.assertEqual(payload["features"], ALL_FEATURES)

    def test_predict_endpoint_with_valid_payload(self):
        response = self.client.post("/predict", json=VALID_PAYLOAD)
        self.assertEqual(response.status_code, 200)

        result = response.json()
        self.assertIn("prediction", result)
        self.assertIn("interpretation", result)
        self.assertIn(result["prediction"], [0, 1])

    def test_predict_endpoint_rejects_invalid_input(self):
        invalid_payload = dict(VALID_PAYLOAD)
        invalid_payload["bulk_density"] = -1.0

        response = self.client.post("/predict", json=invalid_payload)
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
