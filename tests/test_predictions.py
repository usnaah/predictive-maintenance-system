"""Tests for the saved model artifact and prediction logic."""
import unittest

import pandas as pd

import core


class PredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = core.Predictor()          # fails loudly if train_model.py has not been run
        cls.df = pd.read_csv(core.BASE_DIR / "data" / "ai4i2020.csv")

    def test_artifact_loads_with_expected_contents(self):
        self.assertEqual(self.predictor.model_name, "Random Forest")
        self.assertTrue(0 < self.predictor.threshold < 1)
        for col in core.RAW_NUMERIC:
            self.assertIn(col, self.predictor.ranges)

    def test_missing_artifact_raises(self):
        with self.assertRaises(FileNotFoundError):
            core.Predictor(core.BASE_DIR / "models" / "does_not_exist.joblib")

    def test_probabilities_are_valid_and_consistent(self):
        out = self.predictor.predict_frame(self.df[core.RAW_FEATURES].head(200))
        self.assertTrue(out["failure_probability"].between(0, 1).all())
        expected = (out["failure_probability"] >= self.predictor.threshold).astype(int)
        self.assertTrue((out["predicted_failure"] == expected).all())
        self.assertTrue(set(out["risk_category"]) <= {"Low", "Medium", "High"})

    def test_single_and_batch_predictions_agree(self):
        rows = self.df[core.RAW_FEATURES].head(10)
        batch = self.predictor.predict_frame(rows)
        for i in range(len(rows)):
            single = self.predictor.predict_one(rows.iloc[i].to_dict())
            self.assertAlmostEqual(single["probability"], batch["failure_probability"].iloc[i], places=9)
            self.assertEqual(single["risk"], batch["risk_category"].iloc[i])

    def test_column_order_does_not_matter(self):
        rows = self.df[core.RAW_FEATURES].head(5)
        shuffled = rows[list(reversed(core.RAW_FEATURES))]
        a = self.predictor.predict_frame(rows)["failure_probability"].values
        b = self.predictor.predict_frame(shuffled)["failure_probability"].values
        self.assertTrue((a == b).all())

    def test_risk_category_boundaries(self):
        t = 0.4
        self.assertEqual(core.risk_category(0.4, t), "High")
        self.assertEqual(core.risk_category(0.399, t), "Medium")
        self.assertEqual(core.risk_category(0.2, t), "Medium")
        self.assertEqual(core.risk_category(0.199, t), "Low")

    def test_obviously_stressed_machine_scores_higher_than_a_typical_one(self):
        typical = {"Type": "M", "Air temperature [K]": 300.0, "Process temperature [K]": 310.0,
                   "Rotational speed [rpm]": 1500.0, "Torque [Nm]": 40.0, "Tool wear [min]": 100.0}
        stressed = {**typical, "Torque [Nm]": 70.0, "Tool wear [min]": 230.0, "Rotational speed [rpm]": 1300.0}
        self.assertGreater(self.predictor.predict_one(stressed)["probability"],
                           self.predictor.predict_one(typical)["probability"])

    def test_out_of_range_inputs_are_flagged(self):
        rec = {"Type": "L", "Air temperature [K]": 300.0, "Process temperature [K]": 310.0,
               "Rotational speed [rpm]": 9000.0, "Torque [Nm]": 40.0, "Tool wear [min]": 100.0}
        self.assertTrue(self.predictor.predict_one(rec)["out_of_range"])


if __name__ == "__main__":
    unittest.main()
