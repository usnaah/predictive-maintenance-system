"""Tests for feature engineering, validation rules and the leakage safeguards."""
import io
import unittest

import numpy as np
import pandas as pd

import core
import train_model


def sample_frame():
    return pd.DataFrame({
        "Type": ["L", "M", "H"],
        "Air temperature [K]": [298.0, 300.0, 302.0],
        "Process temperature [K]": [308.0, 310.5, 312.0],
        "Rotational speed [rpm]": [1500, 1400, 2000],
        "Torque [Nm]": [40.0, 50.0, 30.0],
        "Tool wear [min]": [10, 150, 220],
    })


class FeatureEngineeringTests(unittest.TestCase):
    def test_engineered_values_are_correct(self):
        out = core.add_features(sample_frame())
        self.assertAlmostEqual(out["temp_diff_K"].iloc[0], 10.0)
        self.assertAlmostEqual(out["power_W"].iloc[0], 40.0 * 1500 * 2 * np.pi / 60)
        self.assertAlmostEqual(out["wear_x_torque"].iloc[1], 150 * 50.0)

    def test_does_not_modify_input(self):
        df = sample_frame()
        before = df.copy()
        core.add_features(df)
        pd.testing.assert_frame_equal(df, before)

    def test_leakage_columns_not_in_features(self):
        used = set(core.RAW_FEATURES) | set(core.ENGINEERED)
        for col in core.LEAKAGE_COLUMNS + core.IDENTIFIER_COLUMNS + [core.TARGET]:
            self.assertNotIn(col, used)


class PipelineTests(unittest.TestCase):
    def test_pipeline_output_has_no_nan_and_expected_width(self):
        pipe = train_model.build_pipeline(train_model.model_spaces()["Logistic Regression"][0])
        df = pd.read_csv(core.BASE_DIR / "data" / "ai4i2020.csv")
        pipe[:-1].fit(df[core.RAW_FEATURES])
        out = pipe[:-1].transform(df[core.RAW_FEATURES].head(50))
        self.assertFalse(np.isnan(np.asarray(out)).any())
        self.assertEqual(out.shape[1], len(core.NUMERIC_MODEL) + len(core.TYPE_VALUES))

    def test_scaler_is_fitted_on_training_data_only(self):
        df = pd.read_csv(core.BASE_DIR / "data" / "ai4i2020.csv")
        train, test = df.iloc[:8000], df.iloc[8000:]
        pipe = train_model.build_pipeline(train_model.model_spaces()["Logistic Regression"][0])
        pipe[:-1].fit(train[core.RAW_FEATURES])
        scaler = pipe.named_steps["prep"].named_transformers_["num"]
        expected_mean = core.add_features(train[core.RAW_FEATURES])[core.NUMERIC_MODEL].mean().values
        np.testing.assert_allclose(scaler.mean_, expected_mean)           # test rows did not influence the scaler


class ValidationTests(unittest.TestCase):
    good = {"Type": "L", "Air temperature [K]": "300", "Process temperature [K]": "310",
            "Rotational speed [rpm]": "1500", "Torque [Nm]": "40", "Tool wear [min]": "100"}

    def test_valid_record(self):
        clean, errors = core.validate_record(self.good)
        self.assertEqual(errors, {})
        self.assertEqual(clean["Rotational speed [rpm]"], 1500.0)

    def test_rejects_bad_values(self):
        for field, bad in [("Torque [Nm]", "abc"), ("Torque [Nm]", ""), ("Torque [Nm]", "nan"), ("Torque [Nm]", "inf"),
                           ("Torque [Nm]", "-5"), ("Tool wear [min]", "99999"), ("Type", "X"), ("Type", "")]:
            with self.subTest(field=field, bad=bad):
                clean, errors = core.validate_record({**self.good, field: bad})
                self.assertIsNone(clean)
                self.assertIn(field, errors)

    def test_batch_missing_columns(self):
        df = sample_frame().drop(columns=["Torque [Nm]"])
        with self.assertRaises(core.BatchError) as ctx:
            core.validate_batch(df)
        self.assertIn("Torque [Nm]", str(ctx.exception))

    def test_batch_duplicate_columns(self):
        df = sample_frame()
        df.columns = ["Type", "Type"] + list(df.columns[2:])
        with self.assertRaises(core.BatchError):
            core.validate_batch(df)

    def test_batch_empty(self):
        with self.assertRaises(core.BatchError):
            core.validate_batch(sample_frame().iloc[0:0])

    def test_batch_too_many_rows(self):
        big = pd.concat([sample_frame()] * (core.MAX_BATCH_ROWS // 3 + 2), ignore_index=True)
        with self.assertRaises(core.BatchError):
            core.validate_batch(big)

    def test_batch_flags_bad_rows_but_keeps_good_ones(self):
        df = sample_frame()
        df["Torque [Nm]"] = df["Torque [Nm]"].astype(object)
        df.loc[1, "Torque [Nm]"] = "oops"
        df.loc[2, "Type"] = "Z"
        valid, issues = core.validate_batch(df)
        self.assertEqual(len(valid), 1)
        self.assertEqual([i["csv_line"] for i in issues], [3, 4])

    def test_batch_trims_headers_and_normalises_type(self):
        df = sample_frame()
        df["Type"] = [" l", "m ", "H"]
        df.columns = [f" {c} " for c in df.columns]
        valid, issues = core.validate_batch(df)
        self.assertEqual(len(valid), 3)
        self.assertEqual(issues, [])


if __name__ == "__main__":
    unittest.main()
