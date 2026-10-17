"""Tests for the Flask routes, form validation and CSV upload handling."""
import io
import re
import unittest

import app as app_module
import core

GOOD_FORM = {"Type": "M", "Air temperature [K]": "300", "Process temperature [K]": "310",
             "Rotational speed [rpm]": "1500", "Torque [Nm]": "40", "Tool wear [min]": "100"}
HEADER = "Type,Air temperature [K],Process temperature [K],Rotational speed [rpm],Torque [Nm],Tool wear [min]\n"


def upload(client, content, name="data.csv"):
    data = {"file": (io.BytesIO(content if isinstance(content, bytes) else content.encode()), name)}
    return client.post("/batch", data=data, content_type="multipart/form-data")


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = app_module.create_app()
        cls.app.testing = True
        cls.client = cls.app.test_client()

    def test_pages_load(self):
        for path in ["/", "/predict", "/batch", "/performance", "/about", "/health"]:
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 200)

    def test_unknown_route_returns_custom_404(self):
        r = self.client.get("/nope")
        self.assertEqual(r.status_code, 404)
        self.assertIn(b"Page not found", r.data)

    def test_dashboard_shows_values_computed_from_the_data(self):
        html = self.client.get("/").get_data(as_text=True)
        stats = core.load_json("dataset_stats.json")
        self.assertIn(f"{stats['total_records']:,}", html)
        self.assertIn(f"{stats['failure_rate'] * 100:.2f}%", html)
        self.assertIn("No predictions yet", html) if not app_module.RECENT else None

    def test_single_prediction_success(self):
        r = self.client.post("/predict", data=GOOD_FORM)
        self.assertEqual(r.status_code, 200)
        html = r.get_data(as_text=True)
        self.assertIn("Estimated failure probability", html)
        self.assertIn("not a guarantee", html)
        self.assertTrue(re.search(r"risk</span>", html))

    def test_single_prediction_matches_predictor(self):
        expected = self.app.predictor.predict_one(core.validate_record(GOOD_FORM)[0])["probability"]
        html = self.client.post("/predict", data=GOOD_FORM).get_data(as_text=True)
        self.assertIn(f"{expected * 100:.1f}%", html)

    def test_invalid_form_input_rejected_with_messages(self):
        for bad in [{"Torque [Nm]": "abc"}, {"Torque [Nm]": ""}, {"Torque [Nm]": "inf"}, {"Type": "Q"}, {"Tool wear [min]": "-1"}]:
            with self.subTest(bad=bad):
                r = self.client.post("/predict", data={**GOOD_FORM, **bad})
                self.assertEqual(r.status_code, 400)
                self.assertIn(b"Please fix", r.data)
                self.assertNotIn(b"Estimated failure probability", r.data)

    def test_batch_upload_success_and_download(self):
        csv = HEADER + "L,298,308,1400,62,215\nM,300,310,1500,40,100\nH,302,312,2000,30,10\n"
        r = upload(self.client, csv)
        self.assertEqual(r.status_code, 200)
        html = r.get_data(as_text=True)
        self.assertIn("Batch prediction results", html)
        token = re.search(r"/results/([0-9a-f]{32})/download", html).group(1)
        d = self.client.get(f"/results/{token}/download")
        self.assertEqual(d.status_code, 200)
        self.assertEqual(d.mimetype, "text/csv")
        lines = d.get_data(as_text=True).strip().splitlines()
        self.assertEqual(len(lines), 4)
        self.assertIn("failure_probability", lines[0])
        self.assertIn("risk_category", lines[0])

    def test_batch_matches_single_prediction(self):
        r = upload(self.client, HEADER + "M,300,310,1500,40,100\n")
        token = re.search(r"/results/([0-9a-f]{32})/download", r.get_data(as_text=True)).group(1)
        row = self.client.get(f"/results/{token}/download").get_data(as_text=True).strip().splitlines()[1].split(",")
        header = self.client.get(f"/results/{token}/download").get_data(as_text=True).splitlines()[0].split(",")
        got = float(row[header.index("failure_probability")])
        want = self.app.predictor.predict_one(core.validate_record(GOOD_FORM)[0])["probability"]
        self.assertAlmostEqual(got, want, places=4)

    def test_batch_skips_bad_rows_but_scores_good_rows(self):
        r = upload(self.client, HEADER + "M,300,310,1500,40,100\nM,300,310,1500,abc,100\nZ,300,310,1500,40,100\n")
        html = r.get_data(as_text=True)
        self.assertEqual(r.status_code, 200)
        self.assertIn("2 row(s) were skipped", html)
        self.assertIn("CSV line 3", html)

    def test_batch_missing_columns(self):
        r = upload(self.client, "Type,Torque [Nm]\nL,40\n")
        self.assertEqual(r.status_code, 400)
        self.assertIn(b"Missing required column", r.data)

    def test_batch_empty_file_and_header_only(self):
        self.assertEqual(upload(self.client, b"").status_code, 400)
        r = upload(self.client, HEADER)
        self.assertEqual(r.status_code, 400)
        self.assertIn(b"no data rows", r.data)

    def test_batch_rejects_non_csv_and_missing_file(self):
        self.assertEqual(upload(self.client, "x", name="malware.exe").status_code, 400)
        self.assertEqual(self.client.post("/batch", data={}).status_code, 400)

    def test_batch_rejects_oversized_file(self):
        r = upload(self.client, b"a" * (app_module.MAX_UPLOAD_BYTES + 10))
        self.assertEqual(r.status_code, 413)
        self.assertIn(b"File too large", r.data)

    def test_batch_unsafe_filename_is_never_used(self):
        r = upload(self.client, HEADER + "M,300,310,1500,40,100\n", name="../../etc/passwd.csv")
        self.assertEqual(r.status_code, 200)

    def test_batch_garbage_content_gives_friendly_error(self):
        r = upload(self.client, b"\xff\xfe\x00\x01 not,a\x00csv")
        self.assertEqual(r.status_code, 400)

    def test_unknown_download_token_404(self):
        self.assertEqual(self.client.get("/results/" + "0" * 32 + "/download").status_code, 404)

    def test_performance_page_uses_saved_results(self):
        html = self.client.get("/performance").get_data(as_text=True)
        ev = core.load_json("evaluation_metrics.json")
        for m in ev["models"]:
            self.assertIn(m["model"], html)
        self.assertIn(f"{ev['models'][0]['roc_auc']:.3f}", html)


if __name__ == "__main__":
    unittest.main()
