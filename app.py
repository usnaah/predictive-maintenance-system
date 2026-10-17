"""Flask web application - Predictive Maintenance Intelligence System.

The app NEVER trains a model: it loads the artifact saved by train_model.py
and uses the shared logic in core.py for validation, features and risk bands.
"""
import io
import logging
import os
import uuid
from collections import OrderedDict, deque
from datetime import datetime

import pandas as pd
from flask import Flask, Response, abort, flash, redirect, render_template, request, url_for

import core

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("pm.app")

MAX_UPLOAD_BYTES = 2 * 1024 * 1024                  # 2 MB
RECENT = deque(maxlen=8)                            # in-memory session history (cleared on restart)
BATCH_RESULTS: "OrderedDict[str, pd.DataFrame]" = OrderedDict()   # in-memory only, never written to disk
MAX_STORED_BATCHES = 10


def create_app(model_path=core.MODEL_PATH):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", os.urandom(16).hex())   # only used for flash messages
    predictor = core.Predictor(model_path)
    app.predictor = predictor

    # --------------------------------------------------------- pages ---
    @app.route("/")
    def dashboard():
        stats = core.load_json("dataset_stats.json")
        evaluation = core.load_json("evaluation_metrics.json")
        selected = None
        if evaluation:
            selected = next((m for m in evaluation["models"] if m["model"] == evaluation["selected_model"]), None)
        chart = None
        imp_path = core.OUTPUT_DIR / "feature_importance.csv"
        if imp_path.exists():
            imp = pd.read_csv(imp_path)
            imp = imp[imp["level"] == "raw input"].sort_values("importance_mean", ascending=False)
            chart = {"labels": imp["feature"].tolist(), "values": imp["importance_mean"].clip(lower=0).tolist(),
                     "format": "dec", "horizontal": True, "labelWidth": 190, "title": "Permutation importance"}
        return render_template("index.html", stats=stats, evaluation=evaluation, selected=selected,
                               importance_chart=chart, recent=list(RECENT), predictor=predictor, active="dashboard")

    @app.route("/predict", methods=["GET", "POST"])
    def predict():
        values, errors, result = {}, {}, None
        if request.method == "POST":
            values = {k: request.form.get(k, "") for k in core.RAW_FEATURES}
            clean, errors = core.validate_record(values)
            if clean:
                result = predictor.predict_one(clean)
                RECENT.appendleft({"time": datetime.now().strftime("%H:%M:%S"), "type": clean["Type"],
                                   "probability": result["probability"], "risk": result["risk"],
                                   "predicted": result["predicted_failure"]})
            else:
                log.info("Rejected form input: %s", errors)
        return render_template("predict.html", specs=core.FIELD_SPECS, values=values, errors=errors,
                               result=result, active="predict"), (400 if errors else 200)

    @app.route("/batch", methods=["GET", "POST"])
    def batch():
        if request.method == "GET":
            return render_template("batch_predict.html", specs=core.FIELD_SPECS, active="batch")
        file = request.files.get("file")
        if not file or not file.filename:
            return _batch_error("Please choose a CSV file to upload.")
        if not file.filename.lower().endswith(".csv"):
            return _batch_error("Only .csv files are accepted.")
        raw = file.read()                                # kept in memory only; no path is ever built from the filename
        if not raw.strip():
            return _batch_error("The uploaded file is empty.")
        try:
            df = pd.read_csv(io.BytesIO(raw))
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError):
            return _batch_error("The file could not be read as a CSV. Check the delimiter and encoding (UTF-8).")
        try:
            valid, issues = core.validate_batch(df)
        except core.BatchError as exc:
            return _batch_error(str(exc))
        if valid.empty:
            return _batch_error("No valid rows were found. See the row problems below.", issues=issues)
        scored = predictor.predict_frame(valid)
        token = uuid.uuid4().hex
        BATCH_RESULTS[token] = scored
        while len(BATCH_RESULTS) > MAX_STORED_BATCHES:
            BATCH_RESULTS.popitem(last=False)
        summary = {
            "uploaded_rows": int(len(df)), "scored_rows": int(len(scored)), "rejected_rows": len(issues),
            "predicted_failures": int(scored["predicted_failure"].sum()),
            "risk_counts": {r: int((scored["risk_category"] == r).sum()) for r in ("High", "Medium", "Low")},
        }
        shown = scored.sort_values("failure_probability", ascending=False).head(200)
        return render_template("results.html", token=token, summary=summary, issues=issues[:50],
                               more_issues=max(0, len(issues) - 50), rows=shown.to_dict("records"),
                               shown=len(shown), threshold=predictor.threshold, active="batch")

    def _batch_error(message, issues=None):
        return render_template("batch_predict.html", specs=core.FIELD_SPECS, error=message,
                               issues=(issues or [])[:50], active="batch"), 400

    @app.route("/results/<token>/download")
    def download(token):
        df = BATCH_RESULTS.get(token)
        if df is None:
            abort(404)
        out = df.rename(columns={"csv_line": "source_csv_line"}).copy()
        out["failure_probability"] = out["failure_probability"].round(4)
        out["predicted_failure"] = out["predicted_failure"].map({0: "No failure predicted", 1: "Failure predicted"})
        return Response(out.to_csv(index=False), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=batch_predictions.csv"})

    @app.route("/performance")
    def performance():
        evaluation = core.load_json("evaluation_metrics.json")
        importance = None
        path = core.OUTPUT_DIR / "feature_importance.csv"
        if path.exists():
            importance = pd.read_csv(path).to_dict("records")
        return render_template("model_performance.html", evaluation=evaluation, importance=importance,
                               predictor=predictor, active="performance")

    @app.route("/about")
    def about():
        return render_template("about.html", predictor=predictor, active="about")

    @app.route("/health")
    def health():
        return {"status": "ok", "model": predictor.model_name, "threshold": predictor.threshold}

    @app.context_processor
    def inject_predictor():
        return {"predictor": predictor}          # lets every page show the loaded model name

    # ------------------------------------------------------- filters ---
    @app.template_filter("pct")
    def pct(value, digits=1):
        return "-" if value is None else f"{value * 100:.{digits}f}%"

    @app.template_filter("num")
    def num(value, digits=3):
        return "-" if value is None else f"{value:.{digits}f}"

    # ------------------------------------------------------- errors ---
    @app.errorhandler(404)
    def not_found(_):
        return render_template("error.html", code=404, title="Page not found",
                               message="The page or result you asked for does not exist (batch results are kept only in memory for this session)."), 404

    @app.errorhandler(413)
    def too_large(_):
        return render_template("batch_predict.html", specs=core.FIELD_SPECS, active="batch",
                               error=f"File too large. The limit is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."), 413

    @app.errorhandler(500)
    def server_error(exc):
        log.exception("Unhandled error: %s", exc)
        return render_template("error.html", code=500, title="Something went wrong",
                               message="An unexpected error occurred. Details were written to the server log."), 500

    return app


if __name__ == "__main__":
    application = create_app()
    application.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
