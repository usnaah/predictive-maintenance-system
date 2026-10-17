# Requirements Checklist

Maps every item of the mentor's email to the file or feature that implements it, and how it was verified.
"Verified" means it was actually run in the build environment; anything that could not be run here is marked clearly.

## 1. Data understanding and EDA
| Requirement | Where implemented | Verified |
|---|---|---|
| Understand dataset, columns, business problem | Notebook sections 1-4 | Notebook executed end to end |
| Cleaning: missing values, duplicates, outliers, inconsistent data | Notebook section 5 and section 8 (outliers) | None missing/duplicated; outliers analysed and deliberately kept |
| Failure distribution and class imbalance | Notebook section 6 (visualization 1) | 339 failures = 3.39% |
| Univariate and multivariate analysis | Notebook sections 8-9 | Executed |
| Relationships between temperature, speed, torque, tool wear and failures | Notebook visualizations 4-9 | Executed |
| At least eight visualizations with insights | Notebook: 10 numbered visualizations plus confusion matrices, ROC/PR curves, threshold plot, importance plots; each followed by an explanation | Executed |

## 2. Feature engineering and preprocessing
| Requirement | Where | Verified |
|---|---|---|
| Identify features and target | `core.RAW_FEATURES`, `core.TARGET`; notebook section 7 | Unit test: no leakage/identifier columns in features |
| Encoding and scaling | `train_model.build_pipeline` (OneHotEncoder, StandardScaler) | `test_pipeline_output_has_no_nan_and_expected_width` |
| Derived features | `core.add_features` (temp_diff_K, power_W, wear_x_torque) | `test_engineered_values_are_correct` |
| Prevent leakage; fit preprocessing on training data only | Failure-mode columns and IDs excluded; preprocessing inside the Pipeline | `test_scaler_is_fitted_on_training_data_only` |
| Train/test split | Stratified 80/20, `random_state=42` | Class balance printed in notebook |

## 3. Model development
| Requirement | Where | Verified |
|---|---|---|
| Logistic Regression, Decision Tree, Random Forest, ANN | `train_model.model_spaces()` (ANN = `MLPClassifier`) | All four trained and evaluated |
| Hyperparameter tuning | `GridSearchCV`, 5-fold stratified, scored by PR-AUC | Best parameters on the Model Performance page |
| Select model for the business problem, not accuracy alone | Selection by cross-validated F2 at a tuned threshold (training data only) | `outputs/evaluation_metrics.json` |

## 4. Model evaluation
| Requirement | Where | Verified |
|---|---|---|
| Confusion matrix | Notebook section 14; Model Performance page | Executed / screenshot |
| Accuracy, Precision, Recall, F1 | `outputs/model_comparison.csv`; both at tuned and default 0.5 thresholds in notebook | Executed |
| ROC-AUC and PR-AUC | Same table; ROC/PR curves in notebook | Executed |
| Model comparison | Notebook; dashboard; Model Performance page | Screenshot |
| Feature importance / explainability | Permutation importance (`outputs/feature_importance.csv`), RF impurity importance (notebook) | Executed |
| Explain final model choice and missed-failure vs false-alarm trade-off | Notebook sections 14-17; Model Performance page; README | Written from real results |

## 5. Flask web application
| Requirement | Where | Verified |
|---|---|---|
| Dashboard with statistics and evaluation results | `/` -> `templates/index.html` | Test + screenshot; values read from `outputs/*.json` |
| Form for operating parameters | `/predict` -> `predict.html` | Tests + screenshot |
| Predicted class and failure probability | Result panel in `predict.html` | `test_single_prediction_matches_predictor` |
| CSV upload for batch predictions | `/batch` -> `batch_predict.html` | `test_batch_upload_success_and_download` |
| Results page with predictions and risk categories | `results.html`, CSV download at `/results/<token>/download` | Test + screenshot |
| Input validation and error handling | `core.validate_record`, `core.validate_batch`, error handlers in `app.py` | 15+ tests: bad values, missing columns, empty, oversized, wrong type, garbage file |
| Clean UI with HTML, CSS, Bootstrap | `templates/`, `static/css/style.css`; Bootstrap 5 loaded from CDN in `base.html` | **Bootstrap itself could not be loaded in the offline build environment**; `style.css` styles every Bootstrap class used, so screenshots show the app as it renders without the CDN |
| Model saved with joblib and loaded by Flask | `models/predictive_maintenance_model.joblib`, `core.Predictor` | `test_artifact_loads_with_expected_contents` |
| Training and web code separate | `train_model.py` vs `app.py`; shared logic in `core.py` | Flask starts without training |

## 6. Documentation and submission
| Deliverable | Where | Status |
|---|---|---|
| Jupyter Notebook (EDA, preprocessing, features, training, evaluation) | `notebooks/predictive_maintenance_eda.ipynb` | Executed in the build environment with a custom cell runner (Jupyter itself is not installed here); outputs and figures are embedded. Please also run it once in Jupyter |
| Flask application source | `app.py`, `core.py`, `templates/`, `static/` | Complete |
| Saved model and preprocessing | `models/predictive_maintenance_model.joblib` (pipeline includes preprocessing) | Complete |
| requirements.txt | Pinned versions used in the build | Complete |
| README | `README.md` | Complete |
| Screenshots | `docs/screenshots/` (10 images, desktop and mobile) | Real captures of the running app |
| GitHub repository | Repository-ready folder; publishing steps in README | **You must create and push the repository yourself**; nothing has been published |

## Additional expectations
| Expectation | Status |
|---|---|
| Clean, modular, reusable code with comments | `core.py` shared by training and app; docstrings throughout |
| No hard-coded predictions or metrics | Dashboard and performance pages read `outputs/`; tests compare page values with saved results |
| Works for individual and batch input | Tests confirm both paths give identical probabilities |
| Be prepared to explain the code | `docs/PROJECT_EXPLANATION.md` (18 topics + 24 viva questions) |

## Extra requirements from the project brief
Leakage investigation and documentation (notebook section 7, `core.py`, About page); documented risk thresholds (About page, README); clear separation of historical versus predicted counts (dashboard); empty state when no predictions exist (dashboard); uploads processed in memory with no path handling (`app.py`); automated tests (38) for preprocessing, predictions, batch, invalid input, missing columns, empty CSV, artifact loading and routes; logging in `app.py`; `.gitignore` excluding environments, caches and logs but keeping the model and outputs.

## Test status
`python -m unittest discover -s tests -t . -v` -> 38 tests, all passing in the build environment (Python 3.12.3, scikit-learn 1.8.0).
**Not tested here:** other Python or library versions, Windows/macOS, the live Bootstrap CDN, Jupyter kernel execution, production deployment.
