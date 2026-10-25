"""
test_prediction_pipeline.py
End-to-end tests for the prediction module and ML pipeline.
"""
import os
import sys
import pytest
import numpy as np
import pandas as pd

# Ensure src/ is on the path (conftest also does this, belt-and-suspenders)
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import prediction as pred_module
from prediction import (
    predict, validate_inputs, load_pipeline,
    get_pipeline_info, FEATURES, FEATURE_RANGES, CLASS_LABELS,
)


# ── Pipeline loading ───────────────────────────────────────────────────────────
class TestPipelineLoading:
    def test_model_file_exists(self):
        path = os.path.join(ROOT, "models", "selected_model_scenario_A.joblib")
        assert os.path.isfile(path), "Model A .joblib file is missing"

    def test_pipeline_loads_without_error(self):
        pipe = load_pipeline()
        assert pipe is not None

    def test_pipeline_steps_are_correct(self):
        pipe  = load_pipeline()
        steps = [name for name, _ in pipe.steps]
        assert steps == ["imputer", "scaler", "model"], (
            f"Expected ['imputer','scaler','model'], got {steps}"
        )

    def test_pipeline_model_is_random_forest(self):
        pipe = load_pipeline()
        model_name = type(pipe.named_steps["model"]).__name__
        assert model_name == "RandomForestClassifier", (
            f"Expected RandomForestClassifier, got {model_name}"
        )

    def test_scaler_fitted_on_4_features(self):
        pipe = load_pipeline()
        n = pipe.named_steps["scaler"].n_features_in_
        assert n == 4, f"Scaler fitted on {n} features, expected 4"

    def test_load_is_singleton(self):
        """Loading twice must return the same object (cached)."""
        pred_module._pipeline = None   # reset cache
        pipe1 = load_pipeline()
        pipe2 = load_pipeline()
        assert pipe1 is pipe2, "load_pipeline() should cache and return same object"


# ── Feature order & constants ──────────────────────────────────────────────────
class TestFeatureOrder:
    def test_feature_list_length(self):
        assert len(FEATURES) == 4

    def test_feature_exact_names_and_order(self):
        assert FEATURES == ["Temperature", "pH", "Salinity", "Turbidity"]

    def test_class_labels_mapping(self):
        assert CLASS_LABELS[0] == "Unsafe"
        assert CLASS_LABELS[1] == "Safe"

    def test_no_leakage_cols_in_features(self):
        leakage = {"Dissolved_Oxygen", "Nitrate", "Chlorophyll_a"}
        for lf in leakage:
            assert lf not in FEATURES, (
                f"Leakage column '{lf}' found in Scenario A FEATURES!"
            )

    def test_pipeline_info_features_match(self):
        info = get_pipeline_info()
        assert info["features"] == FEATURES


# ── Input validation ───────────────────────────────────────────────────────────
class TestInputValidation:
    """validate_inputs() must accept values within range and reject outside."""

    VALID = {"Temperature": 22.5, "pH": 7.8, "Salinity": 35.0, "Turbidity": 15.0}

    def test_valid_mid_range(self):
        ok, msg = validate_inputs(self.VALID)
        assert ok is True, f"Valid mid-range inputs rejected: {msg}"

    def test_valid_at_exact_minimum(self):
        vals = {"Temperature": 10.0, "pH": 6.5, "Salinity": 30.0, "Turbidity": 0.5}
        ok, _ = validate_inputs(vals)
        assert ok is True, "Exact minimum boundary should be valid"

    def test_valid_at_exact_maximum(self):
        vals = {"Temperature": 32.0, "pH": 8.8, "Salinity": 40.0, "Turbidity": 50.0}
        ok, _ = validate_inputs(vals)
        assert ok is True, "Exact maximum boundary should be valid"

    def test_invalid_temperature_below_min(self):
        vals = {**self.VALID, "Temperature": 9.9}
        ok, msg = validate_inputs(vals)
        assert ok is False
        assert "Temperature" in msg

    def test_invalid_temperature_above_max(self):
        vals = {**self.VALID, "Temperature": 32.1}
        ok, msg = validate_inputs(vals)
        assert ok is False

    def test_invalid_ph_below_min(self):
        vals = {**self.VALID, "pH": 6.4}
        ok, msg = validate_inputs(vals)
        assert ok is False
        assert "pH" in msg

    def test_invalid_ph_above_max(self):
        vals = {**self.VALID, "pH": 9.0}
        ok, msg = validate_inputs(vals)
        assert ok is False

    def test_invalid_salinity_below_min(self):
        vals = {**self.VALID, "Salinity": 29.9}
        ok, msg = validate_inputs(vals)
        assert ok is False

    def test_invalid_salinity_above_max(self):
        vals = {**self.VALID, "Salinity": 40.1}
        ok, msg = validate_inputs(vals)
        assert ok is False

    def test_invalid_turbidity_below_min(self):
        vals = {**self.VALID, "Turbidity": 0.4}
        ok, msg = validate_inputs(vals)
        assert ok is False

    def test_invalid_turbidity_above_max(self):
        vals = {**self.VALID, "Turbidity": 50.1}
        ok, msg = validate_inputs(vals)
        assert ok is False

    def test_missing_feature_key(self):
        vals = {"Temperature": 22.5, "pH": 7.8}  # Salinity & Turbidity absent
        ok, msg = validate_inputs(vals)
        assert ok is False


# ── Prediction output ──────────────────────────────────────────────────────────
class TestPredictFunction:
    def test_returns_dict_with_required_keys(self):
        result = predict(22.5, 7.8, 35.0, 15.0)
        for key in ["prediction", "label", "prob_unsafe", "prob_safe", "valid", "error"]:
            assert key in result, f"Key '{key}' missing from predict() output"

    def test_valid_prediction_value(self):
        result = predict(22.5, 7.8, 35.0, 15.0)
        assert result["valid"] is True
        assert result["prediction"] in [0, 1]

    def test_label_matches_prediction(self):
        result = predict(22.5, 7.8, 35.0, 15.0)
        assert result["label"] == CLASS_LABELS[result["prediction"]]

    def test_probabilities_sum_to_one(self):
        result = predict(22.5, 7.8, 35.0, 15.0)
        total = round(result["prob_safe"] + result["prob_unsafe"], 4)
        assert abs(total - 1.0) < 0.001, (
            f"Probabilities don't sum to 1: {total}"
        )

    def test_probabilities_in_unit_interval(self):
        result = predict(22.5, 7.8, 35.0, 15.0)
        assert 0.0 <= result["prob_safe"]   <= 1.0
        assert 0.0 <= result["prob_unsafe"] <= 1.0

    def test_invalid_input_returns_valid_false(self):
        result = predict(9.0, 7.8, 35.0, 15.0)  # Temp < 10.0
        assert result["valid"]      is False
        assert result["prediction"] is None
        assert result["error"]      is not None
        assert len(result["error"]) > 0

    def test_invalid_input_no_prediction(self):
        result = predict(22.5, 99.0, 35.0, 15.0)  # pH unrealistic
        assert result["valid"]      is False
        assert result["label"]      is None

    def test_multiple_valid_predictions_consistent(self):
        """Same input must always produce same output (deterministic RF)."""
        r1 = predict(20.0, 7.5, 34.0, 20.0)
        r2 = predict(20.0, 7.5, 34.0, 20.0)
        assert r1["prediction"] == r2["prediction"]
        assert r1["prob_safe"]  == r2["prob_safe"]

    def test_named_dataframe_passes_to_pipeline(self):
        """
        Regression check: prediction must not raise sklearn feature-name warnings.
        The pipeline should receive a named DataFrame, not a raw numpy array.
        """
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            try:
                predict(22.5, 7.8, 35.0, 15.0)
            except UserWarning as w:
                if "feature names" in str(w):
                    pytest.fail(
                        f"sklearn feature-name warning raised: {w}"
                    )


# ── Pipeline inference with raw data ──────────────────────────────────────────
class TestRawPipelineInference:
    """
    Confirm the full sklearn pipeline transforms raw (unscaled) input correctly.
    """
    def test_pipeline_predict_with_named_df(self):
        pipe = load_pipeline()
        X = pd.DataFrame([{
            "Temperature": 22.5, "pH": 7.8,
            "Salinity": 35.0, "Turbidity": 15.0
        }])[FEATURES]
        pred = pipe.predict(X)
        assert pred[0] in [0, 1]

    def test_pipeline_predict_proba_shape(self):
        pipe = load_pipeline()
        X = pd.DataFrame([
            {"Temperature": 22.5, "pH": 7.8, "Salinity": 35.0, "Turbidity": 15.0},
            {"Temperature": 15.0, "pH": 7.0, "Salinity": 32.0, "Turbidity": 40.0},
        ])[FEATURES]
        proba = pipe.predict_proba(X)
        assert proba.shape == (2, 2), f"Expected (2,2), got {proba.shape}"

    def test_pipeline_output_classes(self):
        pipe = load_pipeline()
        assert list(pipe.classes_) == [0, 1]
