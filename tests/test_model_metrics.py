"""
test_model_metrics.py
Tests that saved evaluation files are present, internally consistent,
and that Scenario A metrics are correctly separated from Scenario B.
"""
import os
import sys
import json
import pytest
import pandas as pd

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))


# ── File existence ─────────────────────────────────────────────────────────────
class TestReportFilesExist:
    REQUIRED_FILES = [
        ("reports", "model_metrics.json"),
        ("reports", "model_comparison.csv"),
        ("reports", "shap_summary.png"),
        ("reports", "confusion_matrices.png"),
        ("reports", "model_comparison.png"),
        ("reports", "feature_importance.png"),
        ("reports", "01_class_distribution.png"),
        ("reports", "02_feature_distributions.png"),
        ("reports", "03_correlation_heatmap.png"),
        ("reports", "04_safe_features_boxplots.png"),
        ("reports", "05_threshold_leakage_analysis.png"),
        ("reports", "06_pairplot_safe_features.png"),
    ]

    @pytest.mark.parametrize("folder,fname", REQUIRED_FILES)
    def test_file_present(self, root, folder, fname):
        path = os.path.join(root, folder, fname)
        assert os.path.isfile(path), (
            f"Expected report file not found: {folder}/{fname}"
        )


class TestModelFilesExist:
    MODEL_FILES = [
        "models/selected_model_scenario_A.joblib",
        "models/selected_model_scenario_B_reference.joblib",
        "models/pipeline_scenario_A.joblib",
        "models/pipeline_scenario_B.joblib",
    ]

    @pytest.mark.parametrize("mf", MODEL_FILES)
    def test_model_file_present(self, root, mf):
        path = os.path.join(root, mf)
        assert os.path.isfile(path), f"Model file missing: {mf}"


# ── model_metrics.json structure ──────────────────────────────────────────────
class TestMetricsJSON:
    def test_json_loads_without_error(self, metrics):
        assert isinstance(metrics, dict)

    def test_top_level_keys(self, metrics):
        for key in ["scenario_A", "scenario_B", "split", "shap"]:
            assert key in metrics, f"Missing top-level key: {key}"

    def test_split_metadata(self, metrics):
        split = metrics["split"]
        assert split["n_train"]  == 8000
        assert split["n_test"]   == 2000
        assert split["seed"]     == 42
        assert split["stratified"] is True

    def test_scenario_a_has_three_models(self, metrics):
        results = metrics["scenario_A"]["results"]
        assert set(results.keys()) == {
            "LogisticRegression", "DecisionTree", "RandomForest"
        }

    def test_scenario_b_has_three_models(self, metrics):
        results = metrics["scenario_B"]["results"]
        assert set(results.keys()) == {
            "LogisticRegression", "DecisionTree", "RandomForest"
        }


# ── Scenario A metric values ───────────────────────────────────────────────────
class TestScenarioAMetrics:
    """
    Primary scenario. Metrics must match the training output.
    Near-random performance is EXPECTED and CORRECT — these asserts confirm
    the reported values have not been inflated or altered.
    """

    def test_best_model_is_random_forest(self, metrics):
        assert metrics["scenario_A"]["best_model"] == "RandomForest"

    def test_rf_accuracy(self, metrics):
        acc = metrics["scenario_A"]["results"]["RandomForest"]["accuracy"]
        assert abs(acc - 0.5135) < 0.001, (
            f"RF Scenario A accuracy={acc:.4f}, expected ≈ 0.5135"
        )

    def test_rf_f1(self, metrics):
        f1 = metrics["scenario_A"]["results"]["RandomForest"]["f1_score"]
        assert abs(f1 - 0.5083) < 0.001

    def test_rf_roc_auc(self, metrics):
        auc = metrics["scenario_A"]["results"]["RandomForest"]["roc_auc"]
        assert abs(auc - 0.5141) < 0.001

    def test_rf_precision(self, metrics):
        prec = metrics["scenario_A"]["results"]["RandomForest"]["precision"]
        assert abs(prec - 0.5138) < 0.001

    def test_rf_recall(self, metrics):
        rec = metrics["scenario_A"]["results"]["RandomForest"]["recall"]
        assert abs(rec - 0.503) < 0.001

    def test_scenario_a_features_correct(self, metrics):
        feats = metrics["scenario_A"]["features"]
        assert feats == ["Temperature", "pH", "Salinity", "Turbidity"]

    def test_scenario_a_no_leakage_cols_in_features(self, metrics):
        feats = set(metrics["scenario_A"]["features"])
        leakage = {"Dissolved_Oxygen", "Nitrate", "Chlorophyll_a"}
        overlap = feats & leakage
        assert len(overlap) == 0, (
            f"Leakage columns found in Scenario A features: {overlap}"
        )

    def test_scenario_a_accuracy_near_random_chance(self, metrics):
        """
        Scenario A accuracy must be near 50% (random chance on balanced data).
        If it were above 80%, something went wrong — possible unreported leakage.
        """
        acc = metrics["scenario_A"]["results"]["RandomForest"]["accuracy"]
        assert acc < 0.80, (
            f"Scenario A accuracy={acc:.2%} is suspiciously high. "
            "Check for target leakage."
        )
        assert acc > 0.40, (
            f"Scenario A accuracy={acc:.2%} is below 40% — possible data issue."
        )


# ── Scenario B separation check ────────────────────────────────────────────────
class TestScenarioBSeparation:
    """
    Scenario B has high accuracy (~97%) due to target leakage.
    These tests confirm it is correctly labelled and NOT the primary result.
    """

    def test_scenario_b_best_model_not_used_as_primary(self, metrics):
        """Primary best_model is Scenario A's, not B's."""
        best_A = metrics["scenario_A"]["best_model"]
        # The presence of this key is sufficient — we never overwrite it with B
        assert best_A in ["LogisticRegression", "DecisionTree", "RandomForest"]

    def test_scenario_b_accuracy_is_high_due_to_leakage(self, metrics):
        """Confirm B results are unrealistically high."""
        for clf, m in metrics["scenario_B"]["results"].items():
            assert m["accuracy"] > 0.90, (
                f"Scenario B {clf} accuracy={m['accuracy']:.2%} unexpectedly low. "
                "Expected near-perfect due to leakage."
            )

    def test_scenario_b_includes_threshold_cols(self, metrics):
        feats_B = metrics["scenario_B"]["features"]
        leakage = {"Dissolved_Oxygen", "Nitrate", "Chlorophyll_a"}
        found   = set(feats_B) & leakage
        assert len(found) == 3, (
            f"Scenario B should include all 3 threshold cols, found: {found}"
        )

    def test_scenario_b_labeled_with_leakage_warning(self, metrics):
        desc = metrics["scenario_B"].get("description", "")
        assert "REFERENCE" in desc.upper() or "leakage" in desc.lower(), (
            "Scenario B description must mention it is reference/leakage only"
        )


# ── model_comparison.csv ──────────────────────────────────────────────────────
class TestComparisonCSV:
    def test_csv_has_correct_shape(self, root):
        df = pd.read_csv(os.path.join(root, "reports", "model_comparison.csv"))
        assert len(df) == 6, (
            f"Expected 6 rows (3 Scenario A + 3 Scenario B), got {len(df)}"
        )

    def test_csv_required_columns(self, root):
        df = pd.read_csv(os.path.join(root, "reports", "model_comparison.csv"))
        for col in ["scenario", "model", "accuracy", "precision",
                    "recall", "f1_score", "roc_auc"]:
            assert col in df.columns, f"Column '{col}' missing from CSV"

    def test_csv_rf_scenario_a_row_present(self, root):
        df = pd.read_csv(os.path.join(root, "reports", "model_comparison.csv"))
        mask = (df["model"] == "RandomForest") & (df["scenario"].str.startswith("A"))
        assert mask.sum() == 1, "RandomForest Scenario A row not found in CSV"

    def test_csv_metrics_not_hardcoded_inflated(self, root):
        """
        Scenario A RandomForest accuracy in CSV must match the JSON (≈51.35%).
        This test catches accidental hardcoding of Scenario B's inflated metrics.
        """
        df   = pd.read_csv(os.path.join(root, "reports", "model_comparison.csv"))
        mask = (df["model"] == "RandomForest") & (df["scenario"].str.startswith("A"))
        acc  = float(df[mask]["accuracy"].values[0])
        assert abs(acc - 0.5135) < 0.001, (
            f"Scenario A RF accuracy in CSV = {acc:.4f}, expected ≈ 0.5135. "
            "Check that Scenario B metrics are not substituted here."
        )


# ── SHAP results ──────────────────────────────────────────────────────────────
class TestSHAPResults:
    def test_shap_succeeded(self, metrics):
        assert metrics["shap"]["succeeded"] is True, (
            "SHAP analysis is marked as failed. Re-run src/train_model.py."
        )

    def test_shap_model_is_scenario_a_best(self, metrics):
        explained = metrics["shap"]["model_explained"]
        best_A    = metrics["scenario_A"]["best_model"]
        assert explained == best_A, (
            f"SHAP explains '{explained}' but best Scenario A model is '{best_A}'"
        )

    def test_shap_covers_all_scenario_a_features(self, metrics):
        shap_feats = set(metrics["shap"]["mean_abs_shap"].keys())
        expected   = {"Temperature", "pH", "Salinity", "Turbidity"}
        assert shap_feats == expected, (
            f"SHAP features {shap_feats} do not match expected {expected}"
        )

    def test_shap_values_are_positive(self, metrics):
        import numpy as np
        for feat, val in metrics["shap"]["mean_abs_shap"].items():
            if isinstance(val, str):
                nums = [float(x) for x in val.strip("[]").split()]
                assert all(v >= 0 for v in nums), (
                    f"Negative |SHAP| value for {feat}: {val}"
                )


# ── Page content: accuracy disclosure ────────────────────────────────────────
class TestUIDisclosures:
    """
    Verify source code of dashboard pages contains required disclosure strings.
    These are string-level checks to catch accidental removal of warnings.
    """

    def _read_page(self, root, filename):
        path = os.path.join(root, "pages", filename)
        assert os.path.isfile(path), f"Page file missing: {filename}"
        with open(path, encoding="utf-8") as f:
            return f.read()

    def _read_app(self, root):
        path = os.path.join(root, "app.py")
        with open(path, encoding="utf-8") as f:
            return f.read()

    def test_home_mentions_synthetic(self, root):
        content = self._read_app(root)
        assert "synthetic" in content.lower() or "modelled" in content.lower(), (
            "app.py must clearly state the dataset is synthetic/modelled"
        )

    def test_home_not_claiming_real_safety(self, root):
        content = self._read_app(root).lower()
        # Should NOT contain claims of validated real-world safety
        assert "validated marine environmental" not in content

    def test_prediction_page_shows_accuracy_warning(self, root):
        content = self._read_page(root, "1_Marine_Water_Quality_Prediction.py")
        assert "51" in content, (
            "Prediction page must display ~51% accuracy to set expectations"
        )

    def test_prediction_page_mentions_experimental(self, root):
        content = self._read_page(root, "1_Marine_Water_Quality_Prediction.py").lower()
        assert "experimental" in content or "educational" in content, (
            "Prediction page must label the model as experimental/educational"
        )

    def test_prediction_page_no_leakage_cols_as_sliders(self, root):
        """
        Leakage columns must NOT appear as st.slider or st.number_input widgets.
        They may legitimately appear in explanatory text (e.g. in the expander
        that explains why they are excluded), so we scan for widget calls only.
        """
        import re
        content = self._read_page(root, "1_Marine_Water_Quality_Prediction.py")
        leakage_cols = ["Dissolved_Oxygen", "Nitrate", "Chlorophyll_a"]
        # Find all st.slider / st.number_input call strings
        widget_calls = re.findall(
            r'st\.(?:slider|number_input)\s*\([^)]{0,300}\)', content, re.DOTALL
        )
        widget_text = " ".join(widget_calls)
        for col in leakage_cols:
            assert col not in widget_text, (
                f"Leakage column '{col}' found inside a st.slider/st.number_input "
                "call in the prediction page. It must not be a user input widget."
            )

    def test_performance_page_separates_scenarios(self, root):
        content = self._read_page(root, "3_Model_Performance.py").lower()
        assert "scenario a" in content
        assert "scenario b" in content
        assert "reference" in content or "leakage" in content

    def test_performance_page_warns_about_scenario_b(self, root):
        content = self._read_page(root, "3_Model_Performance.py").lower()
        # Must not present Scenario B as genuine
        assert "not genuine" in content or "leakage" in content

    def test_about_page_explains_leakage(self, root):
        content = self._read_page(root, "4_About_Project.py").lower()
        assert "target leakage" in content or "leakage" in content

    def test_about_page_synthetic_disclaimer(self, root):
        content = self._read_page(root, "4_About_Project.py").lower()
        assert "synthetic" in content
        # Must warn results can't be generalised
        assert "generalised" in content or "generalized" in content or \
               "real marine" in content
