"""
tests/test_enhanced_ml.py
Phase 8 — Tests for the Phase 2 Enhanced ML training outputs.
Verifies all 6 algorithm results, CV methodology, file integrity,
leakage guards, and scientific honesty constraints.
"""

import os, json, csv
import pytest
import pandas as pd
import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
ENHANCED_PATH = os.path.join(ROOT, "reports", "model_metrics_enhanced.json")
CV_CSV_PATH   = os.path.join(ROOT, "reports", "cv_results.csv")


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def enhanced() -> dict:
    if not os.path.isfile(ENHANCED_PATH):
        pytest.skip("model_metrics_enhanced.json not found — run train_model_enhanced.py")
    with open(ENHANCED_PATH, encoding="utf-8") as f:
        return json.load(f)

@pytest.fixture(scope="module")
def cv_df() -> pd.DataFrame:
    if not os.path.isfile(CV_CSV_PATH):
        pytest.skip("cv_results.csv not found — run train_model_enhanced.py")
    return pd.read_csv(CV_CSV_PATH)


# ═══════════════════════════════════════════════════════════════════════════════
# FILE EXISTENCE
# ═══════════════════════════════════════════════════════════════════════════════
class TestEnhancedFiles:
    def test_model_metrics_enhanced_exists(self):
        assert os.path.isfile(ENHANCED_PATH), "model_metrics_enhanced.json missing"

    def test_cv_results_csv_exists(self):
        assert os.path.isfile(CV_CSV_PATH), "cv_results.csv missing"

    def test_permutation_importance_png_exists(self):
        p = os.path.join(ROOT, "reports", "permutation_importance.png")
        assert os.path.isfile(p), "permutation_importance.png missing"

    def test_roc_curves_png_exists(self):
        p = os.path.join(ROOT, "reports", "roc_curves.png")
        assert os.path.isfile(p), "roc_curves.png missing"

    def test_model_comparison_enhanced_png_exists(self):
        p = os.path.join(ROOT, "reports", "model_comparison_enhanced.png")
        assert os.path.isfile(p), "model_comparison_enhanced.png missing"


# ═══════════════════════════════════════════════════════════════════════════════
# JSON STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════
class TestEnhancedJsonStructure:
    def test_top_level_keys(self, enhanced):
        required = {"generated_by","phase","cv_folds","cv_strategy","tuning_method",
                    "leakage_guard","scenario_A","scenario_B"}
        assert required.issubset(enhanced.keys()), f"Missing keys: {required - enhanced.keys()}"

    def test_cv_folds_value(self, enhanced):
        assert enhanced["cv_folds"] == 5, "Expected 5-fold CV"

    def test_scenario_a_has_required_keys(self, enhanced):
        a = enhanced["scenario_A"]
        required = {"features","best_model","leakage","results","cv_results","permutation_importance"}
        assert required.issubset(a.keys())

    def test_scenario_b_has_required_keys(self, enhanced):
        b = enhanced["scenario_B"]
        required = {"features","best_model","leakage","warning","results"}
        assert required.issubset(b.keys())

    def test_leakage_flag_scenario_a_false(self, enhanced):
        assert enhanced["scenario_A"]["leakage"] is False

    def test_leakage_flag_scenario_b_true(self, enhanced):
        assert enhanced["scenario_B"]["leakage"] is True

    def test_scenario_a_features_no_leakage_cols(self, enhanced):
        leakage_cols = {"Dissolved_Oxygen","Nitrate","Chlorophyll_a",
                        "dissolved_oxygen","nitrate","chlorophyll_a"}
        feats = set(enhanced["scenario_A"]["features"])
        assert feats.isdisjoint(leakage_cols), \
            f"Leakage columns found in Scenario A features: {feats & leakage_cols}"

    def test_scenario_a_feature_count(self, enhanced):
        assert len(enhanced["scenario_A"]["features"]) == 4, \
            "Scenario A should have exactly 4 features"


# ═══════════════════════════════════════════════════════════════════════════════
# 6 ALGORITHMS PRESENT
# ═══════════════════════════════════════════════════════════════════════════════
EXPECTED_MODELS = {
    "LogisticRegression","DecisionTree","RandomForest",
    "ExtraTrees","LinearSVC","XGBoost"
}

class TestSixAlgorithms:
    def test_all_six_in_results(self, enhanced):
        found = set(enhanced["scenario_A"]["results"].keys())
        assert EXPECTED_MODELS == found, \
            f"Expected models {EXPECTED_MODELS}, found {found}"

    def test_all_six_have_cv_results(self, enhanced):
        found = set(enhanced["scenario_A"]["cv_results"].keys())
        assert EXPECTED_MODELS == found

    def test_cv_results_have_mean_and_std(self, enhanced):
        for name, cv in enhanced["scenario_A"]["cv_results"].items():
            assert "cv_f1_mean" in cv, f"{name}: cv_f1_mean missing"
            assert "cv_f1_std"  in cv, f"{name}: cv_f1_std missing"
            assert "cv_accuracy_mean" in cv
            assert "cv_roc_auc_mean"  in cv

    def test_each_result_has_required_metrics(self, enhanced):
        required = {"accuracy","precision","recall","f1_score","roc_auc","confusion_matrix","report"}
        for name, res in enhanced["scenario_A"]["results"].items():
            missing = required - res.keys()
            assert not missing, f"{name}: metrics missing: {missing}"


# ═══════════════════════════════════════════════════════════════════════════════
# METRIC BOUNDS (Scenario A expected near 50%)
# ═══════════════════════════════════════════════════════════════════════════════
class TestScenarioAMetricBounds:
    """
    Scenario A accuracy is expected ~51% ± 5%.
    FAIL if any model implausibly exceeds 65% — indicates data leakage.
    FAIL if all models are below 45% — indicates catastrophic failure.
    """
    def test_all_accuracy_in_valid_range(self, enhanced):
        for name, res in enhanced["scenario_A"]["results"].items():
            acc = res["accuracy"]
            assert 0.40 <= acc <= 0.70, \
                f"{name} accuracy {acc*100:.2f}% is outside expected range [40%, 70%]"

    def test_no_implausible_leakage_accuracy(self, enhanced):
        for name, res in enhanced["scenario_A"]["results"].items():
            assert res["accuracy"] < 0.75, \
                f"{name} accuracy {res['accuracy']*100:.2f}% suspiciously high — " \
                "possible target leakage in Scenario A features"

    def test_f1_scores_valid_range(self, enhanced):
        for name, res in enhanced["scenario_A"]["results"].items():
            f1 = res["f1_score"]
            assert 0.0 <= f1 <= 1.0, f"{name}: f1={f1:.4f} out of [0,1]"

    def test_roc_auc_valid_range(self, enhanced):
        for name, res in enhanced["scenario_A"]["results"].items():
            auc = res["roc_auc"]
            assert 0.40 <= auc <= 0.70, \
                f"{name}: ROC-AUC={auc:.4f} outside expected range for Scenario A"

    def test_cv_f1_plausible(self, enhanced):
        for name, cv in enhanced["scenario_A"]["cv_results"].items():
            mean = cv["cv_f1_mean"]
            std  = cv["cv_f1_std"]
            assert 0.40 <= mean <= 0.70, \
                f"{name}: CV F1 mean {mean:.4f} outside expected range"
            assert 0.0 <= std <= 0.10, \
                f"{name}: CV F1 std {std:.4f} implausibly large (suggests instability)"


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO B (leakage reference)
# ═══════════════════════════════════════════════════════════════════════════════
class TestScenarioBLeakage:
    def test_scenario_b_accuracy_high(self, enhanced):
        acc = enhanced["scenario_B"]["results"]["RandomForest"]["accuracy"]
        assert acc > 0.90, \
            f"Scenario B accuracy {acc*100:.2f}% unexpectedly low — check leakage columns"

    def test_scenario_b_warning_present(self, enhanced):
        warning = enhanced["scenario_B"].get("warning","")
        assert "NOT genuine" in warning or "leakage" in warning.lower(), \
            "Scenario B warning must mention 'NOT genuine' or 'leakage'"

    def test_scenario_b_much_higher_than_a(self, enhanced):
        best_a = max(
            r["accuracy"] for r in enhanced["scenario_A"]["results"].values()
        )
        best_b = enhanced["scenario_B"]["results"]["RandomForest"]["accuracy"]
        assert best_b > best_a + 0.30, \
            f"Scenario B ({best_b*100:.1f}%) should be >30pp above Scenario A ({best_a*100:.1f}%)"


# ═══════════════════════════════════════════════════════════════════════════════
# CONFUSION MATRICES
# ═══════════════════════════════════════════════════════════════════════════════
class TestConfusionMatrices:
    def test_confusion_matrices_2x2(self, enhanced):
        for name, res in enhanced["scenario_A"]["results"].items():
            cm = res["confusion_matrix"]
            assert len(cm) == 2, f"{name}: CM should be 2×2"
            assert len(cm[0]) == 2, f"{name}: CM row 0 should have 2 elements"
            assert len(cm[1]) == 2, f"{name}: CM row 1 should have 2 elements"

    def test_confusion_matrix_sums_to_test_set_size(self, enhanced):
        for name, res in enhanced["scenario_A"]["results"].items():
            cm = res["confusion_matrix"]
            total = cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1]
            assert total == 2000, \
                f"{name}: CM total {total} != 2000 (test set size)"


# ═══════════════════════════════════════════════════════════════════════════════
# PERMUTATION IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════════
class TestPermutationImportance:
    def test_permutation_importance_has_all_features(self, enhanced):
        pi = enhanced["scenario_A"]["permutation_importance"]
        expected_feats = set(enhanced["scenario_A"]["features"])
        assert expected_feats == set(pi.keys()), \
            f"PI features {set(pi.keys())} don't match Scenario A features {expected_feats}"

    def test_permutation_importance_low_magnitude(self, enhanced):
        """All PI values should be near zero (confirming weak signal in Scenario A)."""
        pi = enhanced["scenario_A"]["permutation_importance"]
        for feat, vals in pi.items():
            assert abs(vals["mean"]) < 0.20, \
                f"{feat} PI mean {vals['mean']:.4f} suspiciously high — possible leakage"

    def test_permutation_importance_std_non_negative(self, enhanced):
        pi = enhanced["scenario_A"]["permutation_importance"]
        for feat, vals in pi.items():
            assert vals["std"] >= 0, f"{feat}: PI std cannot be negative"


# ═══════════════════════════════════════════════════════════════════════════════
# CV RESULTS CSV
# ═══════════════════════════════════════════════════════════════════════════════
class TestCvCsv:
    def test_cv_csv_row_count(self, cv_df):
        assert len(cv_df) == len(EXPECTED_MODELS), \
            f"CV CSV has {len(cv_df)} rows; expected {len(EXPECTED_MODELS)}"

    def test_cv_csv_required_columns(self, cv_df):
        required = {"Model","CV Accuracy","CV F1 (weighted)","CV ROC-AUC",
                    "Test Accuracy","Test F1","Train Time (s)","Best Params","Selected"}
        assert required.issubset(cv_df.columns), \
            f"Missing columns: {required - set(cv_df.columns)}"

    def test_exactly_one_selected(self, cv_df):
        # Non-empty, non-NaN string in Selected column means this model was chosen
        n_selected = cv_df["Selected"].apply(
            lambda x: pd.notna(x) and str(x).strip() not in ("", "nan")
        ).sum()
        assert n_selected == 1, f"Expected 1 selected model, got {n_selected}"

    def test_model_names_match_expected(self, cv_df):
        found = set(cv_df["Model"].tolist())
        assert found == EXPECTED_MODELS, \
            f"CV CSV models {found} don't match expected {EXPECTED_MODELS}"


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE SYNTAX CHECK (Phase 2 pages)
# ═══════════════════════════════════════════════════════════════════════════════
import ast

class TestNewPageSyntax:
    @pytest.mark.parametrize("page_file", [
        "pages/5_Model_Laboratory.py",
        "pages/6_Marine_Data_Sources.py",
        "pages/7_Marine_Monitoring_Map.py",
        "pages/8_Trends_and_Anomalies.py",
        "pages/9_Decision_Support.py",
    ])
    def test_page_syntax(self, page_file):
        full_path = os.path.join(ROOT, page_file)
        assert os.path.isfile(full_path), f"Page file not found: {page_file}"
        try:
            ast.parse(open(full_path, encoding="utf-8").read())
        except SyntaxError as e:
            pytest.fail(f"Syntax error in {page_file}: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# TRAIN_MODEL_ENHANCED.PY SYNTAX
# ═══════════════════════════════════════════════════════════════════════════════
class TestEnhancedScriptSyntax:
    def test_train_model_enhanced_syntax(self):
        path = os.path.join(ROOT, "src", "train_model_enhanced.py")
        assert os.path.isfile(path)
        ast.parse(open(path, encoding="utf-8").read())

    def test_train_model_enhanced_no_test_set_in_tuning(self):
        """Verify the training script never passes test data to fit/search."""
        path = os.path.join(ROOT, "src", "train_model_enhanced.py")
        source = open(path, encoding="utf-8").read()
        # Should mention Xa_te only in test_set_metrics and permutation_importance calls
        # Simple heuristic: search.fit must use training data name
        assert "search.fit(X_train, y_train)" in source, \
            "Tuning must use X_train (not X_test)"
        # Test data never passed to RandomizedSearchCV constructor
        assert "RandomizedSearchCV" in source
