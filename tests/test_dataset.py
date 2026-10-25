"""
test_dataset.py
End-to-end dataset integrity tests for Phase 5.
"""
import os
import pytest
import pandas as pd
import numpy as np

EXPECTED_ROWS    = 10_000
EXPECTED_COLS    = 9
TARGET_COL       = "is_safe"
SAFE_FEATS       = ["Temperature", "pH", "Salinity", "Turbidity"]
THRESHOLD_COLS   = ["Dissolved_Oxygen", "Nitrate", "Chlorophyll_a"]
ALL_FEATURES     = SAFE_FEATS + THRESHOLD_COLS
EXPECTED_COLUMNS = ["Sample_ID"] + ALL_FEATURES + [TARGET_COL]


class TestDatasetExists:
    def test_csv_file_present(self, root):
        path = os.path.join(root, "data", "marine_water_quality.csv")
        assert os.path.isfile(path), (
            f"Dataset not found at {path}. "
            "Run the dataset download step."
        )


class TestDatasetStructure:
    def test_row_count(self, dataset):
        assert len(dataset) == EXPECTED_ROWS, (
            f"Expected {EXPECTED_ROWS:,} rows, found {len(dataset):,}"
        )

    def test_column_count(self, dataset):
        assert dataset.shape[1] == EXPECTED_COLS, (
            f"Expected {EXPECTED_COLS} columns, found {dataset.shape[1]}: "
            f"{list(dataset.columns)}"
        )

    def test_expected_columns_present(self, dataset):
        for col in EXPECTED_COLUMNS:
            assert col in dataset.columns, f"Column '{col}' missing from dataset"

    def test_target_column_binary(self, dataset):
        unique_vals = set(dataset[TARGET_COL].unique())
        assert unique_vals == {0, 1}, (
            f"Target must be binary {{0,1}}, found {unique_vals}"
        )

    def test_feature_dtypes_numeric(self, dataset):
        for col in ALL_FEATURES:
            assert pd.api.types.is_numeric_dtype(dataset[col]), (
                f"Feature '{col}' should be numeric, got {dataset[col].dtype}"
            )


class TestDatasetQuality:
    def test_no_missing_values(self, dataset):
        null_counts = dataset.isnull().sum()
        total_nulls = null_counts.sum()
        assert total_nulls == 0, (
            f"Found {total_nulls} missing values:\n{null_counts[null_counts > 0]}"
        )

    def test_no_duplicate_rows(self, dataset):
        n_dupes = dataset.duplicated().sum()
        assert n_dupes == 0, f"Found {n_dupes} duplicate rows"

    def test_perfect_class_balance(self, dataset):
        n_safe   = int((dataset[TARGET_COL] == 1).sum())
        n_unsafe = int((dataset[TARGET_COL] == 0).sum())
        assert n_safe   == 5000, f"Expected 5000 Safe samples, got {n_safe}"
        assert n_unsafe == 5000, f"Expected 5000 Unsafe samples, got {n_unsafe}"


class TestFeatureRanges:
    """Verify dataset ranges match prediction module bounds used in Phase 4."""

    def test_temperature_range(self, dataset):
        assert dataset["Temperature"].min() >= 10.0
        assert dataset["Temperature"].max() <= 32.0

    def test_ph_range(self, dataset):
        assert dataset["pH"].min() >= 6.5
        assert dataset["pH"].max() <= 8.8

    def test_salinity_range(self, dataset):
        assert dataset["Salinity"].min() >= 30.0
        assert dataset["Salinity"].max() <= 40.0

    def test_turbidity_range(self, dataset):
        assert dataset["Turbidity"].min() >= 0.5
        assert dataset["Turbidity"].max() <= 50.0


class TestLeakageProperties:
    """
    The four Scenario A features must have near-zero Pearson correlation
    with the target. A genuine high correlation would indicate leakage.
    """
    LEAKAGE_THRESHOLD = 0.05  # |r| must be below this for safe features

    def test_temperature_low_correlation(self, dataset):
        r = abs(dataset["Temperature"].corr(dataset[TARGET_COL]))
        assert r < self.LEAKAGE_THRESHOLD, (
            f"Temperature Pearson r={r:.4f} >= {self.LEAKAGE_THRESHOLD}. "
            "Possible leakage!"
        )

    def test_ph_low_correlation(self, dataset):
        r = abs(dataset["pH"].corr(dataset[TARGET_COL]))
        assert r < self.LEAKAGE_THRESHOLD

    def test_salinity_low_correlation(self, dataset):
        r = abs(dataset["Salinity"].corr(dataset[TARGET_COL]))
        assert r < self.LEAKAGE_THRESHOLD

    def test_turbidity_low_correlation(self, dataset):
        r = abs(dataset["Turbidity"].corr(dataset[TARGET_COL]))
        assert r < self.LEAKAGE_THRESHOLD

    def test_threshold_cols_high_correlation(self, dataset):
        """
        Threshold-defining columns MUST have high correlation with target —
        this confirms they are the true label generators and must be excluded
        from Scenario A.
        """
        HIGH = 0.5
        for col in THRESHOLD_COLS:
            r = abs(dataset[col].corr(dataset[TARGET_COL]))
            assert r > HIGH, (
                f"{col} correlation r={r:.4f} is unexpectedly low. "
                "Expected high correlation as it defines the target."
            )
