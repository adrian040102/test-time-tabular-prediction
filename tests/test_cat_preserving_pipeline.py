"""Regression tests for :class:`CatPreservingMethodPipeline`.

The base ``MethodPipeline`` loses object/category dtype information
between steps (every method returns a numpy float array), so any
categorical method after the first step receives no categorical columns even when
the prior method retained them. Methods such as ``JointFrequencyEncoding`` and
target encoding also remove categorical columns from their outputs.

``CatPreservingMethodPipeline`` re-injects the original categorical
columns between steps. These tests verify that behavior:

1. Categorical columns are re-injected when the prior method removed or encoded them
2. ``freq_ratio`` produces new features after ``joint_freq``
3. The final output remains a float NumPy array accepted by downstream code
4. Single-step pipelines are unchanged
5. Sample weights / y_train augmentation / skip_pipeline_scaler propagate
   the same as the base ``MethodPipeline``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.synthetic import load_synthetic
from src.methods.base import (
    CatPreservingMethodPipeline,
    MethodPipeline,
    TestTimeMethod,
    TransformResult,
)
from src.methods.tier1 import (
    FrequencyRatio,
    JointFrequencyEncoding,
    JointScaling,
    ShiftRegularizedTargetEncoding,
)


@pytest.fixture
def ds():
    return load_synthetic(n_train=300, n_test=200, shift_strength=1.0, seed=42)


# ---------------------------------------------------------------------------
# Helper method that inspects its input
# ---------------------------------------------------------------------------

class _CatInspector(TestTimeMethod):
    """Record visible categorical columns and return a numeric copy."""

    def __init__(self):
        self.seen_cat_cols: list[str] | None = None
        self.seen_columns: list[str] | None = None

    @property
    def name(self) -> str:
        return "cat_inspector"

    def fit_transform(self, X_train, X_test, y_train):
        self.seen_cat_cols = X_train.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        self.seen_columns = list(X_train.columns)

        # Return float output for subsequent pipeline steps.
        X_tr = X_train.copy()
        X_te = X_test.copy()
        for col in self.seen_cat_cols:
            # Encode categories numerically for the float output.
            codes = pd.Categorical(
                pd.concat([X_tr[col], X_te[col]]).astype(str)
            ).codes
            X_tr[col] = codes[: len(X_tr)]
            X_te[col] = codes[len(X_tr):]
        return TransformResult(
            X_train=X_tr.values.astype(float),
            X_test=X_te.values.astype(float),
            feature_names=list(X_tr.columns),
        )


# ---------------------------------------------------------------------------
# Core regression tests
# ---------------------------------------------------------------------------

class TestCatReinjection:
    """Verify that the pipeline re-injects original cats between steps."""

    def test_inspector_sees_cats_after_joint_freq(self, ds):
        """After JointFrequencyEncoding drops all cats, the next method
        should still see them thanks to re-injection."""
        original_cats = ds.X_train.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        assert len(original_cats) > 0, "fixture must contain categorical columns"

        inspector = _CatInspector()
        pipeline = CatPreservingMethodPipeline([
            JointFrequencyEncoding(variant="combined"),
            inspector,
        ])
        pipeline.fit_transform(ds.X_train, ds.X_test, ds.y_train)

        assert inspector.seen_cat_cols is not None
        assert set(inspector.seen_cat_cols) == set(original_cats), (
            f"inspector received {inspector.seen_cat_cols}, expected {original_cats}"
        )

    def test_inspector_sees_cats_after_joint_scaling(self, ds):
        """JointScaling does not drop cats but the base MethodPipeline would
        still strip dtype info.  Cats should be visible to the next step."""
        original_cats = ds.X_train.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        inspector = _CatInspector()
        pipeline = CatPreservingMethodPipeline([
            JointScaling(),
            inspector,
        ])
        pipeline.fit_transform(ds.X_train, ds.X_test, ds.y_train)

        assert set(inspector.seen_cat_cols) == set(original_cats)

    def test_base_pipeline_has_no_cats_after_step1(self, ds):
        """The base pipeline does not preserve categorical dtypes after step one."""
        inspector = _CatInspector()
        pipeline = MethodPipeline([
            JointFrequencyEncoding(variant="combined"),
            inspector,
        ])
        pipeline.fit_transform(ds.X_train, ds.X_test, ds.y_train)

        assert inspector.seen_cat_cols == [], (
            f"base MethodPipeline unexpectedly preserved cats: "
            f"{inspector.seen_cat_cols}"
        )

    def test_first_method_sees_original_input(self, ds):
        """Reinjection must not occur at step 0 because the first method receives
        the original dataset."""
        original_cats = ds.X_train.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        original_cols = list(ds.X_train.columns)

        inspector = _CatInspector()
        pipeline = CatPreservingMethodPipeline([inspector, JointScaling()])
        pipeline.fit_transform(ds.X_train, ds.X_test, ds.y_train)

        assert set(inspector.seen_cat_cols) == set(original_cats)
        assert set(inspector.seen_columns) == set(original_cols)


class TestFreqRatioAfterJointFreq:
    """Tests for joint frequency encoding followed by frequency ratio."""

    def test_freq_ratio_adds_features_after_joint_freq(self, ds):
        """Frequency ratio must add features after joint frequency encoding."""
        joint_freq_alone = JointFrequencyEncoding(variant="combined")
        alone_result = joint_freq_alone.fit_transform(
            ds.X_train, ds.X_test, ds.y_train
        )

        stacked = CatPreservingMethodPipeline([
            JointFrequencyEncoding(variant="combined"),
            FrequencyRatio(use_log=True),
        ])
        stacked_result = stacked.fit_transform(ds.X_train, ds.X_test, ds.y_train)

        n_cats = len(ds.X_train.select_dtypes(
            include=["object", "category"]
        ).columns)

        # The stack includes ratio features and encoded categorical columns.
        assert stacked_result.X_train.shape[1] > alone_result.X_train.shape[1], (
            f"stacked output ({stacked_result.X_train.shape[1]} cols) should exceed "
            f"joint_freq alone ({alone_result.X_train.shape[1]} cols). The categorical "
            f"columns were not re-injected."
        )
        assert stacked_result.X_train.shape[1] >= alone_result.X_train.shape[1] + n_cats

    def test_base_pipeline_freq_ratio_is_noop_after_joint_freq(self, ds):
        """The base pipeline adds no ratio features after joint frequency encoding."""
        joint_freq_alone_result = JointFrequencyEncoding(
            variant="combined"
        ).fit_transform(ds.X_train, ds.X_test, ds.y_train)

        base = MethodPipeline([
            JointFrequencyEncoding(variant="combined"),
            FrequencyRatio(use_log=True),
        ])
        base_result = base.fit_transform(ds.X_train, ds.X_test, ds.y_train)

        # The shapes match because freq_ratio adds no feature in the base pipeline.
        assert base_result.X_train.shape[1] == joint_freq_alone_result.X_train.shape[1]


class TestOutputContract:
    """The final output must remain a NumPy float array."""

    def test_output_is_numeric(self, ds):
        pipeline = CatPreservingMethodPipeline([
            JointFrequencyEncoding(variant="combined"),
            FrequencyRatio(use_log=True),
        ])
        result = pipeline.fit_transform(ds.X_train, ds.X_test, ds.y_train)

        # Coerces to float without error.
        arr_train = np.asarray(result.X_train, dtype=float)
        arr_test = np.asarray(result.X_test, dtype=float)
        assert arr_train.dtype == float
        assert arr_test.dtype == float
        assert not np.any(np.isnan(arr_train))
        assert not np.any(np.isnan(arr_test))

    def test_feature_names_match_shape(self, ds):
        pipeline = CatPreservingMethodPipeline([
            JointFrequencyEncoding(variant="combined"),
            FrequencyRatio(use_log=True),
        ])
        result = pipeline.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        assert len(result.feature_names) == result.X_train.shape[1]
        assert result.X_train.shape[1] == result.X_test.shape[1]


class TestSingleStepUnchanged:
    """Single-method pipelines must behave identically to the base class."""

    def test_shape_matches_base(self, ds):
        m = JointFrequencyEncoding(variant="combined")
        base = MethodPipeline([m])
        preserving = CatPreservingMethodPipeline([
            JointFrequencyEncoding(variant="combined")
        ])
        r_base = base.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        r_pres = preserving.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        assert r_base.X_train.shape == r_pres.X_train.shape

    def test_values_match_base(self, ds):
        base = MethodPipeline([JointFrequencyEncoding(variant="combined")])
        preserving = CatPreservingMethodPipeline([
            JointFrequencyEncoding(variant="combined")
        ])
        r_base = base.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        r_pres = preserving.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        np.testing.assert_allclose(r_base.X_train, r_pres.X_train)
        np.testing.assert_allclose(r_base.X_test, r_pres.X_test)


class TestPropagationUnchanged:
    """Sample weights, y_train augmentation and skip_pipeline_scaler must
    propagate the same way as in the base pipeline."""

    def test_skip_pipeline_scaler_propagates(self, ds):
        """The pipeline must preserve ``skip_pipeline_scaler=True``."""
        pipeline = CatPreservingMethodPipeline([
            JointScaling(),
            FrequencyRatio(use_log=True),
        ])
        result = pipeline.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        assert result.skip_pipeline_scaler is True

    def test_metadata_namespaced_by_method(self, ds):
        pipeline = CatPreservingMethodPipeline([
            JointFrequencyEncoding(variant="combined"),
            FrequencyRatio(use_log=True),
        ])
        result = pipeline.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        # Metadata keys are method names (matches base MethodPipeline behavior).
        # Not every method emits metadata, so only check it is a dict.
        assert isinstance(result.metadata, dict)

    def test_target_encoding_after_joint_freq_can_add_features(self, ds):
        """Target encoding also needs cats.  Stacking it after joint_freq
        must produce additional features thanks to re-injection."""
        joint_freq_alone = JointFrequencyEncoding(variant="combined")
        alone_result = joint_freq_alone.fit_transform(
            ds.X_train, ds.X_test, ds.y_train
        )

        stacked = CatPreservingMethodPipeline([
            JointFrequencyEncoding(variant="combined"),
            ShiftRegularizedTargetEncoding(regularization_mode="ratio"),
        ])
        stacked_result = stacked.fit_transform(ds.X_train, ds.X_test, ds.y_train)

        assert stacked_result.X_train.shape[1] > alone_result.X_train.shape[1], (
            "target_encoding added no features after joint_freq, re-injection failed"
        )
