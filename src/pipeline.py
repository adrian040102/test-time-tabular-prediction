"""
Pipeline orchestration: Dataset → LeakageGuard → Method → Model → Evaluate → CSV.

Central entry point for running a single experiment configuration.
Supports both classification and regression tasks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.data.base import TabularDataset, LeakageGuard
from src.methods.base import (
    MethodContext,
    RowProvenance,
    TestTimeMethod,
    TransformResult,
    _propagate_context,
    _propagate_seed,
)
from src.models.base import ModelWrapper


# Reserved result-row columns. Scalar metadata with a colliding key is
# namespaced as ``meta_<key>`` when merged into an experiment row. This also
# protects ``fold`` and ``error``, which the worker adds separately.
RESERVED_RESULT_COLUMNS = frozenset({
    "dataset", "method", "model", "seed", "fold", "task_type",
    "accuracy", "auc_roc", "f1", "log_loss", "rmse", "mae", "r2", "error",
})


@dataclass
class ExperimentResult:
    """Container for a single experiment run's results.

    Holds both classification metrics (accuracy, auc_roc, f1, log_loss) and
    regression metrics (rmse, mae, r2).  Unused metrics default to NaN.
    """

    dataset: str
    method: str
    model: str
    seed: int
    task_type: str = "classification"
    # Classification metrics
    accuracy: float = float("nan")
    auc_roc: float = float("nan")
    f1: float = float("nan")
    log_loss: float = float("nan")
    # Regression metrics
    rmse: float = float("nan")
    mae: float = float("nan")
    r2: float = float("nan")
    # Timing
    train_time: float = 0.0
    method_time: float = 0.0
    total_time: float = 0.0
    # Sizes
    n_train: int = 0
    n_test: int = 0
    n_features_in: int = 0
    n_features_out: int = 0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("metadata")
        # Only include scalar/string metadata values in the CSV-friendly
        # dict.  Large array values (e.g. y_train_augmented) are stripped
        # to avoid corrupting the results DataFrame / CSV output.
        for k, v in self.metadata.items():
            if isinstance(v, np.ndarray) or (hasattr(v, '__len__') and not isinstance(v, str) and len(v) > 10):
                continue  # skip large non-scalar values
            # Never let metadata overwrite a reserved metric or identity column.
            # Namespace a collision while preserving its diagnostic value.
            if k in RESERVED_RESULT_COLUMNS:
                k = f"meta_{k}"
            d[k] = v
        return d

    def summary_line(self) -> str:
        if self.task_type == "regression":
            return (
                f"{self.method:<35} | {self.model:<12} | "
                f"RMSE={self.rmse:.4f} | MAE={self.mae:.4f} | "
                f"R2={self.r2:.4f} | "
                f"MethodT={self.method_time:.1f}s | TrainT={self.train_time:.1f}s"
            )
        return (
            f"{self.method:<35} | {self.model:<12} | "
            f"Acc={self.accuracy:.4f} | AUC={self.auc_roc:.4f} | "
            f"F1={self.f1:.4f} | "
            f"MethodT={self.method_time:.1f}s | TrainT={self.train_time:.1f}s"
        )


def _prepare_method_frames(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    categorical_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Make declared categoricals visible to methods that inspect dtypes.

    Numeric-stored categories are cast to object without changing their
    values. The dataset declaration, not dtype inference, is authoritative.
    """
    X_train_out = X_train.copy()
    X_test_out = X_test.copy()
    missing = [
        col for col in categorical_features
        if col not in X_train_out.columns or col not in X_test_out.columns
    ]
    if missing:
        raise ValueError(
            "Dataset categorical declaration references missing columns: "
            f"{missing}"
        )
    for col in categorical_features:
        X_train_out[col] = X_train_out[col].astype("object")
        X_test_out[col] = X_test_out[col].astype("object")
    return X_train_out, X_test_out


def _resolve_row_provenance(
    result: TransformResult,
    *,
    n_input_train: int,
    n_input_test: int,
) -> RowProvenance:
    """Validate explicit provenance or provide identity for unchanged rows."""
    n_output_train = len(result.X_train)
    if result.row_provenance_train is None:
        if n_output_train != n_input_train:
            raise ValueError(
                "Method changed training-row cardinality without declaring "
                "row_provenance_train"
            )
        provenance = RowProvenance.identity_train(n_input_train)
    else:
        provenance = result.row_provenance_train
    provenance.validate(
        n_rows=n_output_train,
        n_train=n_input_train,
        n_test=n_input_test,
    )
    if len(result.X_test) != n_input_test:
        raise ValueError(
            "Methods must preserve test-row cardinality and order: "
            f"{len(result.X_test)} != {n_input_test}"
        )
    return provenance


def _model_categorical_declaration(
    dataset_cats: list[str],
    method_cats: list[str] | None,
    feature_names: list[str],
) -> list[str]:
    """Ordered union of dataset-declared and method-declared categoricals."""
    available = set(feature_names)
    ordered: list[str] = []
    for col in [*dataset_cats, *(method_cats or [])]:
        if col in available and col not in ordered:
            ordered.append(col)
    return ordered


def _restore_raw_categoricals(
    X_train_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
    *,
    context: MethodContext,
    provenance: RowProvenance,
) -> None:
    """Restore raw declared cats using train/test row provenance in-place."""
    for col in context.categorical_features:
        if col not in X_train_df.columns or col not in X_test_df.columns:
            continue
        train_values = context.raw_X_train[col].to_numpy(dtype=object)
        test_values = context.raw_X_test[col].to_numpy(dtype=object)
        restored = np.empty(len(provenance.source_split), dtype=object)
        from_train = provenance.source_split == "train"
        from_test = provenance.source_split == "test"
        restored[from_train] = train_values[provenance.source_position[from_train]]
        restored[from_test] = test_values[provenance.source_position[from_test]]
        X_train_df[col] = restored
        X_test_df[col] = test_values


def run_single_experiment(
    dataset: TabularDataset,
    method: TestTimeMethod,
    model: ModelWrapper,
    seed: int = 42,
    bootstrap_train: bool = True,
    log_transform_target: bool = False,
    capture_path: Path | str | None = None,
    capture_fold: int | None = None,
    capture_method_key: str | None = None,
) -> ExperimentResult:
    """
    Run a single experiment: apply method → train model → evaluate.

    The dataset's y_test is separated by LeakageGuard before anything
    else happens, ensuring the pipeline never has access.

    Args:
        dataset: Loaded TabularDataset (y_test will be removed by guard).
        method: TestTimeMethod to apply.
        model: ModelWrapper to train and predict.
        capture_method_key: Canonical registry key to store in a capture.
            Falls back to the method display name for non-registry callers.
        seed: Random seed for reproducibility.
        bootstrap_train: If True, bootstrap-resample the training set using
            ``seed``.  This creates genuine variance across seeds even when
            the underlying data is fixed (real-world datasets with semantic
            train/test splits).  The test set is always kept intact.

    Returns:
        ExperimentResult with all metrics.
    """
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    t_start = time.time()
    task_type = dataset.task_type
    _propagate_seed(method, seed)

    # --- Step 1: Separate y_test ---
    guard = LeakageGuard(dataset)

    # --- Step 1b: Bootstrap-resample training data ---
    X_train = dataset.X_train
    y_train = dataset.y_train

    if bootstrap_train:
        n = len(X_train)
        idx = rng.choice(n, size=n, replace=True)
        X_train = X_train.iloc[idx].reset_index(drop=True)
        y_train = y_train[idx]

    # --- Step 1c: Optional log-transform for skewed regression targets ---
    # For datasets such as tschalzev_svpc, log1p(y) handles sparse targets with
    # extreme outliers and valid zero values. Predictions are
    # inverse-transformed with expm1 before evaluation.
    _log_target = False
    if log_transform_target and task_type == "regression":
        if (y_train >= 0).all():
            y_train = np.log1p(y_train)
            _log_target = True

    # Dataset declarations are authoritative even when a categorical column
    # is stored numerically. Object dtype makes that declaration visible to
    # methods that use dtype-based helper functions.
    X_method_train, X_method_test = _prepare_method_frames(
        X_train, dataset.X_test, list(dataset.cat_features)
    )
    model_supports_sample_weight = bool(
        getattr(model, "supports_sample_weight", True)
    )
    method_context = MethodContext(
        task_type=task_type,
        categorical_features=tuple(dataset.cat_features),
        numeric_features=tuple(dataset.num_features),
        raw_X_train=X_method_train.copy(deep=True),
        raw_X_test=X_method_test.copy(deep=True),
        model_name=model.name,
        model_supports_sample_weight=model_supports_sample_weight,
    )
    _propagate_context(method, method_context)

    # --- Step 2: Apply test-time method ---
    t_method_start = time.time()
    result: TransformResult = method.fit_transform(
        X_method_train, X_method_test, y_train
    )
    method_time = time.time() - t_method_start
    row_provenance = _resolve_row_provenance(
        result,
        n_input_train=len(X_method_train),
        n_input_test=len(X_method_test),
    )

    # --- Step 2b: Universal post-method preprocessing ---
    # Impute NaN values (needed for Ridge, LogReg, PCA-based features and
    # any dataset with missing values like shifts_weather).
    # Fit imputer on train, transform both. No test-label leakage.
    #
    # Models that opt in via ``prefers_raw_features = True`` (e.g.
    # LightGBMTabArenaWrapper) skip imputation and standardization:
    # they handle NaN natively and tree splits are scale-invariant. The
    # universal imputer would also destroy LightGBM's ability to learn the
    # is-NaN signal as a split.
    prefers_raw = bool(getattr(model, "prefers_raw_features", False))

    X_tr_arr = np.asarray(result.X_train, dtype=float)
    X_te_arr = np.asarray(result.X_test, dtype=float)

    if not prefers_raw:
        if np.isnan(X_tr_arr).any() or np.isnan(X_te_arr).any():
            # keep_empty_features=True: an all-NaN column is filled (with 0)
            # and retained, so the column count still matches result.feature_names
            # at the DataFrame rebuild below. Without it the imputer silently
            # drops empty columns -> shape mismatch.
            imputer = SimpleImputer(strategy="median", keep_empty_features=True)
            X_tr_arr = imputer.fit_transform(X_tr_arr)
            X_te_arr = imputer.transform(X_te_arr)

        # Replace any remaining inf values (can arise from log-ratios etc.)
        X_tr_arr = np.nan_to_num(X_tr_arr, nan=0.0, posinf=1e10, neginf=-1e10)
        X_te_arr = np.nan_to_num(X_te_arr, nan=0.0, posinf=1e10, neginf=-1e10)

        # Standardize features for scale-sensitive models such as Ridge and
        # logistic regression. Tree-based models are scale-invariant to this
        # transformation.
        #
        # Methods that perform their own scaling (e.g. M5 JointScaling) set
        # result.skip_pipeline_scaler=True to prevent this train-only scaler
        # from overwriting their joint statistics.  Without this opt-out,
        # joint scaling is a no-op for linear models: composing two linear
        # transforms (joint then train-only) is equivalent to train-only alone.
        if not result.skip_pipeline_scaler:
            scaler = StandardScaler()
            X_tr_arr = scaler.fit_transform(X_tr_arr)
            X_te_arr = scaler.transform(X_te_arr)
    else:
        # Raw-feature models (LightGBM bag8 and TabPFN) handle NaN values
        # directly, so this branch skips imputation and scaling. Infinite values
        # from operations such as log ratios are invalid model inputs and are
        # replaced while NaN values are preserved.
        X_tr_arr = np.nan_to_num(X_tr_arr, nan=np.nan, posinf=1e10, neginf=-1e10)
        X_te_arr = np.nan_to_num(X_te_arr, nan=np.nan, posinf=1e10, neginf=-1e10)

    # --- Step 2c: Handle augmented y_train from methods that expand the
    # training set (M7 PseudoLabelSelfTraining, M17 TDDA).  These methods
    # add augmented rows to X_train and return the corresponding y values
    # via result.y_train. The compatibility path also checks metadata for
    # "y_train_augmented" as a fallback.
    y_for_model = y_train
    y_aug = None
    if result.y_train is not None and len(result.y_train) == X_tr_arr.shape[0]:
        y_aug = result.y_train
    elif result.metadata and "y_train_augmented" in result.metadata:
        # Compatibility fallback for previously generated method output.
        candidate = result.metadata["y_train_augmented"]
        if len(candidate) == X_tr_arr.shape[0]:
            y_aug = candidate

    if y_aug is not None:
        y_for_model = np.asarray(y_aug)

    if len(y_for_model) != X_tr_arr.shape[0]:
        raise ValueError(
            f"Training label length ({len(y_for_model)}) does not match "
            f"transformed training rows ({X_tr_arr.shape[0]})"
        )

    if (
        result.sample_weights_train is not None
        and len(result.sample_weights_train) != X_tr_arr.shape[0]
    ):
        raise ValueError(
            "sample_weights_train length "
            f"({len(result.sample_weights_train)}) does not match transformed "
            f"training rows ({X_tr_arr.shape[0]})"
        )

    sample_weight_emitted = result.sample_weights_train is not None
    sample_weight_for_model = result.sample_weights_train
    if sample_weight_emitted and not model_supports_sample_weight:
        # TabICL and TabPFN do not accept sample weights. Some methods return
        # weights together with transformed features or added training rows.
        # Keep those other outputs and omit only the unsupported weights. Pure
        # weight-only methods are excluded when sweep manifests are built.
        sample_weight_for_model = None

    if not sample_weight_emitted:
        sample_weight_policy = "not_emitted"
    elif sample_weight_for_model is None:
        sample_weight_policy = "ignored_model_unsupported"
    else:
        sample_weight_policy = "applied"

    # --- Step 3: Train model ---
    # Wrap as DataFrames with feature names to avoid sklearn/LightGBM warnings
    X_train_df = pd.DataFrame(X_tr_arr, columns=result.feature_names)
    X_test_df = pd.DataFrame(X_te_arr, columns=result.feature_names)

    fit_kwargs: dict = {}
    cat_decl = _model_categorical_declaration(
        list(dataset.cat_features),
        result.categorical_features,
        list(result.feature_names),
    )
    if prefers_raw:
        _restore_raw_categoricals(
            X_train_df,
            X_test_df,
            context=method_context,
            provenance=row_provenance,
        )
        if cat_decl:
            fit_kwargs["categorical_feature"] = list(cat_decl)

    # Check compatibility for wrappers that reject method-generated scaled
    # coordinates.
    if getattr(model, "refuses_prescaled_input", False) and result.skip_pipeline_scaler:
        fit_kwargs["_was_prescaled"] = True

    # Capture the exact DataFrames and aligned arrays immediately before fit.
    if capture_path is not None:
        from src.capture import CaptureKey, build_artifact, save_artifact
        artifact = build_artifact(
            key=CaptureKey(
                dataset=dataset.name,
                method=(
                    capture_method_key
                    if capture_method_key is not None
                    else method.name if hasattr(method, "name") else str(method)
                ),
                model=model.name,
                seed=int(seed),
                fold=capture_fold,
            ),
            task_type=task_type,
            X_train_pre=method_context.raw_X_train,
            X_test_pre=method_context.raw_X_test,
            y_train_pre=y_train,
            cat_features=list(dataset.cat_features),
            num_features=list(dataset.num_features),
            X_train_post=result.X_train,
            X_test_post=result.X_test,
            feature_names_post=list(result.feature_names),
            y_train_post=result.y_train,
            sample_weights_post=result.sample_weights_train,
            method_metadata=result.metadata or {},
            skip_pipeline_scaler=bool(result.skip_pipeline_scaler),
            X_train_model_input=X_train_df,
            X_test_model_input=X_test_df,
            y_train_model_input=np.asarray(y_for_model),
            sample_weights_model_input=sample_weight_for_model,
            categorical_features_model_input=cat_decl,
            row_provenance_train=row_provenance,
            model_supports_sample_weight=model_supports_sample_weight,
        )
        save_artifact(capture_path, artifact)

    t_train_start = time.time()
    model.fit(
        X_train_df,
        y_for_model,
        sample_weight=sample_weight_for_model,
        **fit_kwargs,
    )
    train_time = time.time() - t_train_start

    # --- Step 4: Predict ---
    y_pred = model.predict(X_test_df)
    y_prob = model.predict_proba(X_test_df)  # None for regression

    # Invert the log transformation when it was applied to the target.
    if _log_target:
        y_pred = np.expm1(y_pred)

    # --- Step 5: Evaluate (only way to access y_test) ---
    metrics = guard.evaluate(y_pred, y_prob)

    total_time = time.time() - t_start

    # Build result with the appropriate metrics
    exp_metadata = dict(result.metadata or {})
    weight_metadata = {
        "sample_weight_emitted": bool(sample_weight_emitted),
        "sample_weight_applied": bool(sample_weight_for_model is not None),
        "sample_weight_policy": sample_weight_policy,
    }
    for key, value in weight_metadata.items():
        if key in exp_metadata and exp_metadata[key] != value:
            raise RuntimeError(
                f"Method metadata collision for reserved field {key!r}: "
                f"method={exp_metadata[key]!r}, pipeline={value!r}"
            )
        exp_metadata[key] = value
    for key, value in model.execution_metadata.items():
        if key in exp_metadata and exp_metadata[key] != value:
            raise RuntimeError(
                f"Model execution metadata collision for {key!r}: "
                f"method={exp_metadata[key]!r}, model={value!r}"
            )
        exp_metadata[key] = value

    exp = ExperimentResult(
        dataset=dataset.name,
        method=method.name if hasattr(method, "name") else str(method),
        model=model.name,
        seed=seed,
        task_type=task_type,
        train_time=train_time,
        method_time=method_time,
        total_time=total_time,
        n_train=dataset.n_train,
        n_test=guard.n_test,
        n_features_in=dataset.n_features,
        n_features_out=result.X_train.shape[1],
        metadata=exp_metadata,
    )

    if task_type == "regression":
        exp.rmse = metrics["rmse"]
        exp.mae = metrics["mae"]
        exp.r2 = metrics["r2"]
    else:
        exp.accuracy = metrics["accuracy"]
        exp.auc_roc = metrics.get("auc_roc", float("nan"))
        exp.f1 = metrics["f1"]
        exp.log_loss = metrics.get("log_loss", float("nan"))

    return exp


def run_sweep(
    datasets: list[TabularDataset],
    methods: list[TestTimeMethod],
    models: list[ModelWrapper],
    seeds: list[int] = None,
    results_dir: str | Path = "results",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run a full sweep over all dataset × method × model × seed combinations.

    Args:
        datasets: List of datasets to evaluate.
        methods: List of methods to compare.
        models: List of models to use.
        seeds: List of random seeds (default: [42]).
        results_dir: Directory to save CSV results.
        verbose: Print progress.

    Returns:
        DataFrame with all results.
    """
    if seeds is None:
        seeds = [42]

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    total = len(datasets) * len(methods) * len(models) * len(seeds)
    count = 0

    for dataset in datasets:
        for method in methods:
            for model in models:
                for seed in seeds:
                    count += 1
                    if verbose:
                        print(
                            f"[{count}/{total}] "
                            f"{dataset.name} | {method.name} | {model.name} | seed={seed}"
                        )

                    # LeakageGuard sets dataset.y_test = None, so each run
                    # needs its own copy to avoid corrupting subsequent runs.
                    try:
                        ds_copy = dataset.copy()
                        result = run_single_experiment(ds_copy, method, model, seed)
                        if verbose:
                            print(f"  → {result.summary_line()}")
                        all_results.append(result.to_dict())
                    except Exception as e:
                        print(f"  ✗ FAILED: {e}")
                        all_results.append({
                            "dataset": dataset.name,
                            "method": method.name,
                            "model": model.name,
                            "seed": seed,
                            "error": str(e),
                        })

    df = pd.DataFrame(all_results)

    # Save results
    csv_path = results_dir / "sweep_results.csv"
    df.to_csv(csv_path, index=False)
    if verbose:
        print(f"\nResults saved to {csv_path}")

    return df
