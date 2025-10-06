"""Utility script to train an occupancy regression model from CO₂ telemetry.

This script merges the UCI occupancy dataset with an optional local CSV, builds
feature-engineered data using only CO₂-derived statistics, runs a quick
cross-validation sweep across RandomForest and XGBoost regressors, then trains a
final pipeline and writes it to disk as a joblib artifact that the FastAPI app
can load.

Example usage (from the project/ directory):

    python train_occupancy_model.py \
        --local dataset_newyear.csv \
        --uci-train datatraining.txt \
        --uci-test1 datatest.txt \
        --uci-test2 datatest2.txt \
        --output models/occupancy_from_co2_model.h5

Adjust the filenames to match the files you have available. All paths are
resolved relative to the current working directory unless absolute paths are
supplied. The output file is a joblib dictionary with keys:
    • "pipeline": fitted sklearn Pipeline
    • "feature_cols": list of feature column names in training order
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

FEATURE_TEMPLATE: List[str] = [
    "co2",
    "co2_lag_1",
    "co2_lag_3",
    "co2_lag_5",
    "co2_roll_mean_5",
    "co2_roll_std_5",
    "co2_diff_1",
    "co2_pct_change_1",
    "hour",
    "dayofweek",
]


@dataclass
class DatasetConfig:
    local: Optional[Path]
    uci_train: Path
    uci_test1: Path
    uci_test2: Path


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train occupancy model from CO₂ data")
    parser.add_argument("--local", type=Path, default=None, help="Optional local CSV with columns including CO2/count")
    parser.add_argument("--uci-train", type=Path, default=Path("datatraining.txt"), help="UCI training CSV")
    parser.add_argument("--uci-test1", type=Path, default=Path("datatest.txt"), help="UCI test CSV 1")
    parser.add_argument("--uci-test2", type=Path, default=Path("datatest2.txt"), help="UCI test CSV 2")
    parser.add_argument("--output", type=Path, default=Path("models/occupancy_from_co2_model.h5"), help="Output joblib file")
    parser.add_argument("--test-size", type=float, default=0.2, help="Hold-out test fraction")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for splitting and models")
    return parser.parse_args(list(argv) if argv is not None else None)


def load_local_dataset(path: Optional[Path]) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    if not path.exists():
        raise FileNotFoundError(f"Local dataset not found: {path}")
    df = pd.read_csv(path)
    if "epoch" in df.columns:
        df["datetime"] = pd.to_datetime(df["epoch"], unit="s", errors="coerce")
    elif "date" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["datetime"] = pd.to_datetime(df.index, unit="s", errors="coerce")
    if "co2" in df.columns and "CO2" not in df.columns:
        df = df.rename(columns={"co2": "CO2"})
    if "count" in df.columns and "Occupancy" not in df.columns:
        df = df.rename(columns={"count": "Occupancy"})
    return df[[c for c in ["datetime", "CO2", "Occupancy"] if c in df.columns]].copy()


def load_uci_datasets(cfg: DatasetConfig) -> pd.DataFrame:
    frames = []
    for path in [cfg.uci_train, cfg.uci_test1, cfg.uci_test2]:
        if not path.exists():
            raise FileNotFoundError(f"Missing UCI dataset: {path}")
        frames.append(pd.read_csv(path))
    df = pd.concat(frames, ignore_index=True)
    if "date" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"], errors="coerce")
    return df[[c for c in ["datetime", "CO2", "Occupancy"] if c in df.columns]].copy()


def combine_datasets(cfg: DatasetConfig) -> pd.DataFrame:
    local_df = load_local_dataset(cfg.local)
    uci_df = load_uci_datasets(cfg)
    combined = pd.concat([uci_df, local_df], ignore_index=True, sort=False)
    combined = combined.sort_values("datetime").dropna(subset=["datetime", "CO2", "Occupancy"])
    combined = combined.reset_index(drop=True)
    combined = combined.astype({"CO2": float, "Occupancy": float})
    return combined


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    df = df.copy()
    df["co2"] = df["CO2"].astype(float)
    for lag in (1, 2, 3, 5, 10):
        df[f"co2_lag_{lag}"] = df["co2"].shift(lag)
    df["co2_roll_mean_5"] = df["co2"].rolling(window=5, min_periods=1).mean().shift(1)
    df["co2_roll_std_5"] = df["co2"].rolling(window=5, min_periods=1).std().fillna(0).shift(1)
    df["co2_diff_1"] = df["co2"].diff(1)
    df["co2_pct_change_1"] = df["co2"].pct_change(1)
    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df = df.dropna().reset_index(drop=True)
    feature_cols = [c for c in FEATURE_TEMPLATE if c in df.columns]
    X = df[feature_cols].astype(float)
    y = df["Occupancy"].astype(float)
    return X, y, feature_cols


def cross_validate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
) -> Tuple[str, Dict[str, Dict[str, float]]]:
    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            random_state=random_state,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            tree_method="hist",
            n_jobs=-1,
            verbosity=0,
        ),
    }
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    results: Dict[str, Dict[str, float]] = {}
    for name, model in models.items():
        r2_scores: List[float] = []
        rmse_scores: List[float] = []
        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)
            model.fit(X_tr_s, y_tr)
            y_pred = model.predict(X_val_s)
            r2_scores.append(r2_score(y_val, y_pred))
            rmse_scores.append(float(np.sqrt(mean_squared_error(y_val, y_pred))))
        results[name] = {
            "R2_mean": float(np.mean(r2_scores)),
            "RMSE_mean": float(np.mean(rmse_scores)),
        }
    best_model = max(results, key=lambda k: results[k]["R2_mean"])
    return best_model, results


def build_final_pipeline(best_name: str, random_state: int) -> Pipeline:
    if best_name == "RandomForest":
        model = RandomForestRegressor(
            n_estimators=500,
            max_depth=12,
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        model = XGBRegressor(
            n_estimators=800,
            learning_rate=0.04,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            tree_method="hist",
            n_jobs=-1,
            verbosity=0,
        )
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    cfg = DatasetConfig(
        local=args.local,
        uci_train=args.uci_train,
        uci_test1=args.uci_test1,
        uci_test2=args.uci_test2,
    )

    print("[Load] Combining datasets...")
    combined = combine_datasets(cfg)
    print(f"[Load] Combined rows: {combined.shape[0]}")

    print("[Features] Engineering CO₂ features...")
    X, y, feature_cols = engineer_features(combined)
    print(f"[Features] Using {len(feature_cols)} features: {feature_cols}")
    print(f"[Features] Samples after engineering: {X.shape[0]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        shuffle=True,
    )

    print("[Model] Running cross-validation sweep...")
    best_name, cv_results = cross_validate_models(X_train, y_train, args.random_state)
    print("[Model] CV summary (JSON):")
    print(json.dumps(cv_results, indent=2))
    print(f"[Model] Selected model: {best_name}")

    pipeline = build_final_pipeline(best_name, args.random_state)
    pipeline.fit(X_train, y_train)

    y_pred_test = pipeline.predict(X_test)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    print(f"[Eval] Hold-out R2: {test_r2:.4f}")
    print(f"[Eval] Hold-out RMSE: {test_rmse:.4f}")

    ensure_parent(args.output)
    artifact = {"pipeline": pipeline, "feature_cols": feature_cols, "cv_results": cv_results}
    joblib.dump(artifact, args.output)
    print(f"[Save] Wrote joblib artifact to {args.output}")


if __name__ == "__main__":
    main()
