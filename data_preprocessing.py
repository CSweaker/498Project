"""Stage 1: leakage-safe preprocessing for CIC-IDS2017 flow CSV files.

The important fix is the order of operations:
1. clean and encode labels;
2. split raw features into train/validation/test;
3. fit StandardScaler on the training split only;
4. transform validation/test with the training-fitted scaler.
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from utils import ensure_dir, set_seed

DEFAULT_DROP_COLUMNS = {
    "Label",
    "label",
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Source Port",
    "Destination Port",
    "Protocol",
    "Timestamp",
    "SimillarHTTP",
    "Inbound",
}


def load_dataset(data_dir: str) -> pd.DataFrame:
    """Load and concatenate all CSV files from data_dir."""
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}. Put CIC-IDS2017 CSV files in this folder."
        )

    frames = []
    for file_path in files:
        print(f"Loading: {file_path}")
        df = pd.read_csv(file_path, encoding="utf-8", low_memory=False)
        df.columns = df.columns.str.strip()
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    print(f"Total rows loaded: {len(data):,}")
    return data


def clean_data(df: pd.DataFrame, label_col: str = "Label") -> pd.DataFrame:
    """Remove duplicate, missing, and infinite rows."""
    if label_col not in df.columns:
        raise KeyError(f"Expected label column {label_col!r}. Available columns: {list(df.columns)[:10]}...")
    before = len(df)
    df = df.drop_duplicates()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[label_col])
    print(f"Rows after duplicate/label cleaning: {len(df):,} (removed {before - len(df):,})")
    return df


def encode_labels(df: pd.DataFrame, label_col: str = "Label", binary: bool = True) -> Tuple[pd.DataFrame, dict]:
    """Encode labels as binary benign/malicious or multiclass integers."""
    df = df.copy()
    raw = df[label_col].astype(str).str.strip()
    if binary:
        df["label"] = np.where(raw.str.upper().eq("BENIGN"), 0, 1).astype(np.int64)
        label_map = {"BENIGN": 0, "MALICIOUS": 1}
    else:
        encoder = LabelEncoder()
        df["label"] = encoder.fit_transform(raw).astype(np.int64)
        label_map = {cls: int(i) for i, cls in enumerate(encoder.classes_)}
    print("Label counts:")
    print(df["label"].value_counts().sort_index())
    return df, label_map


def select_numeric_features(
    df: pd.DataFrame, label_col: str = "Label", extra_drop: List[str] | None = None
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """Select numeric model features and drop invalid rows after numeric conversion."""
    drop_cols = set(DEFAULT_DROP_COLUMNS)
    if extra_drop:
        drop_cols.update(extra_drop)

    feature_cols = [c for c in df.columns if c not in drop_cols]
    X_df = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    valid_mask = X_df.notna().all(axis=1).to_numpy()
    X_df = X_df.loc[valid_mask]
    y = df.loc[valid_mask, "label"].to_numpy(dtype=np.int64)

    print(f"Features used: {len(feature_cols)}")
    print(f"Rows after numeric feature cleaning: {len(X_df):,}")
    return X_df.to_numpy(dtype=np.float32), y, feature_cols, valid_mask


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
):
    """Create train/validation/test splits with stratification."""
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    val_relative = val_size / (1.0 - test_size)
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_raw,
        y_train,
        test_size=val_relative,
        random_state=seed,
        stratify=y_train,
    )
    return X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test


def fit_scale_and_save(
    X_train_raw: np.ndarray,
    X_val_raw: np.ndarray,
    X_test_raw: np.ndarray,
    out_data_dir: str,
):
    """Fit the scaler on train only, transform all splits, and save arrays."""
    out_data = ensure_dir(out_data_dir)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_val = scaler.transform(X_val_raw).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)

    np.save(out_data / "X_train.npy", X_train)
    np.save(out_data / "X_val.npy", X_val)
    np.save(out_data / "X_test.npy", X_test)
    return X_train, X_val, X_test, scaler


def maybe_sample(df: pd.DataFrame, sample_size: int | None, seed: int) -> pd.DataFrame:
    """Optional development-time stratified sample to speed up local tests."""
    if sample_size is None or sample_size <= 0 or sample_size >= len(df):
        return df
    return (
        df.groupby("label", group_keys=False)
        .apply(lambda g: g.sample(max(1, int(sample_size * len(g) / len(df))), random_state=seed))
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-data-dir", default="data")
    parser.add_argument("--out-artifact-dir", default="artifacts")
    parser.add_argument("--label-col", default="Label")
    parser.add_argument("--binary", action="store_true", default=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=0, help="Optional smaller stratified sample for debugging.")
    args = parser.parse_args()

    set_seed(args.seed)
    out_data = ensure_dir(args.out_data_dir)
    out_artifacts = ensure_dir(args.out_artifact_dir)

    df = load_dataset(args.data_dir)
    df = clean_data(df, label_col=args.label_col)
    df, label_map = encode_labels(df, label_col=args.label_col, binary=args.binary)
    df = maybe_sample(df, args.sample_size if args.sample_size > 0 else None, args.seed)

    X_raw, y, feature_cols, _ = select_numeric_features(df, label_col=args.label_col)
    splits = stratified_split(X_raw, y, test_size=args.test_size, val_size=args.val_size, seed=args.seed)
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = splits
    X_train, X_val, X_test, scaler = fit_scale_and_save(X_train_raw, X_val_raw, X_test_raw, out_data)

    np.save(out_data / "y_train.npy", y_train)
    np.save(out_data / "y_val.npy", y_val)
    np.save(out_data / "y_test.npy", y_test)

    class_counts = np.bincount(y_train.astype(int))
    class_weights = (class_counts.sum() / np.maximum(class_counts, 1) / len(class_counts)).astype(np.float32)

    # Bounds are in standardized feature space and are useful for bounded adversarial attacks.
    lower = np.percentile(X_train, 0.5, axis=0).astype(np.float32)
    upper = np.percentile(X_train, 99.5, axis=0).astype(np.float32)
    preprocess = {
        "scaler": scaler,
        "feature_cols": feature_cols,
        "label_map": label_map,
        "class_counts": class_counts.tolist(),
        "class_weights": class_weights.tolist(),
        "feature_bounds": {"lower": lower, "upper": upper},
        "seed": args.seed,
        "test_size": args.test_size,
        "val_size": args.val_size,
    }
    joblib.dump(preprocess, out_artifacts / "preprocess.joblib")

    print(f"Train/Val/Test shapes: {X_train.shape}, {X_val.shape}, {X_test.shape}")
    print(f"Saved arrays to: {Path(args.out_data_dir).resolve()}")
    print(f"Saved preprocessing artifact to: {(out_artifacts / 'preprocess.joblib').resolve()}")


if __name__ == "__main__":
    main()
