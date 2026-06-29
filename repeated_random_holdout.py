import math
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from finetune import (
    MATURITY_BINS,
    MATURITY_LABELS,
    THEORY_FEATURES,
    TargetDataset,
    _make_model,
    _path_safe_label,
    bsiv_abs_error_median,
    eval_loss,
    finetune_model,
    monthly_test_metrics,
    predict,
    prediction_rows,
    set_seed,
    weighted_mae_numpy,
)
from pactran_blr_score import align_pactran_finetune, score_checkpoint_blr_subsamples


DELTA_BINS = [round(x, 1) for x in np.linspace(-1.0, 0.0, 11)]
DELTA_LABELS = [
    f"{DELTA_BINS[i]:.1f}-{DELTA_BINS[i + 1]:.1f}"
    for i in range(len(DELTA_BINS) - 1)
]


def add_holdout_bucket(
    df: pd.DataFrame,
    bucket_type: str,
    maturity_bins: Optional[list[float]] = None,
    maturity_labels: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, list[str]]:
    bucket_type = str(bucket_type).lower()
    out = df.copy()

    if bucket_type == "maturity":
        labels = MATURITY_LABELS if maturity_labels is None else maturity_labels
        bins = MATURITY_BINS if maturity_bins is None else maturity_bins
        out["maturity_bucket"] = pd.cut(
            out["tau"],
            bins=bins,
            labels=labels,
            include_lowest=True,
        )
        return out.dropna(subset=["maturity_bucket"]), list(labels)

    if bucket_type == "delta":
        out["maturity_bucket"] = pd.cut(
            out["delta"],
            bins=DELTA_BINS,
            labels=DELTA_LABELS,
            include_lowest=True,
        )
        return out.dropna(subset=["maturity_bucket"]), list(DELTA_LABELS)

    raise ValueError("bucket_type must be one of: maturity, delta.")


def _prepare_holdout_data(
    heston_dataset: pd.DataFrame,
    bucket_type: str,
    maturity_bins: Optional[list[float]] = None,
    maturity_labels: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, list[str]]:
    required = [
        "date",
        "mid_price",
        "delta",
        "tau",
        *THEORY_FEATURES["BS"],
        *THEORY_FEATURES["HESTON"],
    ]
    if "cp_flag" in heston_dataset.columns:
        required.append("cp_flag")

    missing = sorted(set(required) - set(heston_dataset.columns))
    if missing:
        raise ValueError(f"Holdout dataset is missing required columns: {missing}")

    df = heston_dataset.copy()
    df["_holdout_source_index"] = np.arange(len(df))
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=required).sort_values("date").reset_index(drop=True)
    if "cp_flag" in df.columns:
        df = df.loc[df["cp_flag"].astype(str).str.upper() == "P"].copy()
    df, bucket_labels = add_holdout_bucket(
        df,
        bucket_type=bucket_type,
        maturity_bins=maturity_bins,
        maturity_labels=maturity_labels,
    )
    df["_date"] = df["date"].dt.normalize()
    return df.reset_index(drop=True), bucket_labels


def _date_str(value) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _sample_temporal_holdout(
    bucket_df: pd.DataFrame,
    bucket_label: str,
    seed: int,
    train_size: int,
    test_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    total_size = int(train_size) + int(test_size)
    if total_size <= 1:
        raise ValueError("train_size + test_size must be greater than 1.")
    if len(bucket_df) < total_size:
        raise ValueError(
            f"Not enough samples for bucket={bucket_label}: "
            f"need {total_size}, got {len(bucket_df)}."
        )

    sampled = bucket_df.sample(
        n=total_size,
        replace=False,
        random_state=int(seed),
    ).sort_values(["date", "_holdout_source_index"]).reset_index(drop=True)

    boundary_date = sampled.iloc[int(train_size) - 1]["_date"]
    train_val = sampled.loc[sampled["_date"] <= boundary_date].copy()
    test = sampled.loc[sampled["_date"] > boundary_date].copy()
    if len(train_val) < 2 or len(test) == 0:
        raise ValueError(
            f"Empty temporal holdout for bucket={bucket_label}: "
            f"n_trainval={len(train_val)}, n_test={len(test)}."
        )
    if train_val["date"].max() >= test["date"].min():
        raise ValueError("Temporal leakage: train dates must be before test dates.")

    metadata = {
        "holdout_seed": int(seed),
        "maturity_bucket": str(bucket_label),
        "train_start_date": _date_str(train_val["date"].min()),
        "train_end_date": _date_str(train_val["date"].max()),
        "test_start_date": _date_str(test["date"].min()),
        "test_end_date": _date_str(test["date"].max()),
        "n_sampled": int(len(sampled)),
        "n_trainval": int(len(train_val)),
        "n_test": int(len(test)),
    }
    return train_val.reset_index(drop=True), test.reset_index(drop=True), metadata


def _sample_universal_temporal_holdout(
    data: pd.DataFrame,
    bucket_labels: list[str],
    seed: int,
    train_size: int,
    test_size: int,
    bucket_type: str,
) -> dict:
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be positive.")

    rng = np.random.default_rng(int(seed))
    train_frames = []
    test_frames = []
    bucket_rows = []

    for bucket_label in bucket_labels:
        bucket_df = data.loc[
            data["maturity_bucket"].astype(str).eq(str(bucket_label))
        ].copy()
        if len(bucket_df) < train_size:
            raise ValueError(
                f"Not enough train samples for bucket={bucket_label}: "
                f"need {train_size}, got {len(bucket_df)}."
            )
        train_random_state = int(rng.integers(0, np.iinfo(np.int32).max))
        train_sample = bucket_df.sample(
            n=int(train_size),
            replace=False,
            random_state=train_random_state,
        ).sort_values(["date", "_holdout_source_index"]).reset_index(drop=True)
        train_frames.append(train_sample)

    train_val = (
        pd.concat(train_frames, ignore_index=True)
        .sort_values(["date", "_holdout_source_index"])
        .reset_index(drop=True)
    )
    global_train_end_date = train_val["_date"].max()

    for bucket_label, train_sample in zip(bucket_labels, train_frames):
        bucket_df = data.loc[
            data["maturity_bucket"].astype(str).eq(str(bucket_label))
        ].copy()
        test_pool = bucket_df.loc[bucket_df["_date"] > global_train_end_date].copy()
        if len(test_pool) < test_size:
            raise ValueError(
                f"Not enough future test samples for bucket={bucket_label}: "
                f"need {test_size}, got {len(test_pool)} after "
                f"global_train_end_date={_date_str(global_train_end_date)}."
            )
        test_random_state = int(rng.integers(0, np.iinfo(np.int32).max))
        test_sample = test_pool.sample(
            n=int(test_size),
            replace=False,
            random_state=test_random_state,
        ).sort_values(["date", "_holdout_source_index"]).reset_index(drop=True)
        test_frames.append(test_sample)
        bucket_rows.append(
            {
                "holdout_seed": int(seed),
                "bucket_type": str(bucket_type).lower(),
                "train_scope": "universal",
                "maturity_bucket": str(bucket_label),
                "train_start_date": _date_str(train_sample["date"].min()),
                "train_end_date": _date_str(train_sample["date"].max()),
                "global_train_start_date": _date_str(train_val["date"].min()),
                "global_train_end_date": _date_str(global_train_end_date),
                "test_start_date": _date_str(test_sample["date"].min()),
                "test_end_date": _date_str(test_sample["date"].max()),
                "n_trainval_bucket": int(len(train_sample)),
                "n_test": int(len(test_sample)),
                "status": "ok",
                "error_message": "",
            }
        )

    test = (
        pd.concat(test_frames, ignore_index=True)
        .sort_values(["date", "_holdout_source_index"])
        .reset_index(drop=True)
    )
    if train_val["date"].max() >= test["date"].min():
        raise ValueError("Temporal leakage: universal train dates must be before test dates.")

    metadata = {
        "holdout_seed": int(seed),
        "bucket_type": str(bucket_type).lower(),
        "train_scope": "universal",
        "train_start_date": _date_str(train_val["date"].min()),
        "train_end_date": _date_str(train_val["date"].max()),
        "global_train_start_date": _date_str(train_val["date"].min()),
        "global_train_end_date": _date_str(global_train_end_date),
        "test_start_date": _date_str(test["date"].min()),
        "test_end_date": _date_str(test["date"].max()),
        "n_trainval": int(len(train_val)),
        "n_test": int(len(test)),
        "n_buckets": int(len(bucket_labels)),
    }
    return {
        "train_val": train_val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
        "metadata": metadata,
        "bucket_rows": bucket_rows,
    }


def _run_one_finetune_split(
    train_val: pd.DataFrame,
    test: pd.DataFrame,
    split_metadata: dict,
    theory: str,
    method: str,
    seed: int,
    output_dir: Path,
    checkpoint_path: Optional[Path],
    val_ratio: float,
    tl_lr: float,
    dl_lr: float,
    max_epochs: int,
    early_stop_consecutive: int,
    eps: float = 1e-4,
    batch_denominator: int = 600,
    device: Optional[str] = None,
) -> tuple[dict, list[dict], pd.DataFrame]:
    theory = str(theory).upper()
    method = str(method).upper()
    if theory not in THEORY_FEATURES:
        raise ValueError(f"Unknown theory={theory!r}. Expected one of {sorted(THEORY_FEATURES)}.")
    if method not in {"DL", "TL"}:
        raise ValueError("method must be 'DL' or 'TL'.")

    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    feature_cols = THEORY_FEATURES[theory]
    required = ["date", "mid_price", "delta", *feature_cols]
    train_val = train_val.dropna(subset=required).reset_index(drop=True)
    test = test.dropna(subset=required).reset_index(drop=True)
    if len(train_val) < 2 or len(test) == 0:
        raise ValueError(
            f"Empty split for {theory}-{method}: "
            f"bucket={split_metadata['maturity_bucket']} "
            f"n_trainval={len(train_val)}, n_test={len(test)}."
        )

    rng = np.random.default_rng(seed)
    idx = np.arange(len(train_val))
    rng.shuffle(idx)
    cut = int((1.0 - val_ratio) * len(idx))
    cut = min(max(cut, 1), len(idx) - 1)
    train_idx = idx[:cut]
    val_idx = idx[cut:]

    X_train = train_val.iloc[train_idx][feature_cols].to_numpy(np.float32)
    y_train = train_val.iloc[train_idx]["mid_price"].to_numpy(np.float32)
    d_train = train_val.iloc[train_idx]["delta"].to_numpy(np.float32)
    X_val = train_val.iloc[val_idx][feature_cols].to_numpy(np.float32)
    y_val = train_val.iloc[val_idx]["mid_price"].to_numpy(np.float32)
    d_val = train_val.iloc[val_idx]["delta"].to_numpy(np.float32)

    batch_size = max(math.ceil(len(train_idx) / batch_denominator), 1)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        TargetDataset(X_train, y_train, d_train, device=device),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=generator,
    )
    val_loader = DataLoader(
        TargetDataset(X_val, y_val, d_val, device=device),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    lr = tl_lr if method == "TL" else dl_lr
    model = _make_model(
        feature_cols=feature_cols,
        method=method,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    print(
        f"\n[holdout seed={seed} bucket={split_metadata['maturity_bucket']} "
        f"{theory}-{method}] n_train={len(train_idx)} n_val={len(val_idx)} "
        f"n_test={len(test)} batch_size={batch_size} lr={lr}"
    )
    model, best_val, history = finetune_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=lr,
        max_epochs=max_epochs,
        eps=eps,
        early_stop_consecutive=early_stop_consecutive,
    )

    X_test = test[feature_cols].to_numpy(np.float32)
    y_test = test["mid_price"].to_numpy(np.float32)
    d_test = test["delta"].to_numpy(np.float32)
    test_loader = DataLoader(
        TargetDataset(X_test, y_test, d_test, device=device),
        batch_size=8192,
        shuffle=False,
        drop_last=False,
    )
    pred_test = predict(model, X_test, device=device)
    test_weighted_mae = eval_loss(model, test_loader, eps=eps)
    test_mae = float(np.mean(np.abs(pred_test - y_test)))
    bsiv_metrics = bsiv_abs_error_median(test, pred_test)
    monthly_rows = monthly_test_metrics(
        test=test,
        pred_test=pred_test,
        theory=theory,
        method=method,
        seed=seed,
        maturity_bucket=split_metadata["maturity_bucket"],
        eps=eps,
    )

    history_dir = Path(output_dir) / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    run_id = (
        f"{theory.lower()}_{method.lower()}_holdout_seed{seed}_"
        f"{_path_safe_label(split_metadata['maturity_bucket'])}"
    )
    prediction_df = prediction_rows(
        test=test,
        pred_test=pred_test,
        run_id=run_id,
        theory=theory,
        method=method,
        seed=seed,
        maturity_bucket=split_metadata["maturity_bucket"],
        test_start_date=split_metadata["test_start_date"],
        eps=eps,
    )
    history.to_csv(history_dir / f"{run_id}.csv", index=False)
    pd.DataFrame(monthly_rows).to_csv(
        history_dir / f"{run_id}_monthly_test.csv",
        index=False,
    )

    row = {
        **split_metadata,
        "theory": theory,
        "method": method,
        "seed": seed,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "batch_size": batch_size,
        "lr": lr,
        "best_val": best_val,
        "test_weighted_mae": test_weighted_mae,
        "test_mae": test_mae,
        "test_mse": float(np.mean((pred_test - y_test) ** 2)),
        "prediction_path": str(Path(output_dir) / "predictions.parquet"),
        "finetune_checkpoint_path": "",
        "status": "ok",
        "error_message": "",
    }
    row.update(bsiv_metrics)
    print(
        f"[holdout seed={seed} bucket={split_metadata['maturity_bucket']} "
        f"{theory}-{method}] best_val={best_val:.6f} "
        f"test_weighted_mae={test_weighted_mae:.6f} test_mae={test_mae:.6f}"
    )
    return row, monthly_rows, prediction_df


def _metric_row_for_predictions(
    test: pd.DataFrame,
    pred_test: np.ndarray,
    split_metadata: dict,
    theory: str,
    method: str,
    seed: int,
    maturity_bucket: str,
    n_train: int,
    n_val: int,
    batch_size: int,
    lr: float,
    best_val: float,
    output_dir: Path,
    eps: float,
) -> dict:
    y_test = test["mid_price"].to_numpy(np.float32)
    d_test = test["delta"].to_numpy(np.float32)
    row = {
        **split_metadata,
        "maturity_bucket": str(maturity_bucket),
        "theory": theory,
        "method": method,
        "seed": int(seed),
        "n_train": int(n_train),
        "n_val": int(n_val),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "best_val": float(best_val),
        "test_weighted_mae": weighted_mae_numpy(pred_test, y_test, d_test, eps=eps),
        "test_mae": float(np.mean(np.abs(pred_test - y_test))),
        "test_mse": float(np.mean((pred_test - y_test) ** 2)),
        "prediction_path": str(Path(output_dir) / "predictions.parquet"),
        "finetune_checkpoint_path": "",
        "status": "ok",
        "error_message": "",
    }
    row.update(bsiv_metrics := bsiv_abs_error_median(test, pred_test))
    row["n_used_bsiv"] = bsiv_metrics["n_used_bsiv"]
    row["n_test"] = int(len(test))
    return row


def _run_one_universal_finetune_split(
    train_val: pd.DataFrame,
    test: pd.DataFrame,
    split_metadata: dict,
    bucket_labels: list[str],
    theory: str,
    method: str,
    seed: int,
    output_dir: Path,
    checkpoint_path: Optional[Path],
    val_ratio: float,
    tl_lr: float,
    dl_lr: float,
    max_epochs: int,
    early_stop_consecutive: int,
    eps: float = 1e-4,
    batch_denominator: int = 600,
    device: Optional[str] = None,
) -> tuple[list[dict], list[dict], pd.DataFrame]:
    theory = str(theory).upper()
    method = str(method).upper()
    if theory not in THEORY_FEATURES:
        raise ValueError(f"Unknown theory={theory!r}. Expected one of {sorted(THEORY_FEATURES)}.")
    if method not in {"DL", "TL"}:
        raise ValueError("method must be 'DL' or 'TL'.")

    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    feature_cols = THEORY_FEATURES[theory]
    required = ["date", "mid_price", "delta", "maturity_bucket", *feature_cols]
    train_val = train_val.dropna(subset=required).reset_index(drop=True)
    test = test.dropna(subset=required).reset_index(drop=True)
    if len(train_val) < 2 or len(test) == 0:
        raise ValueError(
            f"Empty universal split for {theory}-{method}: "
            f"n_trainval={len(train_val)}, n_test={len(test)}."
        )

    rng = np.random.default_rng(seed)
    idx = np.arange(len(train_val))
    rng.shuffle(idx)
    cut = int((1.0 - val_ratio) * len(idx))
    cut = min(max(cut, 1), len(idx) - 1)
    train_idx = idx[:cut]
    val_idx = idx[cut:]

    X_train = train_val.iloc[train_idx][feature_cols].to_numpy(np.float32)
    y_train = train_val.iloc[train_idx]["mid_price"].to_numpy(np.float32)
    d_train = train_val.iloc[train_idx]["delta"].to_numpy(np.float32)
    X_val = train_val.iloc[val_idx][feature_cols].to_numpy(np.float32)
    y_val = train_val.iloc[val_idx]["mid_price"].to_numpy(np.float32)
    d_val = train_val.iloc[val_idx]["delta"].to_numpy(np.float32)

    batch_size = max(math.ceil(len(train_idx) / batch_denominator), 1)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        TargetDataset(X_train, y_train, d_train, device=device),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=generator,
    )
    val_loader = DataLoader(
        TargetDataset(X_val, y_val, d_val, device=device),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    lr = tl_lr if method == "TL" else dl_lr
    model = _make_model(
        feature_cols=feature_cols,
        method=method,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    print(
        f"\n[universal holdout seed={seed} {theory}-{method}] "
        f"n_train={len(train_idx)} n_val={len(val_idx)} n_test={len(test)} "
        f"batch_size={batch_size} lr={lr}"
    )
    model, best_val, history = finetune_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=lr,
        max_epochs=max_epochs,
        eps=eps,
        early_stop_consecutive=early_stop_consecutive,
    )

    X_test = test[feature_cols].to_numpy(np.float32)
    pred_test = predict(model, X_test, device=device)

    history_dir = Path(output_dir) / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{theory.lower()}_{method.lower()}_universal_holdout_seed{seed}"
    history.to_csv(history_dir / f"{run_id}.csv", index=False)

    rows = []
    monthly_rows = []
    prediction_frames = []
    row_metadata = {
        **split_metadata,
        "n_trainval_total": int(len(train_val)),
        "n_train_total": int(len(train_idx)),
        "n_val_total": int(len(val_idx)),
        "n_test_total": int(len(test)),
    }

    overall_row = _metric_row_for_predictions(
        test=test,
        pred_test=pred_test,
        split_metadata=row_metadata,
        theory=theory,
        method=method,
        seed=seed,
        maturity_bucket="all",
        n_train=len(train_idx),
        n_val=len(val_idx),
        batch_size=batch_size,
        lr=lr,
        best_val=best_val,
        output_dir=output_dir,
        eps=eps,
    )
    rows.append(overall_row)
    monthly_rows.extend(
        monthly_test_metrics(
            test=test,
            pred_test=pred_test,
            theory=theory,
            method=method,
            seed=seed,
            maturity_bucket="all",
            eps=eps,
        )
    )

    test_with_pred = test.reset_index(drop=True).copy()
    test_with_pred["_pred"] = pred_test
    for bucket_label in ["all", *bucket_labels]:
        if bucket_label == "all":
            continue
        group = test_with_pred.loc[
            test_with_pred["maturity_bucket"].astype(str).eq(str(bucket_label))
        ].copy()
        if group.empty:
            continue
        group_pred = group["_pred"].to_numpy(dtype=float)
        group_test = group.drop(columns=["_pred"]).reset_index(drop=True)
        bucket_metadata = {
            **row_metadata,
            "test_start_date": _date_str(group_test["date"].min()),
            "test_end_date": _date_str(group_test["date"].max()),
        }
        rows.append(
            _metric_row_for_predictions(
                test=group_test,
                pred_test=group_pred,
                split_metadata=bucket_metadata,
                theory=theory,
                method=method,
                seed=seed,
                maturity_bucket=str(bucket_label),
                n_train=len(train_idx),
                n_val=len(val_idx),
                batch_size=batch_size,
                lr=lr,
                best_val=best_val,
                output_dir=output_dir,
                eps=eps,
            )
        )
        monthly_rows.extend(
            monthly_test_metrics(
                test=group_test,
                pred_test=group_pred,
                theory=theory,
                method=method,
                seed=seed,
                maturity_bucket=str(bucket_label),
                eps=eps,
            )
        )
        prediction_frames.append(
            prediction_rows(
                test=group_test,
                pred_test=group_pred,
                run_id=f"{run_id}_{_path_safe_label(bucket_label)}",
                theory=theory,
                method=method,
                seed=seed,
                maturity_bucket=str(bucket_label),
                test_start_date=bucket_metadata["test_start_date"],
                eps=eps,
            )
        )

    pd.DataFrame(monthly_rows).to_csv(
        history_dir / f"{run_id}_monthly_test.csv",
        index=False,
    )
    print(
        f"[universal holdout seed={seed} {theory}-{method}] "
        f"best_val={best_val:.6f} overall_test_mae={overall_row['test_mae']:.6f}"
    )
    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    return rows, monthly_rows, predictions


def _run_seed_finetune(
    splits: list[dict],
    output_dir: Path,
    bs_checkpoint_path: Path,
    heston_checkpoint_path: Path,
    seed: int,
    theories: Iterable[str],
    methods: Iterable[str],
    val_ratio: float,
    tl_lr: float,
    dl_lr: float,
    max_epochs: int,
    early_stop_consecutive: int,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = {
        "BS": Path(bs_checkpoint_path),
        "HESTON": Path(heston_checkpoint_path),
    }

    rows = []
    monthly_rows = []
    prediction_frames = []
    for split in splits:
        for theory in theories:
            theory = str(theory).upper()
            for method in methods:
                method = str(method).upper()
                try:
                    row, month_rows, pred_rows = _run_one_finetune_split(
                        train_val=split["train_val"],
                        test=split["test"],
                        split_metadata=split["metadata"],
                        theory=theory,
                        method=method,
                        seed=int(seed),
                        output_dir=output_dir,
                        checkpoint_path=checkpoint_paths[theory] if method == "TL" else None,
                        val_ratio=val_ratio,
                        tl_lr=tl_lr,
                        dl_lr=dl_lr,
                        max_epochs=max_epochs,
                        early_stop_consecutive=early_stop_consecutive,
                    )
                    rows.append(row)
                    monthly_rows.extend(month_rows)
                    prediction_frames.append(pred_rows)
                except Exception as exc:
                    rows.append(
                        {
                            **split["metadata"],
                            "theory": theory,
                            "method": method,
                            "seed": int(seed),
                            "status": "failed",
                            "error_message": str(exc),
                        }
                    )
                pd.DataFrame(rows).to_csv(output_dir / "results.csv", index=False)
                pd.DataFrame(monthly_rows).to_csv(output_dir / "monthly_results.csv", index=False)
                if prediction_frames:
                    pd.concat(prediction_frames, ignore_index=True).to_parquet(
                        output_dir / "predictions.parquet",
                        index=False,
                    )

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "results.csv", index=False)
    pd.DataFrame(monthly_rows).to_csv(output_dir / "monthly_results.csv", index=False)
    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_parquet(
            output_dir / "predictions.parquet",
            index=False,
        )
    return results


def _run_seed_universal_finetune(
    split: dict,
    bucket_labels: list[str],
    output_dir: Path,
    bs_checkpoint_path: Path,
    heston_checkpoint_path: Path,
    seed: int,
    theories: Iterable[str],
    methods: Iterable[str],
    val_ratio: float,
    tl_lr: float,
    dl_lr: float,
    max_epochs: int,
    early_stop_consecutive: int,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = {
        "BS": Path(bs_checkpoint_path),
        "HESTON": Path(heston_checkpoint_path),
    }

    rows = []
    monthly_rows = []
    prediction_frames = []
    for theory in theories:
        theory = str(theory).upper()
        for method in methods:
            method = str(method).upper()
            try:
                result_rows, month_rows, pred_rows = _run_one_universal_finetune_split(
                    train_val=split["train_val"],
                    test=split["test"],
                    split_metadata=split["metadata"],
                    bucket_labels=bucket_labels,
                    theory=theory,
                    method=method,
                    seed=int(seed),
                    output_dir=output_dir,
                    checkpoint_path=checkpoint_paths[theory] if method == "TL" else None,
                    val_ratio=val_ratio,
                    tl_lr=tl_lr,
                    dl_lr=dl_lr,
                    max_epochs=max_epochs,
                    early_stop_consecutive=early_stop_consecutive,
                )
                rows.extend(result_rows)
                monthly_rows.extend(month_rows)
                if not pred_rows.empty:
                    prediction_frames.append(pred_rows)
            except Exception as exc:
                rows.append(
                    {
                        **split["metadata"],
                        "maturity_bucket": "all",
                        "theory": theory,
                        "method": method,
                        "seed": int(seed),
                        "status": "failed",
                        "error_message": str(exc),
                    }
                )
            pd.DataFrame(rows).to_csv(output_dir / "results.csv", index=False)
            pd.DataFrame(monthly_rows).to_csv(output_dir / "monthly_results.csv", index=False)
            if prediction_frames:
                pd.concat(prediction_frames, ignore_index=True).to_parquet(
                    output_dir / "predictions.parquet",
                    index=False,
                )

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "results.csv", index=False)
    pd.DataFrame(monthly_rows).to_csv(output_dir / "monthly_results.csv", index=False)
    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_parquet(
            output_dir / "predictions.parquet",
            index=False,
        )
    return results


def _run_seed_pactran(
    splits: list[dict],
    checkpoint_specs: Iterable[dict],
    output_csv: Path,
    target_col: str,
    sigma2: float,
    sigma_pi2: float,
    prior_center: str,
    posterior_dir: Optional[Path],
    subsample_size: Optional[int],
    subsample_frac: Optional[float],
    n_subsamples: int,
    subsample_seed: int,
    device: Optional[str] = None,
) -> pd.DataFrame:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if posterior_dir is not None:
        posterior_dir = Path(posterior_dir)
        posterior_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for split in splits:
        metadata = split["metadata"]
        train_val = split["train_val"]
        for spec in checkpoint_specs:
            try:
                row, posterior = score_checkpoint_blr_subsamples(
                    spec=spec,
                    target_dataset=train_val,
                    target_col=target_col,
                    date_before=None,
                    only_put=False,
                    sigma2=sigma2,
                    sigma_pi2=sigma_pi2,
                    prior_center=prior_center,
                    device=device,
                    subsample_size=subsample_size,
                    subsample_frac=subsample_frac,
                    n_subsamples=n_subsamples,
                    subsample_seed=subsample_seed,
                )
                theory = row.pop("name")
                row.update(metadata)
                row["theory"] = theory
                row["n_pac_bucket"] = row.pop("n_month_samples")
                if posterior_dir is not None:
                    posterior_path = (
                        posterior_dir
                        / f"{theory}_{_path_safe_label(metadata['maturity_bucket'])}_posterior.pt"
                    )
                    torch.save(posterior, posterior_path)
                    row["posterior_path"] = str(posterior_path)
                else:
                    row["posterior_path"] = ""
            except Exception as exc:
                row = {
                    **metadata,
                    "theory": spec.get("name", ""),
                    "checkpoint_path": str(spec.get("checkpoint_path", "")),
                    "expected_train_nll_subsample_mean": np.nan,
                    "kl_subsample_mean": np.nan,
                    "pac_score_1_subsample_mean": np.nan,
                    "pac_score_1_subsample_std": np.nan,
                    "bound_objective_subsample_mean": np.nan,
                    "bound_objective_subsample_std": np.nan,
                    "n_subsamples": np.nan,
                    "subsample_frac": np.nan if subsample_frac is None else float(subsample_frac),
                    "subsample_size": np.nan,
                    "n_pac_bucket": int(len(train_val)),
                    "prior_center": prior_center,
                    "status": "failed",
                    "error_message": str(exc),
                    "posterior_path": "",
                }
            rows.append(row)

    results = pd.DataFrame(rows)
    ok = results["status"].eq("ok")
    results.loc[ok, "pac_rank"] = results.loc[ok].groupby(
        ["maturity_bucket"]
    )[
        "pac_score_1_subsample_mean"
    ].rank(method="first", ascending=True)
    results.loc[~ok, "pac_rank"] = np.nan
    results = results.sort_values(
        ["maturity_bucket", "pac_rank", "theory"],
        na_position="last",
    ).reset_index(drop=True)
    results.to_csv(output_csv, index=False)
    return results


def _run_seed_universal_pactran(
    split: dict,
    bucket_labels: list[str],
    checkpoint_specs: Iterable[dict],
    output_csv: Path,
    target_col: str,
    sigma2: float,
    sigma_pi2: float,
    prior_center: str,
    posterior_dir: Optional[Path],
    subsample_size: Optional[int],
    subsample_frac: Optional[float],
    n_subsamples: int,
    subsample_seed: int,
    device: Optional[str] = None,
) -> pd.DataFrame:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if posterior_dir is not None:
        posterior_dir = Path(posterior_dir)
        posterior_dir.mkdir(parents=True, exist_ok=True)

    metadata = split["metadata"]
    train_val = split["train_val"]
    rows = []
    for spec in checkpoint_specs:
        try:
            row, posterior = score_checkpoint_blr_subsamples(
                spec=spec,
                target_dataset=train_val,
                target_col=target_col,
                date_before=None,
                only_put=False,
                sigma2=sigma2,
                sigma_pi2=sigma_pi2,
                prior_center=prior_center,
                device=device,
                subsample_size=subsample_size,
                subsample_frac=subsample_frac,
                n_subsamples=n_subsamples,
                subsample_seed=subsample_seed,
            )
            theory = row.pop("name")
            row.update(metadata)
            row["theory"] = theory
            row["pactran_scope"] = "universal"
            row["n_pac_universal_train"] = row.pop("n_month_samples")
            if posterior_dir is not None:
                posterior_path = posterior_dir / f"{theory}_universal_posterior.pt"
                torch.save(posterior, posterior_path)
                row["posterior_path"] = str(posterior_path)
            else:
                row["posterior_path"] = ""
        except Exception as exc:
            row = {
                **metadata,
                "theory": spec.get("name", ""),
                "checkpoint_path": str(spec.get("checkpoint_path", "")),
                "expected_train_nll_subsample_mean": np.nan,
                "kl_subsample_mean": np.nan,
                "pac_score_1_subsample_mean": np.nan,
                "pac_score_1_subsample_std": np.nan,
                "bound_objective_subsample_mean": np.nan,
                "bound_objective_subsample_std": np.nan,
                "n_subsamples": np.nan,
                "subsample_frac": np.nan if subsample_frac is None else float(subsample_frac),
                "subsample_size": np.nan,
                "n_pac_universal_train": int(len(train_val)),
                "prior_center": prior_center,
                "pactran_scope": "universal",
                "status": "failed",
                "error_message": str(exc),
                "posterior_path": "",
            }

        for bucket in ["all", *bucket_labels]:
            bucket_row = dict(row)
            bucket_row["maturity_bucket"] = str(bucket)
            rows.append(bucket_row)

    results = pd.DataFrame(rows)
    ok = results["status"].eq("ok")
    results.loc[ok, "pac_rank"] = results.loc[ok].groupby(
        ["maturity_bucket"]
    )[
        "pac_score_1_subsample_mean"
    ].rank(method="first", ascending=True)
    results.loc[~ok, "pac_rank"] = np.nan
    results = results.sort_values(
        ["maturity_bucket", "pac_rank", "theory"],
        na_position="last",
    ).reset_index(drop=True)
    results.to_csv(output_csv, index=False)
    return results


def run_repeated_random_holdout_experiments(
    heston_dataset: pd.DataFrame,
    checkpoint_specs: Iterable[dict],
    output_root: Path,
    bs_checkpoint_path: Path,
    heston_checkpoint_path: Path,
    seeds: Iterable[int],
    train_size: int,
    test_size: int,
    bucket_type: str,
    theories: Iterable[str] = ("BS", "HESTON"),
    methods: Iterable[str] = ("TL",),
    val_ratio: float = 0.2,
    tl_lr: float = 1e-4,
    dl_lr: float = 1e-3,
    max_epochs: int = 200,
    early_stop_consecutive: int = 2,
    run_finetune: bool = True,
    run_pactran_score: bool = True,
    pactran_target_col: str = "mid_price",
    pactran_sigma2: float = 0.5,
    pactran_sigma_pi2: float = 0.01,
    pactran_prior_center: str = "pretrained",
    pactran_subsample_size: Optional[int] = 10_000,
    pactran_subsample_frac: Optional[float] = None,
    pactran_n_subsamples: int = 5,
    pactran_subsample_seed: int = 123,
    maturity_bins: Optional[list[float]] = None,
    maturity_labels: Optional[list[str]] = None,
) -> pd.DataFrame:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    data, bucket_labels = _prepare_holdout_data(
        heston_dataset=heston_dataset,
        bucket_type=bucket_type,
        maturity_bins=maturity_bins,
        maturity_labels=maturity_labels,
    )

    aligned_frames = []
    split_rows = []
    for seed in seeds:
        seed = int(seed)
        seed_dir = output_root / f"seed_{seed}"
        finetune_dir = seed_dir / "finetune"
        pactran_csv = seed_dir / "pactran" / "scores.csv"
        posterior_dir = seed_dir / "pactran" / "posteriors"
        aligned_csv = seed_dir / "aligned" / "pactran_finetune_aligned.csv"

        splits = []
        for bucket_label in bucket_labels:
            bucket_df = data.loc[
                data["maturity_bucket"].astype(str).eq(str(bucket_label))
            ].copy()
            try:
                train_val, test, metadata = _sample_temporal_holdout(
                    bucket_df=bucket_df,
                    bucket_label=str(bucket_label),
                    seed=seed,
                    train_size=train_size,
                    test_size=test_size,
                )
                metadata["bucket_type"] = str(bucket_type).lower()
                splits.append(
                    {
                        "train_val": train_val,
                        "test": test,
                        "metadata": metadata,
                    }
                )
                split_rows.append({**metadata, "status": "ok", "error_message": ""})
            except Exception as exc:
                split_rows.append(
                    {
                        "holdout_seed": seed,
                        "bucket_type": str(bucket_type).lower(),
                        "maturity_bucket": str(bucket_label),
                        "status": "failed",
                        "error_message": str(exc),
                    }
                )

        pd.DataFrame(split_rows).to_csv(output_root / "splits.csv", index=False)
        if not splits:
            continue

        if run_finetune:
            finetune_results = _run_seed_finetune(
                splits=splits,
                output_dir=finetune_dir,
                bs_checkpoint_path=bs_checkpoint_path,
                heston_checkpoint_path=heston_checkpoint_path,
                seed=seed,
                theories=theories,
                methods=methods,
                val_ratio=val_ratio,
                tl_lr=tl_lr,
                dl_lr=dl_lr,
                max_epochs=max_epochs,
                early_stop_consecutive=early_stop_consecutive,
            )
        else:
            finetune_results = pd.read_csv(finetune_dir / "results.csv")

        if run_pactran_score:
            pactran_results = _run_seed_pactran(
                splits=splits,
                checkpoint_specs=checkpoint_specs,
                output_csv=pactran_csv,
                target_col=pactran_target_col,
                sigma2=pactran_sigma2,
                sigma_pi2=pactran_sigma_pi2,
                prior_center=pactran_prior_center,
                posterior_dir=posterior_dir,
                subsample_size=pactran_subsample_size,
                subsample_frac=pactran_subsample_frac,
                n_subsamples=pactran_n_subsamples,
                subsample_seed=pactran_subsample_seed,
            )
        else:
            pactran_results = pd.read_csv(pactran_csv)

        aligned = align_pactran_finetune(
            finetune_results=finetune_results,
            pactran_results=pactran_results,
            output_csv=aligned_csv,
        )
        aligned_frames.append(aligned)

    if aligned_frames:
        summary = pd.concat(aligned_frames, ignore_index=True)
    else:
        summary = pd.DataFrame()
    summary.to_csv(output_root / "aligned_summary.csv", index=False)
    return summary


def run_repeated_random_holdout_universal_experiments(
    heston_dataset: pd.DataFrame,
    checkpoint_specs: Iterable[dict],
    output_root: Path,
    bs_checkpoint_path: Path,
    heston_checkpoint_path: Path,
    seeds: Iterable[int],
    train_size: int,
    test_size: int,
    bucket_type: str,
    theories: Iterable[str] = ("BS", "HESTON"),
    methods: Iterable[str] = ("TL",),
    val_ratio: float = 0.2,
    tl_lr: float = 1e-4,
    dl_lr: float = 1e-3,
    max_epochs: int = 200,
    early_stop_consecutive: int = 2,
    run_finetune: bool = True,
    run_pactran_score: bool = True,
    pactran_target_col: str = "mid_price",
    pactran_sigma2: float = 0.5,
    pactran_sigma_pi2: float = 0.01,
    pactran_prior_center: str = "pretrained",
    pactran_subsample_size: Optional[int] = 10_000,
    pactran_subsample_frac: Optional[float] = None,
    pactran_n_subsamples: int = 5,
    pactran_subsample_seed: int = 123,
    maturity_bins: Optional[list[float]] = None,
    maturity_labels: Optional[list[str]] = None,
) -> pd.DataFrame:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    data, bucket_labels = _prepare_holdout_data(
        heston_dataset=heston_dataset,
        bucket_type=bucket_type,
        maturity_bins=maturity_bins,
        maturity_labels=maturity_labels,
    )

    aligned_frames = []
    split_rows = []
    for seed in seeds:
        seed = int(seed)
        seed_dir = output_root / f"seed_{seed}"
        finetune_dir = seed_dir / "finetune"
        pactran_csv = seed_dir / "pactran" / "scores.csv"
        posterior_dir = seed_dir / "pactran" / "posteriors"
        aligned_csv = seed_dir / "aligned" / "pactran_finetune_aligned.csv"

        try:
            split = _sample_universal_temporal_holdout(
                data=data,
                bucket_labels=bucket_labels,
                seed=seed,
                train_size=train_size,
                test_size=test_size,
                bucket_type=bucket_type,
            )
            split_rows.extend(split["bucket_rows"])
        except Exception as exc:
            message = f"[universal holdout seed={seed}] split failed: {exc}"
            print(message)
            split_rows.append(
                {
                    "holdout_seed": seed,
                    "bucket_type": str(bucket_type).lower(),
                    "train_scope": "universal",
                    "maturity_bucket": "all",
                    "status": "failed",
                    "error_message": str(exc),
                }
            )
            pd.DataFrame(split_rows).to_csv(output_root / "splits.csv", index=False)
            continue

        pd.DataFrame(split_rows).to_csv(output_root / "splits.csv", index=False)

        if run_finetune:
            finetune_results = _run_seed_universal_finetune(
                split=split,
                bucket_labels=bucket_labels,
                output_dir=finetune_dir,
                bs_checkpoint_path=bs_checkpoint_path,
                heston_checkpoint_path=heston_checkpoint_path,
                seed=seed,
                theories=theories,
                methods=methods,
                val_ratio=val_ratio,
                tl_lr=tl_lr,
                dl_lr=dl_lr,
                max_epochs=max_epochs,
                early_stop_consecutive=early_stop_consecutive,
            )
        else:
            finetune_results = pd.read_csv(finetune_dir / "results.csv")

        if run_pactran_score:
            pactran_results = _run_seed_universal_pactran(
                split=split,
                bucket_labels=bucket_labels,
                checkpoint_specs=checkpoint_specs,
                output_csv=pactran_csv,
                target_col=pactran_target_col,
                sigma2=pactran_sigma2,
                sigma_pi2=pactran_sigma_pi2,
                prior_center=pactran_prior_center,
                posterior_dir=posterior_dir,
                subsample_size=pactran_subsample_size,
                subsample_frac=pactran_subsample_frac,
                n_subsamples=pactran_n_subsamples,
                subsample_seed=pactran_subsample_seed,
            )
        else:
            pactran_results = pd.read_csv(pactran_csv)

        aligned = align_pactran_finetune(
            finetune_results=finetune_results,
            pactran_results=pactran_results,
            output_csv=aligned_csv,
        )
        aligned_frames.append(aligned)

    if aligned_frames:
        summary = pd.concat(aligned_frames, ignore_index=True)
    else:
        summary = pd.DataFrame()
    summary.to_csv(output_root / "aligned_summary.csv", index=False)
    return summary
