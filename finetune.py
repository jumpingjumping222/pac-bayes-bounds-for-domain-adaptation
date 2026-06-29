import copy
import math
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from model import ResMLP
from pretrain import FEATURE_COLUMNS_BSM, FEATURE_COLUMNS_HESTON


THEORY_FEATURES = {
    "BS": FEATURE_COLUMNS_BSM,
    "HESTON": FEATURE_COLUMNS_HESTON,
}
MATURITY_BINS = [0.0, 1.0 / 12.0, 3.0 / 12.0, 6.0 / 12.0, 1.0, 2.0, 3.0]
MATURITY_LABELS = ["0-1m", "1-3m", "3-6m", "6m-1y", "1y-2y", "2y-3y"]


class TargetDataset(Dataset):
    def __init__(self, X, y, delta, device):
        self.X = torch.tensor(X, dtype=torch.float32, device=device)
        self.y = torch.tensor(y.reshape(-1, 1), dtype=torch.float32, device=device)
        self.delta = torch.tensor(delta.reshape(-1, 1), dtype=torch.float32, device=device)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i], self.delta[i]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def weighted_mae(model, X, y, delta, eps: float = 1e-4):
    pred = model(X)
    w = 1.0 / (delta.abs() + eps)
    return torch.mean(w * (pred - y).abs())


@torch.no_grad()
def eval_loss(model, loader, eps: float = 1e-4) -> float:
    model.eval()
    total = 0.0
    n = 0
    for X, y, delta in loader:
        loss = weighted_mae(model, X, y, delta, eps=eps)
        batch_size = X.shape[0]
        total += float(loss.detach().cpu()) * batch_size
        n += batch_size
    return total / max(n, 1)


def finetune_model(
    model,
    train_loader,
    val_loader,
    lr: float,
    max_epochs: int = 200,
    eps: float = 1e-4,
    early_stop_consecutive: int = 2,
):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = float("inf")
    best_state = None
    worse_streak = 0
    prev_val = None
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_total = 0.0
        train_n = 0

        for X, y, delta in train_loader:
            opt.zero_grad(set_to_none=True)
            loss = weighted_mae(model, X, y, delta, eps=eps)
            loss.backward()
            opt.step()

            batch_size = X.shape[0]
            train_total += float(loss.detach().cpu()) * batch_size
            train_n += batch_size

        train_loss = train_total / max(train_n, 1)
        val_loss = eval_loss(model, val_loader, eps=eps)

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())

        if prev_val is not None and val_loss > prev_val:
            worse_streak += 1
        else:
            worse_streak = 0
        prev_val = val_loss

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val": best_val,
                "worse_streak": worse_streak,
            }
        )
        print(
            f"  epoch={epoch:03d} train={train_loss:.6f} "
            f"val={val_loss:.6f} best={best_val:.6f} "
            f"worse_streak={worse_streak}"
        )

        if worse_streak >= early_stop_consecutive:
            print(
                f"Early stop at epoch={epoch}: validation loss increased "
                f"{early_stop_consecutive} epochs in a row."
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val, pd.DataFrame(history)


@torch.no_grad()
def predict(model, X, device):
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    return model(X_t).detach().cpu().numpy().reshape(-1)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_price_scalar(S, K, T, r, q, sigma, cp_flag) -> float:
    if T <= 0:
        return max(S - K, 0.0) if cp_flag == "C" else max(K - S, 0.0)

    sigma = max(float(sigma), 1e-12)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (
        sigma * math.sqrt(T)
    )
    d2 = d1 - sigma * math.sqrt(T)

    if cp_flag == "C":
        return S * math.exp(-q * T) * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * math.exp(-q * T) * _norm_cdf(-d1)


def _implied_vol_scalar(price, S, K, T, r, q, cp_flag) -> float:
    if not all(np.isfinite([price, S, K, T, r, q])) or S <= 0 or K <= 0 or T <= 0:
        return np.nan

    cp_flag = str(cp_flag).upper()
    if cp_flag not in {"C", "P"}:
        cp_flag = "P"

    if cp_flag == "C":
        lower = max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
        upper = S * math.exp(-q * T)
    else:
        lower = max(K * math.exp(-r * T) - S * math.exp(-q * T), 0.0)
        upper = K * math.exp(-r * T)

    if price < lower - 1e-8 or price > upper + 1e-8:
        return np.nan

    lo = 1e-6
    hi = 5.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        mid_price = _bs_price_scalar(S, K, T, r, q, mid, cp_flag)
        if mid_price < price:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def bsiv_abs_error_median(df: pd.DataFrame, pred_price: np.ndarray) -> dict:
    tau_col = "tau" if "tau" in df.columns else "T"
    cp_flags = df["cp_flag"].to_numpy() if "cp_flag" in df.columns else np.full(len(df), "P")
    market = df["mid_price"].to_numpy(dtype=float)

    iv_mkt = np.array(
        [
            _implied_vol_scalar(market[i], df["S"].iloc[i], df["K"].iloc[i], df[tau_col].iloc[i],
                                df["r"].iloc[i], df["d"].iloc[i], cp_flags[i])
            for i in range(len(df))
        ],
        dtype=float,
    )
    iv_pred = np.array(
        [
            _implied_vol_scalar(pred_price[i], df["S"].iloc[i], df["K"].iloc[i], df[tau_col].iloc[i],
                                df["r"].iloc[i], df["d"].iloc[i], cp_flags[i])
            for i in range(len(df))
        ],
        dtype=float,
    )

    valid = np.isfinite(iv_mkt) & np.isfinite(iv_pred)
    if valid.any():
        median_abs_error = float(np.median(np.abs(iv_pred[valid] - iv_mkt[valid])))
    else:
        median_abs_error = np.nan

    return {
        "test_bsiv_median": median_abs_error,
        "frac_nan_pred_iv": float(np.mean(~np.isfinite(iv_pred))),
        "frac_nan_mkt_iv": float(np.mean(~np.isfinite(iv_mkt))),
        "n_used_bsiv": int(valid.sum()),
    }


def weighted_mae_numpy(pred, y, delta, eps: float = 1e-4) -> float:
    w = 1.0 / (np.abs(delta) + eps)
    return float(np.mean(w * np.abs(pred - y)))


PREDICTION_COLUMNS = [
    "run_id",
    "theory",
    "method",
    "seed",
    "maturity_bucket",
    "test_start_date",
    "source_row_id",
    "date",
    "exdate",
    "cp_flag",
    "test_month",
    "S",
    "K",
    "tau",
    "delta",
    "mid_price",
    "pred_price",
    "pricing_error",
    "abs_error",
    "squared_error",
    "weighted_abs_error",
    "r",
    "d",
    "IV1",
    "impl_volatility",
    "v0",
    "theta",
    "kappa",
    "xi",
    "rho",
]


def prediction_rows(
    test: pd.DataFrame,
    pred_test: np.ndarray,
    run_id: str,
    theory: str,
    method: str,
    seed: int,
    maturity_bucket: str,
    test_start_date: str,
    eps: float = 1e-4,
) -> pd.DataFrame:
    out = pd.DataFrame(index=test.index)
    out["run_id"] = run_id
    out["theory"] = theory
    out["method"] = method
    out["seed"] = seed
    out["maturity_bucket"] = str(maturity_bucket)
    out["test_start_date"] = test_start_date
    out["source_row_id"] = test.index.to_numpy()

    for col in [
        "date",
        "exdate",
        "cp_flag",
        "S",
        "K",
        "tau",
        "delta",
        "mid_price",
        "r",
        "d",
        "IV1",
        "impl_volatility",
        "v0",
        "theta",
        "kappa",
        "xi",
        "rho",
    ]:
        out[col] = test[col].to_numpy() if col in test.columns else np.nan

    out["test_month"] = pd.to_datetime(out["date"]).dt.to_period("M").astype(str)
    pred = np.asarray(pred_test, dtype=float)
    y = out["mid_price"].to_numpy(dtype=float)
    delta = out["delta"].to_numpy(dtype=float)
    error = pred - y
    out["pred_price"] = pred
    out["pricing_error"] = error
    out["abs_error"] = np.abs(error)
    out["squared_error"] = error**2
    out["weighted_abs_error"] = np.abs(error) / (np.abs(delta) + eps)
    return out[PREDICTION_COLUMNS].reset_index(drop=True)


def add_maturity_bucket(
    df: pd.DataFrame,
    maturity_bins: Optional[list[float]] = None,
    maturity_labels: Optional[list[str]] = None,
) -> pd.DataFrame:
    maturity_bins = MATURITY_BINS if maturity_bins is None else maturity_bins
    maturity_labels = MATURITY_LABELS if maturity_labels is None else maturity_labels
    out = df.copy()
    out["maturity_bucket"] = pd.cut(
        out["tau"],
        bins=maturity_bins,
        labels=maturity_labels,
        include_lowest=True,
    )
    return out.dropna(subset=["maturity_bucket"])


def _path_safe_label(value: str) -> str:
    return str(value).replace("/", "-").replace(" ", "_")


def _month_str(value) -> str:
    return str(pd.Period(value, freq="M"))


def _next_month(month: str) -> str:
    return str(pd.Period(month, freq="M") + 1)


def _date_str(value) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _make_rolling_date_windows(
    heston_dataset: pd.DataFrame,
    train_days: int,
    test_days: int,
    step_days: int,
) -> list[dict]:
    if train_days <= 0 or test_days <= 0 or step_days <= 0:
        raise ValueError("train_days, test_days, and step_days must be positive.")

    dates = (
        pd.to_datetime(heston_dataset["date"])
        .dt.normalize()
        .drop_duplicates()
        .sort_values()
        .to_list()
    )
    windows = []
    window_idx = 0
    last_start = len(dates) - train_days - test_days
    for start in range(0, last_start + 1, step_days):
        train_dates = dates[start:start + train_days]
        test_dates = dates[start + train_days:start + train_days + test_days]
        windows.append(
            {
                "window_id": f"w{window_idx:04d}",
                "train_dates": train_dates,
                "test_dates": test_dates,
                "train_start_date": _date_str(train_dates[0]),
                "train_end_date": _date_str(train_dates[-1]),
                "test_start_date": _date_str(test_dates[0]),
                "test_end_date": _date_str(test_dates[-1]),
                "train_days": int(train_days),
                "test_days": int(test_days),
                "step_days": int(step_days),
            }
        )
        window_idx += 1
    return windows


def monthly_test_metrics(
    test: pd.DataFrame,
    pred_test: np.ndarray,
    theory: str,
    method: str,
    seed: int,
    maturity_bucket: str = "all",
    eps: float = 1e-4,
) -> list[dict]:
    out = test.copy()
    out["_pred"] = pred_test
    out["_month"] = pd.to_datetime(out["date"]).dt.to_period("M").astype(str)

    rows = []
    for month, group in out.groupby("_month", sort=True):
        pred = group["_pred"].to_numpy(dtype=float)
        y = group["mid_price"].to_numpy(dtype=float)
        delta = group["delta"].to_numpy(dtype=float)
        bsiv_metrics = bsiv_abs_error_median(group, pred)

        row = {
            "theory": theory,
            "method": method,
            "seed": seed,
            "maturity_bucket": str(maturity_bucket),
            "test_month": month,
            "n_test": int(len(group)),
            "test_weighted_mae": weighted_mae_numpy(pred, y, delta, eps=eps),
            "test_mae": float(np.mean(np.abs(pred - y))),
        }
        row.update(bsiv_metrics)
        rows.append(row)

    return rows


def _prepare_empirical_data(
    heston_dataset: pd.DataFrame,
    feature_cols: list[str],
    test_start_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = ["date", "mid_price", "delta", *feature_cols]
    if "cp_flag" in heston_dataset.columns:
        required.append("cp_flag")

    df = heston_dataset.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=required).sort_values("date").reset_index(drop=True)
    if "cp_flag" in df.columns:
        df = df.loc[df["cp_flag"].astype(str).str.upper() == "P"].copy()

    split_date = pd.Timestamp(test_start_date)
    train_val = df.loc[df["date"] < split_date].copy()
    test = df.loc[df["date"] >= split_date].copy()
    return train_val, test


def _prepare_empirical_monthly_data(
    heston_dataset: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    required = ["date", "mid_price", "delta", *feature_cols]
    if "cp_flag" in heston_dataset.columns:
        required.append("cp_flag")

    df = heston_dataset.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=required).sort_values("date").reset_index(drop=True)
    if "cp_flag" in df.columns:
        df = df.loc[df["cp_flag"].astype(str).str.upper() == "P"].copy()
    df["_date"] = df["date"].dt.normalize()
    df["_month"] = df["date"].dt.to_period("M").astype(str)
    return df


def _make_model(
    feature_cols: list[str],
    method: str,
    checkpoint_path: Optional[Path],
    device: str,
) -> ResMLP:
    checkpoint = None
    checkpoint_model_config = {}
    if method == "TL":
        if checkpoint_path is None:
            raise ValueError("TL finetune requires a checkpoint_path.")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        checkpoint_features = checkpoint.get("feature_cols")
        if checkpoint_features is not None and list(checkpoint_features) != list(feature_cols):
            raise ValueError(
                f"Checkpoint feature_cols={checkpoint_features} do not match "
                f"requested feature_cols={feature_cols}."
            )
        checkpoint_model_config = checkpoint.get("model_config", {})

    model = ResMLP(
        input_dim=len(feature_cols),
        hidden_dim=int(checkpoint_model_config.get("hidden_dim", 22)),
        num_hidden=int(checkpoint_model_config.get("num_hidden", 16)),
        dropout=float(checkpoint_model_config.get("dropout", 0.0)),
    ).to(device)

    if method == "TL":
        model.load_state_dict(checkpoint["state_dict"], strict=True)

    return model


def run_one_finetune(
    heston_dataset: pd.DataFrame,
    theory: str,
    method: str,
    seed: int,
    output_dir: Path,
    test_start_date: str = "2020-01-01",
    val_ratio: float = 0.2,
    tl_lr: float = 1e-4,
    dl_lr: float = 1e-3,
    max_epochs: int = 200,
    early_stop_consecutive: int = 2,
    eps: float = 1e-4,
    batch_denominator: int = 600,
    checkpoint_path: Optional[Path] = None,
    device: Optional[str] = None,
    maturity_bucket: Optional[str] = None,
    maturity_bins: Optional[list[float]] = None,
    maturity_labels: Optional[list[str]] = None,
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

    train_val, test = _prepare_empirical_data(
        heston_dataset=heston_dataset,
        feature_cols=feature_cols,
        test_start_date=test_start_date,
    )
    if maturity_bucket is not None:
        train_val = add_maturity_bucket(
            train_val,
            maturity_bins=maturity_bins,
            maturity_labels=maturity_labels,
        )
        test = add_maturity_bucket(
            test,
            maturity_bins=maturity_bins,
            maturity_labels=maturity_labels,
        )
        train_val = train_val.loc[
            train_val["maturity_bucket"].astype(str).eq(str(maturity_bucket))
        ].copy()
        test = test.loc[
            test["maturity_bucket"].astype(str).eq(str(maturity_bucket))
        ].copy()

    if len(train_val) < 2 or len(test) == 0:
        raise ValueError(
            f"Empty train/test split for {theory}-{method}: "
            f"maturity_bucket={maturity_bucket or 'all'} "
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
        f"\n[{theory}-{method} seed={seed}] "
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
        maturity_bucket=maturity_bucket or "all",
        eps=eps,
    )

    output_dir = Path(output_dir)
    history_dir = output_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)

    run_id = (
        f"{theory.lower()}_{method.lower()}_"
        f"{_path_safe_label(maturity_bucket or 'all')}_seed{seed}"
    )
    prediction_df = prediction_rows(
        test=test,
        pred_test=pred_test,
        run_id=run_id,
        theory=theory,
        method=method,
        seed=seed,
        maturity_bucket=maturity_bucket or "all",
        test_start_date=test_start_date,
        eps=eps,
    )
    history.to_csv(
        history_dir / f"{run_id}.csv",
        index=False,
    )
    pd.DataFrame(monthly_rows).to_csv(
        history_dir / f"{run_id}_monthly_test.csv",
        index=False,
    )

    row = {
        "maturity_bucket": maturity_bucket or "all",
        "theory": theory,
        "method": method,
        "seed": seed,
        "test_start_date": test_start_date,
        "n_trainval": len(train_val),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test),
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
        f"[{theory}-{method} seed={seed}] best_val={best_val:.6f} "
        f"test_weighted_mae={test_weighted_mae:.6f} test_mae={test_mae:.6f} "
        f"bsiv_med={row['test_bsiv_median']:.6f}"
    )
    return row, monthly_rows, prediction_df


def run_finetune_experiments(
    heston_dataset: pd.DataFrame,
    output_dir: Path,
    bs_checkpoint_path: Path,
    heston_checkpoint_path: Path,
    seeds: Iterable[int] = range(10),
    theories: Iterable[str] = ("BS", "HESTON"),
    methods: Iterable[str] = ("DL", "TL"),
    test_start_date: str = "2020-01-01",
    val_ratio: float = 0.2,
    tl_lr: float = 1e-4,
    dl_lr: float = 1e-3,
    max_epochs: int = 200,
    early_stop_consecutive: int = 2,
    use_maturity_buckets: bool = False,
    maturity_bins: Optional[list[float]] = None,
    maturity_labels: Optional[list[str]] = None,
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
    maturity_labels = MATURITY_LABELS if maturity_labels is None else maturity_labels
    maturity_buckets = list(maturity_labels) if use_maturity_buckets else [None]
    for seed in seeds:
        for maturity_bucket in maturity_buckets:
            for theory in theories:
                theory = str(theory).upper()
                for method in methods:
                    method = str(method).upper()
                    try:
                        row, month_rows, pred_rows = run_one_finetune(
                            heston_dataset=heston_dataset,
                            theory=theory,
                            method=method,
                            seed=int(seed),
                            output_dir=output_dir,
                            test_start_date=test_start_date,
                            maturity_bucket=maturity_bucket,
                            maturity_bins=maturity_bins,
                            maturity_labels=maturity_labels,
                            val_ratio=val_ratio,
                            tl_lr=tl_lr,
                            dl_lr=dl_lr,
                            max_epochs=max_epochs,
                            early_stop_consecutive=early_stop_consecutive,
                            checkpoint_path=checkpoint_paths[theory] if method == "TL" else None,
                        )
                        rows.append(row)
                        monthly_rows.extend(month_rows)
                        prediction_frames.append(pred_rows)
                    except Exception as exc:
                        rows.append(
                            {
                                "maturity_bucket": maturity_bucket or "all",
                                "theory": theory,
                                "method": method,
                                "seed": int(seed),
                                "test_start_date": test_start_date,
                                "status": "failed",
                                "error_message": str(exc),
                            }
                        )
                    pd.DataFrame(rows).to_csv(output_dir / "results.csv", index=False)
                    pd.DataFrame(monthly_rows).to_csv(
                        output_dir / "monthly_results.csv",
                        index=False,
                    )
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


def run_one_rolling_finetune(
    heston_dataset: pd.DataFrame,
    theory: str,
    method: str,
    seed: int,
    window: dict,
    output_dir: Path,
    val_ratio: float = 0.2,
    tl_lr: float = 1e-4,
    dl_lr: float = 1e-3,
    max_epochs: int = 200,
    early_stop_consecutive: int = 2,
    eps: float = 1e-4,
    batch_denominator: int = 600,
    checkpoint_path: Optional[Path] = None,
    device: Optional[str] = None,
    maturity_bucket: Optional[str] = None,
    maturity_bins: Optional[list[float]] = None,
    maturity_labels: Optional[list[str]] = None,
) -> tuple[list[dict], pd.DataFrame]:
    theory = str(theory).upper()
    method = str(method).upper()
    if theory not in THEORY_FEATURES:
        raise ValueError(f"Unknown theory={theory!r}. Expected one of {sorted(THEORY_FEATURES)}.")
    if method not in {"DL", "TL"}:
        raise ValueError("method must be 'DL' or 'TL'.")

    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    feature_cols = THEORY_FEATURES[theory]
    df = _prepare_empirical_monthly_data(
        heston_dataset=heston_dataset,
        feature_cols=feature_cols,
    )
    train_dates = set(pd.to_datetime(window["train_dates"]).normalize())
    test_dates = set(pd.to_datetime(window["test_dates"]).normalize())
    train_val = df.loc[df["_date"].isin(train_dates)].copy()
    test = df.loc[df["_date"].isin(test_dates)].copy()

    if maturity_bucket is not None:
        train_val = add_maturity_bucket(
            train_val,
            maturity_bins=maturity_bins,
            maturity_labels=maturity_labels,
        )
        test = add_maturity_bucket(
            test,
            maturity_bins=maturity_bins,
            maturity_labels=maturity_labels,
        )
        train_val = train_val.loc[
            train_val["maturity_bucket"].astype(str).eq(str(maturity_bucket))
        ].copy()
        test = test.loc[
            test["maturity_bucket"].astype(str).eq(str(maturity_bucket))
        ].copy()

    if len(train_val) < 2 or len(test) == 0:
        raise ValueError(
            f"Empty rolling split for {theory}-{method}: "
            f"maturity_bucket={maturity_bucket or 'all'} "
            f"window_id={window['window_id']} n_trainval={len(train_val)}, "
            f"n_test={len(test)}."
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
        f"\n[rolling {window['window_id']} "
        f"{window['train_start_date']}..{window['train_end_date']} -> "
        f"{window['test_start_date']}..{window['test_end_date']} "
        f"{theory}-{method} seed={seed}] "
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
    y_test = test["mid_price"].to_numpy(np.float32)
    pred_test = predict(model, X_test, device=device)

    output_dir = Path(output_dir)
    history_dir = output_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)

    run_id = (
        f"{theory.lower()}_{method.lower()}_{window['window_id']}_"
        f"{_path_safe_label(maturity_bucket or 'all')}_"
        f"train{window['train_start_date']}_{window['train_end_date']}_"
        f"test{window['test_start_date']}_{window['test_end_date']}_seed{seed}"
    )
    prediction_df = prediction_rows(
        test=test,
        pred_test=pred_test,
        run_id=run_id,
        theory=theory,
        method=method,
        seed=seed,
        maturity_bucket=maturity_bucket or "all",
        test_start_date=window["test_start_date"],
        eps=eps,
    )
    history.to_csv(history_dir / f"{run_id}.csv", index=False)

    test_with_pred = test.reset_index(drop=True).copy()
    test_with_pred["_pred"] = pred_test
    rows = []
    eval_groups = (
        [(maturity_bucket, test_with_pred)]
        if maturity_bucket is not None
        else [("all", test_with_pred)]
    )
    for bucket, group in eval_groups:
        pred = group["_pred"].to_numpy(dtype=float)
        y = group["mid_price"].to_numpy(dtype=float)
        delta = group["delta"].to_numpy(dtype=float)
        bsiv_metrics = bsiv_abs_error_median(group, pred)
        row = {
            "window_id": window["window_id"],
            "train_start_date": window["train_start_date"],
            "train_end_date": window["train_end_date"],
            "test_start_date": window["test_start_date"],
            "test_end_date": window["test_end_date"],
            "train_days": window["train_days"],
            "test_days": window["test_days"],
            "step_days": window["step_days"],
            "maturity_bucket": str(bucket),
            "theory": theory,
            "method": method,
            "seed": seed,
            "n_trainval": len(train_val),
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_test_window": len(test),
            "n_test": int(len(group)),
            "batch_size": batch_size,
            "lr": lr,
            "best_val": best_val,
            "test_weighted_mae": weighted_mae_numpy(pred, y, delta, eps=eps),
            "test_mae": float(np.mean(np.abs(pred - y))),
            "test_mse": float(np.mean((pred - y) ** 2)),
            "prediction_path": str(Path(output_dir) / "predictions.parquet"),
            "finetune_checkpoint_path": "",
            "status": "ok",
            "error_message": "",
        }
        row.update(bsiv_metrics)
        rows.append(row)

    print(
        f"[rolling {window['window_id']} {theory}-{method} seed={seed}] "
        f"best_val={best_val:.6f} maturity_buckets={len(rows)}"
    )
    return rows, prediction_df


def run_rolling_finetune_experiments(
    heston_dataset: pd.DataFrame,
    output_dir: Path,
    bs_checkpoint_path: Path,
    heston_checkpoint_path: Path,
    seeds: Iterable[int] = range(10),
    theories: Iterable[str] = ("BS", "HESTON"),
    methods: Iterable[str] = ("DL", "TL"),
    train_days: int = 20,
    test_days: int = 20,
    step_days: int = 1,
    val_ratio: float = 0.2,
    tl_lr: float = 1e-4,
    dl_lr: float = 1e-3,
    max_epochs: int = 200,
    early_stop_consecutive: int = 2,
    use_maturity_buckets: bool = False,
    maturity_bins: Optional[list[float]] = None,
    maturity_labels: Optional[list[str]] = None,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = {
        "BS": Path(bs_checkpoint_path),
        "HESTON": Path(heston_checkpoint_path),
    }

    windows = _make_rolling_date_windows(
        heston_dataset=heston_dataset,
        train_days=train_days,
        test_days=test_days,
        step_days=step_days,
    )
    maturity_labels = MATURITY_LABELS if maturity_labels is None else maturity_labels
    maturity_buckets = list(maturity_labels) if use_maturity_buckets else [None]

    rows = []
    prediction_frames = []
    for seed in seeds:
        for window in windows:
            for maturity_bucket in maturity_buckets:
                for theory in theories:
                    theory = str(theory).upper()
                    for method in methods:
                        method = str(method).upper()
                        try:
                            bucket_rows, pred_rows = run_one_rolling_finetune(
                                heston_dataset=heston_dataset,
                                theory=theory,
                                method=method,
                                seed=int(seed),
                                window=window,
                                output_dir=output_dir,
                                maturity_bucket=maturity_bucket,
                                maturity_bins=maturity_bins,
                                maturity_labels=maturity_labels,
                                val_ratio=val_ratio,
                                tl_lr=tl_lr,
                                dl_lr=dl_lr,
                                max_epochs=max_epochs,
                                early_stop_consecutive=early_stop_consecutive,
                                checkpoint_path=checkpoint_paths[theory] if method == "TL" else None,
                            )
                            rows.extend(bucket_rows)
                            prediction_frames.append(pred_rows)
                        except Exception as exc:
                            row = {
                                "window_id": window["window_id"],
                                "train_start_date": window["train_start_date"],
                                "train_end_date": window["train_end_date"],
                                "test_start_date": window["test_start_date"],
                                "test_end_date": window["test_end_date"],
                                "train_days": window["train_days"],
                                "test_days": window["test_days"],
                                "step_days": window["step_days"],
                                "maturity_bucket": maturity_bucket or "all",
                                "theory": theory,
                                "method": method,
                                "seed": int(seed),
                                "status": "failed",
                                "error_message": str(exc),
                            }
                            rows.append(row)
                        pd.DataFrame(rows).to_csv(output_dir / "rolling_results.csv", index=False)
                        if prediction_frames:
                            pd.concat(prediction_frames, ignore_index=True).to_parquet(
                                output_dir / "predictions.parquet",
                                index=False,
                            )

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "rolling_results.csv", index=False)
    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_parquet(
            output_dir / "predictions.parquet",
            index=False,
        )
    return results
