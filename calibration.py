"""
Heston day-by-day calibration utilities for cleaned SP500 put option data.

This module is intentionally not a script entry point. Calibration defaults live
here; main.py only decides whether to rebuild or load the Heston dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from hsv_bsm_bates_data import heston_price_cos


RAW_REQUIRED_COLUMNS = [
    "date",
    "exdate",
    "mid_price",
    "impl_volatility",
    "tau",
    "delta",
    "gamma",
    "vega",
    "theta",
    "S",
    "K",
    "r",
    "d",
    "IV1",
]

HESTON_PARAM_COLUMNS = ["v0", "theta", "kappa", "xi", "rho"]
DEFAULT_INPUT_PATH = Path("data/option_price_20100101_20250828_raw.parquet")
DEFAULT_HESTON_DATASET_PATH = Path("data/heston_calibrated_dataset_raw.parquet")
DEFAULT_START_DATE = None
DEFAULT_END_DATE = None
DEFAULT_CP_FLAG = "P"
DEFAULT_MIN_OPTIONS_PER_DAY = 50
DEFAULT_MAX_OPTIONS_PER_DAY = 2_000
DEFAULT_N_COS = 128
DEFAULT_L_COS = 12.0
DEFAULT_MAX_NFEV = 80
DEFAULT_LOSS = "soft_l1"
DEFAULT_F_SCALE = 0.01
DEFAULT_WEIGHT_SCHEME = "vega"
DEFAULT_RANDOM_SEED = 42
DEFAULT_KEEP_FAILED_TARGET_DAYS = False
DEFAULT_PARAMETER_BOUNDS = {
    "v0": (1e-6, 1.0),
    "theta": (1e-6, 1.0),
    "kappa": (0.05, 10.0),
    "xi": (0.01, 3.0),
    "rho": (-0.99, 0.99),
}


@dataclass(frozen=True)
class HestonCalibrationConfig:
    output_dataset_path: Path = DEFAULT_HESTON_DATASET_PATH
    input_path: Path = DEFAULT_INPUT_PATH
    output_params_path: Optional[Path] = None
    start_date: Optional[str] = DEFAULT_START_DATE
    end_date: Optional[str] = DEFAULT_END_DATE
    cp_flag: str = DEFAULT_CP_FLAG
    min_options_per_day: int = DEFAULT_MIN_OPTIONS_PER_DAY
    max_options_per_day: Optional[int] = DEFAULT_MAX_OPTIONS_PER_DAY
    n_cos: int = DEFAULT_N_COS
    l_cos: float = DEFAULT_L_COS
    max_nfev: int = DEFAULT_MAX_NFEV
    loss: str = DEFAULT_LOSS
    f_scale: float = DEFAULT_F_SCALE
    weight_scheme: str = DEFAULT_WEIGHT_SCHEME
    parameter_bounds: Mapping[str, tuple[float, float]] = field(
        default_factory=lambda: DEFAULT_PARAMETER_BOUNDS.copy()
    )
    random_seed: int = DEFAULT_RANDOM_SEED
    keep_failed_target_days: bool = DEFAULT_KEEP_FAILED_TARGET_DAYS


def read_option_data(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path, columns=RAW_REQUIRED_COLUMNS)
    elif suffix == ".csv":
        df = pd.read_csv(path, usecols=RAW_REQUIRED_COLUMNS)
    elif suffix in {".pkl", ".pickle"}:
        df = pd.read_pickle(path)
        df = df[RAW_REQUIRED_COLUMNS]
    else:
        raise ValueError(f"Unsupported input format {suffix!r}. Use parquet, csv, or pickle.")

    missing = sorted(set(RAW_REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["exdate"] = pd.to_datetime(df["exdate"])

    # Keep the Heston latent parameter name "theta" available.
    df = df.rename(columns={"theta": "option_theta"})
    return df.sort_values(["date", "exdate", "K"]).reset_index(drop=True)


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    if suffix in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
    elif suffix == ".csv":
        df.to_csv(path, index=False)
    elif suffix in {".pkl", ".pickle"}:
        df.to_pickle(path)
    else:
        raise ValueError(f"Unsupported output format {suffix!r}. Use parquet, csv, or pickle.")


def read_dataframe(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported input format {suffix!r}. Use parquet, csv, or pickle.")


def default_params_path(dataset_path: Path) -> Path:
    return dataset_path.with_name(f"{dataset_path.stem}_day_by_day_parameters{dataset_path.suffix}")


def restrict_date_range(
    dates: list[pd.Timestamp],
    start_date: Optional[str],
    end_date: Optional[str],
) -> list[pd.Timestamp]:
    selected = dates
    if start_date is not None:
        start_ts = pd.Timestamp(start_date)
        selected = [d for d in selected if d >= start_ts]
    if end_date is not None:
        end_ts = pd.Timestamp(end_date)
        selected = [d for d in selected if d <= end_ts]
    return selected


def clean_option_rows(df: pd.DataFrame) -> pd.DataFrame:
    finite_columns = [
        "mid_price",
        "impl_volatility",
        "tau",
        "delta",
        "gamma",
        "vega",
        "option_theta",
        "S",
        "K",
        "r",
        "d",
        "IV1",
    ]

    out = df.copy()
    finite_mask = np.ones(len(out), dtype=bool)
    for col in finite_columns:
        finite_mask &= np.isfinite(out[col].to_numpy(dtype=float))

    return out.loc[
        finite_mask
        & (out["mid_price"] > 0.0)
        & (out["tau"] > 0.0)
        & (out["S"] > 0.0)
        & (out["K"] > 0.0)
    ].copy()


def stratified_day_sample(
    day_df: pd.DataFrame,
    max_options: Optional[int],
    random_seed: int,
) -> pd.DataFrame:
    if max_options is None or len(day_df) <= max_options:
        return day_df

    out = day_df.copy()
    log_moneyness = np.log(out["K"].to_numpy(dtype=float) / out["S"].to_numpy(dtype=float))

    try:
        out["_tau_bin"] = pd.qcut(out["tau"], q=5, duplicates="drop")
        out["_m_bin"] = pd.qcut(log_moneyness, q=5, duplicates="drop")
    except ValueError:
        return out.sample(n=max_options, random_state=random_seed).sort_index()

    sampled_parts = []
    grouped = out.groupby(["_tau_bin", "_m_bin"], observed=True, sort=False)
    n_groups = max(grouped.ngroups, 1)
    per_group = max(int(np.ceil(max_options / n_groups)), 1)

    for _, group in grouped:
        n_take = min(len(group), per_group)
        sampled_parts.append(group.sample(n=n_take, random_state=random_seed))

    sampled = pd.concat(sampled_parts, ignore_index=False)
    if len(sampled) > max_options:
        sampled = sampled.sample(n=max_options, random_state=random_seed)

    return sampled.drop(columns=["_tau_bin", "_m_bin"]).sort_index()


def parameter_bounds_arrays(
    parameter_bounds: Mapping[str, tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.array([parameter_bounds[col][0] for col in HESTON_PARAM_COLUMNS], dtype=float)
    upper = np.array([parameter_bounds[col][1] for col in HESTON_PARAM_COLUMNS], dtype=float)
    return lower, upper


def initial_heston_guess(
    day_df: pd.DataFrame,
    parameter_bounds: Mapping[str, tuple[float, float]],
    previous_params: Optional[np.ndarray],
) -> np.ndarray:
    if previous_params is not None and np.all(np.isfinite(previous_params)):
        return np.asarray(previous_params, dtype=float)

    iv = day_df["impl_volatility"].to_numpy(dtype=float)
    iv = iv[np.isfinite(iv) & (iv > 0.0)]
    if len(iv) == 0:
        iv = day_df["IV1"].to_numpy(dtype=float)
        iv = iv[np.isfinite(iv) & (iv > 0.0)]

    vol0 = float(np.median(iv)) if len(iv) else 0.20
    var0 = float(np.clip(vol0**2, *parameter_bounds["v0"]))
    return np.array([var0, var0, 2.0, 0.50, -0.50], dtype=float)


def residual_weights(day_df: pd.DataFrame, scheme: str) -> np.ndarray:
    scheme = scheme.lower()
    n = len(day_df)

    if scheme == "equal":
        weights = np.ones(n, dtype=float)
    elif scheme == "vega":
        weights = np.abs(day_df["vega"].to_numpy(dtype=float))
        positive = weights[np.isfinite(weights) & (weights > 0.0)]
        floor = np.percentile(positive, 5) if len(positive) else 1.0
        weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, floor)
        weights = np.clip(weights, floor, np.percentile(weights, 95))
    elif scheme == "inverse_price":
        price = day_df["mid_price"].to_numpy(dtype=float)
        weights = 1.0 / np.maximum(np.abs(price), 1e-4)
    else:
        raise ValueError("weight_scheme must be one of: equal, vega, inverse_price")

    weights = np.asarray(weights, dtype=float)
    weights = weights / max(float(np.mean(weights)), 1e-12)
    return np.sqrt(weights)


def heston_prices_for_day(
    params: Iterable[float],
    day_df: pd.DataFrame,
    cp_flag: str,
    n_cos: int,
    l_cos: float,
) -> np.ndarray:
    v0, theta, kappa, xi, rho = [float(x) for x in params]
    prices = np.empty(len(day_df), dtype=float)

    arrays = {
        "S": day_df["S"].to_numpy(dtype=float),
        "K": day_df["K"].to_numpy(dtype=float),
        "tau": day_df["tau"].to_numpy(dtype=float),
        "r": day_df["r"].to_numpy(dtype=float),
        "d": day_df["d"].to_numpy(dtype=float),
    }

    for i in range(len(day_df)):
        prices[i] = heston_price_cos(
            S0=arrays["S"][i],
            K=arrays["K"][i],
            T=arrays["tau"][i],
            r=arrays["r"][i],
            q=arrays["d"][i],
            v0=v0,
            kappa=kappa,
            theta=theta,
            xi=xi,
            rho=rho,
            cp_flag=cp_flag,
            N=n_cos,
            L=l_cos,
        )

    return prices


def calibrate_one_heston_day(
    day_df: pd.DataFrame,
    config: HestonCalibrationConfig,
    previous_params: Optional[np.ndarray],
) -> dict:
    try:
        from scipy.optimize import least_squares
    except ImportError as exc:
        raise ImportError("Heston calibration requires scipy.optimize.least_squares.") from exc

    market = day_df["mid_price"].to_numpy(dtype=float)
    weights = residual_weights(day_df, config.weight_scheme)
    lower, upper = parameter_bounds_arrays(config.parameter_bounds)
    x0 = np.clip(
        initial_heston_guess(day_df, config.parameter_bounds, previous_params),
        lower,
        upper,
    )

    def objective(x: np.ndarray) -> np.ndarray:
        try:
            model = heston_prices_for_day(
                params=x,
                day_df=day_df,
                cp_flag=config.cp_flag,
                n_cos=config.n_cos,
                l_cos=config.l_cos,
            )
        except Exception:
            return np.full_like(market, 1e6, dtype=float)

        residuals = (model - market) * weights
        return np.where(np.isfinite(residuals), residuals, 1e6)

    result = least_squares(
        objective,
        x0=x0,
        bounds=(lower, upper),
        loss=config.loss,
        f_scale=config.f_scale,
        max_nfev=config.max_nfev,
    )

    params = result.x.astype(float)
    fitted = heston_prices_for_day(
        params=params,
        day_df=day_df,
        cp_flag=config.cp_flag,
        n_cos=config.n_cos,
        l_cos=config.l_cos,
    )
    err = fitted - market

    return {
        "calibration_success": bool(result.success),
        "calibration_status": int(result.status),
        "calibration_message": str(result.message),
        "calibration_objective_value": float(result.cost),
        "calibration_rmse": float(np.sqrt(np.mean(err**2))),
        "calibration_mae": float(np.mean(np.abs(err))),
        "calibration_nfev": int(result.nfev),
        "v0": float(params[0]),
        "theta": float(params[1]),
        "kappa": float(params[2]),
        "xi": float(params[3]),
        "rho": float(params[4]),
    }


def calibrate_heston_by_day(
    option_df: pd.DataFrame,
    config: HestonCalibrationConfig,
) -> pd.DataFrame:
    all_dates = sorted(pd.to_datetime(option_df["date"].unique()))
    previous_date_by_target = dict(zip(all_dates[1:], all_dates[:-1]))
    target_dates = restrict_date_range(all_dates[1:], config.start_date, config.end_date)

    grouped = option_df.groupby("date", sort=True)
    previous_params: Optional[np.ndarray] = None
    rows = []

    for target_date in target_dates:
        calibration_date = previous_date_by_target[target_date]
        raw_calibration_df = grouped.get_group(calibration_date)
        clean_calibration_df = clean_option_rows(raw_calibration_df)

        row = {
            "date": target_date,
            "calibration_date": calibration_date,
            "cp_flag": config.cp_flag,
            "n_options_raw": int(len(raw_calibration_df)),
            "n_options_clean": int(len(clean_calibration_df)),
            "n_options_used": 0,
        }

        if len(clean_calibration_df) < config.min_options_per_day:
            row.update(
                {
                    "calibration_success": False,
                    "calibration_status": -1,
                    "calibration_message": "Too few clean options for calibration.",
                    "calibration_objective_value": np.nan,
                    "calibration_rmse": np.nan,
                    "calibration_mae": np.nan,
                    "calibration_nfev": 0,
                    "v0": np.nan,
                    "theta": np.nan,
                    "kappa": np.nan,
                    "xi": np.nan,
                    "rho": np.nan,
                }
            )
        else:
            used_df = stratified_day_sample(
                clean_calibration_df,
                max_options=config.max_options_per_day,
                random_seed=config.random_seed,
            )
            row["n_options_used"] = int(len(used_df))
            fit = calibrate_one_heston_day(used_df, config, previous_params)
            row.update(fit)

            if fit["calibration_success"]:
                previous_params = np.array(
                    [fit["v0"], fit["theta"], fit["kappa"], fit["xi"], fit["rho"]],
                    dtype=float,
                )

        rows.append(row)
        print(
            f"{target_date.date()} <- {calibration_date.date()} | "
            f"success={row['calibration_success']} | "
            f"used={row['n_options_used']} | "
            f"rmse={row['calibration_rmse']}"
        )

    return pd.DataFrame(rows)


def make_heston_training_dataset(
    option_df: pd.DataFrame,
    params_df: pd.DataFrame,
    keep_failed_target_days: bool,
) -> pd.DataFrame:
    target_df = clean_option_rows(option_df)

    param_columns = [
        "date",
        "calibration_date",
        "calibration_success",
        "calibration_rmse",
        "calibration_mae",
        "n_options_used",
        *HESTON_PARAM_COLUMNS,
    ]
    merge_params = params_df[param_columns].copy()
    if not keep_failed_target_days:
        merge_params = merge_params.loc[merge_params["calibration_success"]].copy()

    dataset = target_df.merge(merge_params, on="date", how="inner")
    dataset["model"] = "HestonCalibrated"
    return dataset.sort_values(["date", "exdate", "K"]).reset_index(drop=True)


def build_heston_calibrated_dataset(
    config: HestonCalibrationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    option_df = read_option_data(config.input_path)
    params_df = calibrate_heston_by_day(option_df, config)
    heston_dataset = make_heston_training_dataset(
        option_df=option_df,
        params_df=params_df,
        keep_failed_target_days=config.keep_failed_target_days,
    )

    params_path = config.output_params_path or default_params_path(config.output_dataset_path)
    write_dataframe(params_df, params_path)
    write_dataframe(heston_dataset, config.output_dataset_path)
    return heston_dataset, params_df


def get_heston_dataset(
    heston_dataset_path: Path,
    run_heston_calibration: bool,
) -> pd.DataFrame:
    if run_heston_calibration:
        heston_dataset, _ = build_heston_calibrated_dataset(
            HestonCalibrationConfig(output_dataset_path=heston_dataset_path)
        )
        return heston_dataset

    return read_dataframe(heston_dataset_path)
