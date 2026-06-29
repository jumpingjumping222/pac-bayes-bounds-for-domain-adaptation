import copy
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from hsv_bsm_bates_data import bs_delta, bs_price, option_bounds, safe_clip_price


DEFAULT_SYNTHETIC_DATA_ROOT = Path("output/synthetic_data")


BS_DATA_PRESETS = {
    "bs_indsk_tau_u_sigma_u": {
        "theory": "BS",
        "sampling_mode": "independent_sk",
        "cp_flag": "P",
        "ranges": {
            "S": [1e-4, 5.0],
            "K": [1e-4, 5.0],
            "m": [-0.5, 0.5],
            "tau": [1.0 / 252.0, 4.0],
            "r": [0.0, 0.1],
            "d": [0.0, 0.1],
            "sigma": [1e-4, 1.0],
        },
        "tau_distribution": "uniform",
        "sigma_distribution": "uniform",
        "filter_K_range": True,
    },
    "bs_lm_m05_tau_u_sigma_u": {
        "theory": "BS",
        "sampling_mode": "log_moneyness",
        "cp_flag": "P",
        "ranges": {
            "S": [1e-4, 5.0],
            "K": [1e-4, 5.0],
            "m": [-0.5, 0.5],
            "tau": [1.0 / 252.0, 4.0],
            "r": [0.0, 0.1],
            "d": [0.0, 0.1],
            "sigma": [1e-4, 1.0],
        },
        "tau_distribution": "uniform",
        "sigma_distribution": "uniform",
        "filter_K_range": True,
    },
    "bs_lm_m05_tau_short_sigma_logn": {
        "theory": "BS",
        "sampling_mode": "log_moneyness",
        "cp_flag": "P",
        "ranges": {
            "S": [1e-4, 5.0],
            "K": [1e-4, 5.0],
            "m": [-0.5, 0.5],
            "tau": [1.0 / 252.0, 4.0],
            "r": [0.0, 0.1],
            "d": [0.0, 0.1],
            "sigma": [1e-4, 1.0],
        },
        "tau_distribution": "short_heavy",
        "sigma_distribution": "lognormal",
        "sigma_lognormal": {
            "median": 0.25,
            "shape": 0.45,
        },
        "filter_K_range": True,
    },
}


def _compact_n(n_samples: int) -> str:
    n_samples = int(n_samples)
    if n_samples % 1_000_000 == 0:
        return f"{n_samples // 1_000_000}m"
    if n_samples % 1_000 == 0:
        return f"{n_samples // 1_000}k"
    return str(n_samples)


def _normalize_range(value) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"Range must have two values, got {value!r}.")
    lo, hi = float(value[0]), float(value[1])
    if not lo < hi:
        raise ValueError(f"Range lower bound must be < upper bound, got {value!r}.")
    return lo, hi


def _sample_uniform(rng: np.random.Generator, bounds) -> float:
    return float(rng.uniform(*_normalize_range(bounds)))


def _sample_log_uniform(rng: np.random.Generator, bounds) -> float:
    lo, hi = _normalize_range(bounds)
    lo = max(lo, 1e-12)
    hi = max(hi, lo * (1.0 + 1e-12))
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _sample_short_heavy(rng: np.random.Generator, bounds) -> float:
    lo, hi = _normalize_range(bounds)
    z = rng.beta(0.7, 2.5)
    return float(lo + (hi - lo) * z)


def _sample_tau(rng: np.random.Generator, config: dict) -> float:
    distribution = str(config.get("tau_distribution", "uniform")).lower()
    bounds = config["ranges"]["tau"]
    if distribution == "uniform":
        return _sample_uniform(rng, bounds)
    if distribution == "log_uniform":
        return _sample_log_uniform(rng, bounds)
    if distribution == "short_heavy":
        return _sample_short_heavy(rng, bounds)
    raise ValueError(
        "tau_distribution must be one of: uniform, log_uniform, short_heavy."
    )


def _sample_sigma(rng: np.random.Generator, config: dict) -> float:
    distribution = str(config.get("sigma_distribution", "uniform")).lower()
    bounds = config["ranges"]["sigma"]
    lo, hi = _normalize_range(bounds)
    if distribution == "uniform":
        return _sample_uniform(rng, bounds)
    if distribution == "log_uniform":
        return _sample_log_uniform(rng, bounds)
    if distribution == "lognormal":
        params = config.get("sigma_lognormal", {})
        median = float(params.get("median", 0.25))
        shape = float(params.get("shape", 0.45))
        for _ in range(100):
            sigma = float(rng.lognormal(mean=np.log(median), sigma=shape))
            if lo <= sigma <= hi:
                return sigma
        return float(np.clip(sigma, lo, hi))
    if distribution == "trunc_normal":
        params = config.get("sigma_trunc_normal", {})
        mean = float(params.get("mean", 0.25))
        std = float(params.get("std", 0.10))
        for _ in range(100):
            sigma = float(rng.normal(mean, std))
            if lo <= sigma <= hi:
                return sigma
        return float(np.clip(sigma, lo, hi))
    raise ValueError(
        "sigma_distribution must be one of: uniform, log_uniform, lognormal, trunc_normal."
    )


def make_bs_data_config(
    preset: str = "bs_lm_m05_tau_u_sigma_u",
    n_samples: int = 800_000,
    seed: int = 123,
    data_id: Optional[str] = None,
    overrides: Optional[dict] = None,
) -> dict:
    if preset not in BS_DATA_PRESETS:
        raise ValueError(
            f"Unknown BS data preset={preset!r}. Expected one of {sorted(BS_DATA_PRESETS)}."
        )

    config = copy.deepcopy(BS_DATA_PRESETS[preset])
    if overrides:
        for key, value in overrides.items():
            if key == "ranges":
                config.setdefault("ranges", {}).update(value)
            else:
                config[key] = value

    config["preset"] = preset
    config["n_samples"] = int(n_samples)
    config["seed"] = int(seed)
    config["data_id"] = data_id or f"{preset}_n{_compact_n(n_samples)}_s{int(seed)}"
    return config


def generate_bs_synthetic_dataset(config: dict) -> pd.DataFrame:
    theory = str(config.get("theory", "BS")).upper()
    if theory != "BS":
        raise ValueError("generate_bs_synthetic_dataset only supports theory='BS'.")

    rng = np.random.default_rng(int(config.get("seed", 123)))
    cp_flag = str(config.get("cp_flag", "P")).upper()
    sampling_mode = str(config.get("sampling_mode", "log_moneyness")).lower()
    ranges = config["ranges"]
    filter_k_range = bool(config.get("filter_K_range", True))

    rows = []
    attempts = 0
    n_samples = int(config.get("n_samples", 100_000))
    max_attempts = int(config.get("max_attempts", n_samples * 10))

    while len(rows) < n_samples and attempts < max_attempts:
        attempts += 1

        S = _sample_uniform(rng, ranges["S"])
        tau = _sample_tau(rng, config)
        r = _sample_uniform(rng, ranges["r"])
        d = _sample_uniform(rng, ranges["d"])
        sigma = _sample_sigma(rng, config)
        F = S * np.exp((r - d) * tau)

        if sampling_mode == "log_moneyness":
            m = _sample_uniform(rng, ranges["m"])
            K = F * np.exp(m)
            if filter_k_range:
                k_lo, k_hi = _normalize_range(ranges["K"])
                if not (k_lo <= K <= k_hi):
                    continue
        elif sampling_mode == "independent_sk":
            K = _sample_uniform(rng, ranges["K"])
            m = float(np.log(K / F))
        else:
            raise ValueError("sampling_mode must be one of: log_moneyness, independent_sk.")

        price = bs_price(
            S=S,
            K=K,
            T=tau,
            r=r,
            q=d,
            sigma=sigma,
            cp_flag=cp_flag,
        )
        if not np.isfinite(price):
            continue

        lower, upper = option_bounds(S, K, tau, r, d, cp_flag)
        price = safe_clip_price(price, lower, upper)
        delta = bs_delta(
            S=S,
            K=K,
            T=tau,
            r=r,
            q=d,
            sigma=sigma,
            cp_flag=cp_flag,
        )
        if not np.isfinite(delta):
            continue

        rows.append(
            {
                "S": S,
                "K": K,
                "tau": tau,
                "r": r,
                "d": d,
                "m": m,
                "cp_flag": cp_flag,
                "IV1": sigma,
                "sigma": sigma,
                "y_put": price,
                "delta_put": delta,
                "epsilon": 0.0,
                "model": "BS",
                "data_id": config["data_id"],
                "sampling_mode": sampling_mode,
            }
        )

    df = pd.DataFrame(rows)
    if len(df) < n_samples:
        print(
            f"Warning: only generated {len(df)} valid BS samples out of "
            f"{n_samples} requested."
        )
    return df


def _write_json(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _write_summary(df: pd.DataFrame, path: Path) -> None:
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        pd.DataFrame().to_csv(path, index=False)
        return

    quantiles = numeric.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    quantiles.columns = [f"p{int(q * 100):02d}" for q in quantiles.columns]
    summary = pd.concat(
        [
            numeric.agg(["count", "mean", "std", "min", "max"]).T,
            quantiles,
        ],
        axis=1,
    )
    summary.insert(0, "column", summary.index)
    summary.reset_index(drop=True).to_csv(path, index=False)


def synthetic_dataset_dir(
    data_config: dict,
    output_root: Path = DEFAULT_SYNTHETIC_DATA_ROOT,
) -> Path:
    theory = str(data_config.get("theory", "BS")).upper()
    return Path(output_root) / theory / str(data_config["data_id"])


def load_or_generate_synthetic_dataset(
    data_config: dict,
    output_root: Path = DEFAULT_SYNTHETIC_DATA_ROOT,
    regenerate: bool = False,
) -> tuple[pd.DataFrame, dict]:
    data_dir = synthetic_dataset_dir(data_config, output_root=output_root)
    dataset_path = data_dir / "dataset.parquet"
    config_path = data_dir / "data_config.json"
    summary_path = data_dir / "summary.csv"

    if dataset_path.exists() and not regenerate:
        df = pd.read_parquet(dataset_path)
    else:
        theory = str(data_config.get("theory", "BS")).upper()
        if theory != "BS":
            raise ValueError("Synthetic data generator currently supports only theory='BS'.")
        df = generate_bs_synthetic_dataset(data_config)
        data_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dataset_path, index=False)
        _write_summary(df, summary_path)

    data_dir.mkdir(parents=True, exist_ok=True)
    materialized = copy.deepcopy(data_config)
    materialized["dataset_path"] = str(dataset_path)
    materialized["config_path"] = str(config_path)
    materialized["summary_path"] = str(summary_path)
    _write_json(materialized, config_path)

    return df, materialized
