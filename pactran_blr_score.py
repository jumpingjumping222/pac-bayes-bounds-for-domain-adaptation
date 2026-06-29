from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch

from model import ResMLP


MATURITY_BINS = [0.0, 1.0 / 12.0, 3.0 / 12.0, 6.0 / 12.0, 1.0, 2.0, 3.0]
MATURITY_LABELS = ["0-1m", "1-3m", "3-6m", "6m-1y", "1y-2y", "2y-3y"]


def forward_features(model: ResMLP, X: torch.Tensor) -> torch.Tensor:
    h = model.act(model.inp(X))
    for layer in model.layers:
        h = layer(h)
    return h


def _load_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def _make_model(spec: dict, device: str) -> ResMLP:
    checkpoint = torch.load(spec["checkpoint_path"], map_location=device)
    checkpoint_model_config = checkpoint.get("model_config", {}) if isinstance(checkpoint, dict) else {}
    model = ResMLP(
        input_dim=int(spec["input_dim"]),
        hidden_dim=int(spec.get("hidden_dim", checkpoint_model_config.get("hidden_dim", 22))),
        num_hidden=int(spec.get("num_hidden", checkpoint_model_config.get("num_hidden", 16))),
        dropout=float(spec.get("dropout", checkpoint_model_config.get("dropout", 0.0))),
    ).to(device)

    model.load_state_dict(_load_state_dict(checkpoint), strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def _top_layer_theta(model: ResMLP, dtype: torch.dtype) -> torch.Tensor:
    w = model.out.weight.detach().reshape(-1).to(dtype=dtype)
    b = model.out.bias.detach().reshape(-1).to(dtype=dtype)
    return torch.cat([w, b], dim=0)


def _make_prior_center_theta(
    model: ResMLP,
    dtype: torch.dtype,
    prior_center: str,
) -> torch.Tensor:
    theta_pre = _top_layer_theta(model, dtype=dtype).reshape(-1, 1).cpu()
    prior_center = str(prior_center).lower()
    if prior_center == "pretrained":
        return theta_pre
    if prior_center == "zero":
        return torch.zeros_like(theta_pre)
    raise ValueError("prior_center must be 'pretrained' or 'zero'.")


def _prepare_target_data(
    target_dataset: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    date_before: Optional[str],
    only_put: bool,
) -> pd.DataFrame:
    required = [target_col, "date", *feature_cols]
    if only_put and "cp_flag" in target_dataset.columns:
        required.append("cp_flag")

    missing = sorted(set(required) - set(target_dataset.columns))
    if missing:
        raise ValueError(f"Target dataset is missing required columns: {missing}")

    df = target_dataset.copy()
    df["date"] = pd.to_datetime(df["date"])
    if date_before is not None:
        df = df.loc[df["date"] < pd.Timestamp(date_before)].copy()
    if only_put and "cp_flag" in df.columns:
        df = df.loc[df["cp_flag"].astype(str).str.upper() == "P"].copy()

    df = df.dropna(subset=required).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("No usable target samples after filtering.")
    return df


def _daily_group_key(date_value) -> str:
    return pd.Timestamp(date_value).strftime("%Y-%m-%d")


def _month_str(value) -> str:
    return str(pd.Period(value, freq="M"))


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


def _date_str(value) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _path_safe_label(value: str) -> str:
    return str(value).replace("/", "-").replace(" ", "_")


def _make_rolling_date_windows(
    target_dataset: pd.DataFrame,
    train_days: int,
    test_days: int,
    step_days: int,
) -> list[dict]:
    if train_days <= 0 or test_days <= 0 or step_days <= 0:
        raise ValueError("train_days, test_days, and step_days must be positive.")

    dates = (
        pd.to_datetime(target_dataset["date"])
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


def _feature_matrix(
    model: ResMLP,
    X_np: np.ndarray,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    chunks = []
    with torch.no_grad():
        for start in range(0, len(X_np), batch_size):
            xb = torch.tensor(
                X_np[start:start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            features = forward_features(model, xb).detach().cpu().to(dtype=torch.float64)
            chunks.append(features)

    Phi = torch.cat(chunks, dim=0)
    intercept = torch.ones((Phi.shape[0], 1), dtype=torch.float64)
    return torch.cat([Phi, intercept], dim=1)


def _standardize_hidden_features(
    Phi: torch.Tensor,
    theta_pre: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    hidden = Phi[:, :-1]
    intercept = Phi[:, -1:]

    mu = hidden.mean(dim=0)
    std = torch.clamp(hidden.std(dim=0, unbiased=False), min=eps)
    hidden_scaled = (hidden - mu) / std

    w_pre = theta_pre[:-1]
    b_pre = theta_pre[-1:]
    theta_pre_scaled = torch.cat(
        [
            std.reshape(-1, 1) * w_pre,
            b_pre + (mu.reshape(1, -1) @ w_pre).reshape(1, 1),
        ],
        dim=0,
    )

    stats = {
        "hidden_mean": mu,
        "hidden_std": std,
    }
    return torch.cat([hidden_scaled, intercept], dim=1), theta_pre_scaled, stats


def _solve_with_jitter(A: torch.Tensor, rhs: torch.Tensor, jitter: float):
    eye = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
    last_error = None
    for scale in (0.0, 1.0, 10.0, 100.0, 1000.0):
        try:
            return torch.linalg.solve(A + scale * jitter * eye, rhs)
        except RuntimeError as exc:
            last_error = exc
    raise last_error


def _slogdet_with_jitter(A: torch.Tensor, jitter: float):
    eye = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
    last_sign = None
    last_logdet = None
    for scale in (0.0, 1.0, 10.0, 100.0, 1000.0):
        sign, logdet = torch.linalg.slogdet(A + scale * jitter * eye)
        last_sign, last_logdet = sign, logdet
        if sign > 0 and torch.isfinite(logdet):
            return sign, logdet
    return last_sign, last_logdet


def _mean_std(series: pd.Series) -> tuple[float, float]:
    return float(series.mean()), float(series.std(ddof=0))


def _score_blr_sample(
    model: ResMLP,
    sample_df: pd.DataFrame,
    sample_id: str,
    feature_cols: list[str],
    target_col: str,
    theta_pre: torch.Tensor,
    sigma2: float,
    sigma_pi2: float,
    device: str,
    batch_size: int,
    jitter: float,
) -> tuple[dict, dict]:
    Phi = _feature_matrix(
        model=model,
        X_np=sample_df[feature_cols].to_numpy(np.float32),
        device=device,
        batch_size=batch_size,
    )
    y = torch.tensor(
        sample_df[target_col].to_numpy(np.float64).reshape(-1, 1),
        dtype=torch.float64,
    )

    n_samples, top_layer_dim = Phi.shape
    if theta_pre.shape[0] != top_layer_dim:
        raise ValueError(
            f"Top-layer dim mismatch: theta_pre has {theta_pre.shape[0]}, "
            f"Phi has {top_layer_dim} columns."
        )
    Phi, theta_pre, phi_standardization = _standardize_hidden_features(Phi, theta_pre)

    G = Phi.T @ Phi
    eye = torch.eye(top_layer_dim, dtype=torch.float64)
    A = G / float(sigma2) + eye / float(sigma_pi2)
    rhs = (Phi.T @ y) / float(sigma2) + theta_pre / float(sigma_pi2)
    theta_bar = _solve_with_jitter(A, rhs, jitter=jitter)
    A_inv = _solve_with_jitter(A, eye, jitter=jitter)

    sign_A, logdet_A = _slogdet_with_jitter(A, jitter=jitter)
    if sign_A <= 0 or not torch.isfinite(logdet_A):
        raise ValueError("Failed to compute positive log determinant for posterior precision.")

    residual = y - Phi @ theta_bar
    posterior_mean_sse = (residual.T @ residual).squeeze()
    posterior_cov_sse = torch.trace(G @ A_inv)
    expected_sse = posterior_mean_sse + posterior_cov_sse

    expected_train_nll_total = (
        0.5 * n_samples * np.log(2.0 * np.pi * float(sigma2))
        + 0.5 * float(expected_sse.item()) / float(sigma2)
    )

    theta_diff = theta_bar - theta_pre
    trace_A_inv = float(torch.trace(A_inv).item())
    theta_diff_sq = float((theta_diff.T @ theta_diff).squeeze().item())
    logdet_A_value = float(logdet_A.item())
    pac_score_1 = (
        0.5 * float(expected_sse.item()) / float(sigma2)
        + 0.5
        * (
            (trace_A_inv + theta_diff_sq) / float(sigma_pi2)
            + logdet_A_value
        )
    )
    constant_part = (
        0.5 * n_samples * np.log(2.0 * np.pi * float(sigma2))
        + 0.5
        * (
            -top_layer_dim
            + top_layer_dim * np.log(float(sigma_pi2))
        )
    )
    kl = 0.5 * (
        (trace_A_inv + theta_diff_sq)
        / float(sigma_pi2)
        - top_layer_dim
        + top_layer_dim * np.log(float(sigma_pi2))
        + logdet_A_value
    )
    bound_objective_total = expected_train_nll_total + kl

    row = {
        "sample_id": sample_id,
        "n_samples": int(n_samples),
        "posterior_mean_sse": float(posterior_mean_sse.item()),
        "posterior_cov_sse": float(posterior_cov_sse.item()),
        "expected_sse": float(expected_sse.item()),
        "trace_A_inv": trace_A_inv,
        "theta_diff_sq": theta_diff_sq,
        "logdet_A": logdet_A_value,
        "constant_part": constant_part,
        "pac_score_1": pac_score_1,
        "expected_train_nll_total": expected_train_nll_total,
        "kl": kl,
        "bound_objective_total": bound_objective_total,
    }
    posterior = {
        "sample_id": sample_id,
        "n_samples": int(n_samples),
        "theta_pre": theta_pre,
        "phi_standardization": phi_standardization,
        "theta_bar": theta_bar,
        "A": A,
        "A_inv": A_inv,
        "posterior_mean_sse": float(posterior_mean_sse.item()),
        "posterior_cov_sse": float(posterior_cov_sse.item()),
        "expected_sse": float(expected_sse.item()),
        "trace_A_inv": trace_A_inv,
        "theta_diff_sq": theta_diff_sq,
        "logdet_A": logdet_A_value,
        "constant_part": constant_part,
        "pac_score_1": pac_score_1,
        "expected_train_nll_total": expected_train_nll_total,
        "kl": kl,
        "bound_objective_total": bound_objective_total,
    }
    return row, posterior


def score_checkpoint_blr(
    spec: dict,
    target_dataset: pd.DataFrame,
    target_col: str = "mid_price",
    date_before: Optional[str] = "2020-01-01",
    only_put: bool = True,
    sigma2: float = 0.5,
    sigma_pi2: float = 0.01,
    prior_center: str = "pretrained",
    device: Optional[str] = None,
    batch_size: int = 8192,
    jitter: float = 1e-8,
) -> tuple[dict, dict]:
    """
    PACTran-style top-layer PAC-Bayes regression score.

    After freezing the pretrained ResMLP feature extractor, the final layer is a
    Bayesian linear regression problem with a Gaussian prior centered at the
    pretrained final-layer weights. The score is the optimized PAC-Bayes
    objective n * E_Q[train NLL] + KL(Q || P). The objective is computed
    separately for each trading day and then averaged across days.
    """
    if sigma2 <= 0 or sigma_pi2 <= 0:
        raise ValueError("sigma2 and sigma_pi2 must be positive.")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    feature_cols = list(spec["feature_cols"])
    df = _prepare_target_data(
        target_dataset=target_dataset,
        feature_cols=feature_cols,
        target_col=target_col,
        date_before=date_before,
        only_put=only_put,
    )

    model = _make_model(spec, device=device)
    theta_pre = _make_prior_center_theta(
        model=model,
        dtype=torch.float64,
        prior_center=prior_center,
    )
    daily_rows = []
    daily_posteriors = []
    for date_value, day_df in df.groupby("date", sort=True):
        daily_row, daily_posterior = _score_blr_sample(
            model=model,
            sample_df=day_df,
            sample_id=_daily_group_key(date_value),
            feature_cols=feature_cols,
            target_col=target_col,
            theta_pre=theta_pre,
            sigma2=sigma2,
            sigma_pi2=sigma_pi2,
            device=device,
            batch_size=batch_size,
            jitter=jitter,
        )
        daily_rows.append(daily_row)
        daily_posteriors.append(daily_posterior)

    daily_scores = pd.DataFrame(daily_rows)
    expected_train_nll_daily_mean = float(
        daily_scores["expected_train_nll_total"].mean()
    )
    kl_daily_mean = float(daily_scores["kl"].mean())
    pac_score_1_daily_mean = float(daily_scores["pac_score_1"].mean())
    bound_objective_daily_mean = float(
        daily_scores["bound_objective_total"].mean()
    )

    row = {
        "name": spec["name"],
        "checkpoint_path": str(spec["checkpoint_path"]),
        "expected_train_nll_daily_mean": expected_train_nll_daily_mean,
        "kl_daily_mean": kl_daily_mean,
        "pac_score_1_daily_mean": pac_score_1_daily_mean,
        "bound_objective_daily_mean": bound_objective_daily_mean,
        "n_days": int(len(daily_scores)),
        "prior_center": prior_center,
        "status": "ok",
        "error_message": "",
    }
    posterior = {
        "theta_pre": theta_pre,
        "prior_center": prior_center,
        "daily_posteriors": daily_posteriors,
        "daily_scores": daily_scores.to_dict(orient="records"),
        "feature_cols": feature_cols,
        "name": spec["name"],
        "expected_train_nll_daily_mean": expected_train_nll_daily_mean,
        "kl_daily_mean": kl_daily_mean,
        "pac_score_1_daily_mean": pac_score_1_daily_mean,
        "bound_objective_daily_mean": bound_objective_daily_mean,
        "sigma2": float(sigma2),
        "sigma_pi2": float(sigma_pi2),
    }
    return row, posterior


def score_checkpoint_blr_subsamples(
    spec: dict,
    target_dataset: pd.DataFrame,
    target_col: str = "mid_price",
    date_before: Optional[str] = None,
    only_put: bool = True,
    sigma2: float = 0.5,
    sigma_pi2: float = 0.01,
    prior_center: str = "pretrained",
    device: Optional[str] = None,
    batch_size: int = 8192,
    jitter: float = 1e-8,
    subsample_size: Optional[int] = 10_000,
    subsample_frac: Optional[float] = None,
    n_subsamples: int = 5,
    subsample_seed: int = 123,
) -> tuple[dict, dict]:
    """
    PACTran-style top-layer PAC-Bayes regression score using repeated random
    subsamples from one target sample, usually one rolling month.
    """
    if sigma2 <= 0 or sigma_pi2 <= 0:
        raise ValueError("sigma2 and sigma_pi2 must be positive.")
    if n_subsamples <= 0:
        raise ValueError("n_subsamples must be positive.")
    if subsample_frac is not None and not (0.0 < subsample_frac <= 1.0):
        raise ValueError("subsample_frac must be in (0, 1].")
    if subsample_frac is None and (subsample_size is None or subsample_size <= 0):
        raise ValueError("subsample_size must be positive when subsample_frac is not set.")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    feature_cols = list(spec["feature_cols"])
    df = _prepare_target_data(
        target_dataset=target_dataset,
        feature_cols=feature_cols,
        target_col=target_col,
        date_before=date_before,
        only_put=only_put,
    )

    model = _make_model(spec, device=device)
    theta_pre = _make_prior_center_theta(
        model=model,
        dtype=torch.float64,
        prior_center=prior_center,
    )
    rng = np.random.default_rng(subsample_seed)
    subsample_rows = []
    subsample_posteriors = []
    actual_subsample_size = (
        max(int(np.floor(len(df) * float(subsample_frac))), 1)
        if subsample_frac is not None
        else int(subsample_size)
    )

    for subsample_idx in range(1, n_subsamples + 1):
        random_state = int(rng.integers(0, np.iinfo(np.int32).max))
        sample_df = df.sample(
            n=actual_subsample_size,
            replace=False,
            random_state=random_state,
        ).reset_index(drop=True)
        sample_row, sample_posterior = _score_blr_sample(
            model=model,
            sample_df=sample_df,
            sample_id=f"subsample_{subsample_idx}",
            feature_cols=feature_cols,
            target_col=target_col,
            theta_pre=theta_pre,
            sigma2=sigma2,
            sigma_pi2=sigma_pi2,
            device=device,
            batch_size=batch_size,
            jitter=jitter,
        )
        sample_row["subsample_id"] = subsample_idx
        sample_row["subsample_seed"] = random_state
        sample_posterior["subsample_id"] = subsample_idx
        sample_posterior["subsample_seed"] = random_state
        subsample_rows.append(sample_row)
        subsample_posteriors.append(sample_posterior)

    subsample_scores = pd.DataFrame(subsample_rows)
    expected_train_nll_subsample_mean = float(
        subsample_scores["expected_train_nll_total"].mean()
    )
    kl_subsample_mean = float(subsample_scores["kl"].mean())
    pac_score_1_subsample_mean = float(subsample_scores["pac_score_1"].mean())
    pac_score_1_subsample_std = float(
        subsample_scores["pac_score_1"].std(ddof=0)
    )
    bound_objective_subsample_mean = float(
        subsample_scores["bound_objective_total"].mean()
    )
    bound_objective_subsample_std = float(
        subsample_scores["bound_objective_total"].std(ddof=0)
    )
    component_stats = {}
    for col in [
        "posterior_mean_sse",
        "posterior_cov_sse",
        "expected_sse",
        "trace_A_inv",
        "theta_diff_sq",
        "logdet_A",
        "pac_score_1",
    ]:
        mean_value, std_value = _mean_std(subsample_scores[col])
        component_stats[f"{col}_subsample_mean"] = mean_value
        component_stats[f"{col}_subsample_std"] = std_value

    row = {
        "name": spec["name"],
        "checkpoint_path": str(spec["checkpoint_path"]),
        "expected_train_nll_subsample_mean": expected_train_nll_subsample_mean,
        "kl_subsample_mean": kl_subsample_mean,
        "pac_score_1_subsample_mean": pac_score_1_subsample_mean,
        "pac_score_1_subsample_std": pac_score_1_subsample_std,
        "bound_objective_subsample_mean": bound_objective_subsample_mean,
        "bound_objective_subsample_std": bound_objective_subsample_std,
        "n_subsamples": int(n_subsamples),
        "subsample_frac": np.nan if subsample_frac is None else float(subsample_frac),
        "subsample_size": int(actual_subsample_size),
        "n_month_samples": int(len(df)),
        "prior_center": prior_center,
        "status": "ok",
        "error_message": "",
    }
    row.update(component_stats)
    posterior = {
        "theta_pre": theta_pre,
        "prior_center": prior_center,
        "subsample_posteriors": subsample_posteriors,
        "subsample_scores": subsample_scores.to_dict(orient="records"),
        "feature_cols": feature_cols,
        "name": spec["name"],
        "expected_train_nll_subsample_mean": expected_train_nll_subsample_mean,
        "kl_subsample_mean": kl_subsample_mean,
        "pac_score_1_subsample_mean": pac_score_1_subsample_mean,
        "pac_score_1_subsample_std": pac_score_1_subsample_std,
        "bound_objective_subsample_mean": bound_objective_subsample_mean,
        "bound_objective_subsample_std": bound_objective_subsample_std,
        "subsample_frac": np.nan if subsample_frac is None else float(subsample_frac),
        "subsample_size": int(actual_subsample_size),
        "n_subsamples": int(n_subsamples),
        "n_month_samples": int(len(df)),
        "sigma2": float(sigma2),
        "sigma_pi2": float(sigma_pi2),
    }
    posterior.update(component_stats)
    return row, posterior


def run_pactran_blr_scores(
    target_dataset: pd.DataFrame,
    checkpoint_specs: Iterable[dict],
    output_csv: Path,
    target_col: str = "mid_price",
    date_before: Optional[str] = "2020-01-01",
    only_put: bool = True,
    sigma2: float = 0.5,
    sigma_pi2: float = 0.001,
    prior_center: str = "pretrained",
    device: Optional[str] = None,
    posterior_dir: Optional[Path] = None,
    subsample_size: int = 10_000,
    subsample_frac: Optional[float] = None,
    n_subsamples: int = 5,
    subsample_seed: int = 123,
    use_maturity_buckets: bool = False,
    maturity_bins: Optional[list[float]] = None,
    maturity_labels: Optional[list[str]] = None,
) -> pd.DataFrame:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if posterior_dir is not None:
        posterior_dir = Path(posterior_dir)
        posterior_dir.mkdir(parents=True, exist_ok=True)

    data = target_dataset.copy()
    data["date"] = pd.to_datetime(data["date"])
    if date_before is not None:
        data = data.loc[data["date"] < pd.Timestamp(date_before)].copy()
    if only_put and "cp_flag" in data.columns:
        data = data.loc[data["cp_flag"].astype(str).str.upper() == "P"].copy()

    maturity_labels = MATURITY_LABELS if maturity_labels is None else maturity_labels
    if use_maturity_buckets:
        data = add_maturity_bucket(
            data,
            maturity_bins=maturity_bins,
            maturity_labels=maturity_labels,
        )
        bucket_items = [
            (
                bucket,
                data.loc[data["maturity_bucket"].astype(str).eq(str(bucket))].copy(),
            )
            for bucket in maturity_labels
        ]
    else:
        bucket_items = [("all", data)]

    rows = []
    for bucket, bucket_data in bucket_items:
        for spec in checkpoint_specs:
            try:
                row, posterior = score_checkpoint_blr_subsamples(
                    spec=spec,
                    target_dataset=bucket_data,
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
                row["test_start_date"] = date_before
                row["maturity_bucket"] = str(bucket)
                row["theory"] = theory
                row["n_pac_bucket"] = row.pop("n_month_samples")
                if posterior_dir is not None:
                    posterior_path = (
                        posterior_dir
                        / f"{theory}_{_path_safe_label(bucket)}_posterior.pt"
                    )
                    torch.save(posterior, posterior_path)
                    row["posterior_path"] = str(posterior_path)
                else:
                    row["posterior_path"] = ""
            except Exception as exc:
                row = {
                    "test_start_date": date_before,
                    "maturity_bucket": str(bucket),
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
                    "n_pac_bucket": int(len(bucket_data)),
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
    columns = [
        "test_start_date",
        "maturity_bucket",
        "pac_rank",
        "theory",
        "checkpoint_path",
        "expected_train_nll_subsample_mean",
        "kl_subsample_mean",
        "posterior_mean_sse_subsample_mean",
        "posterior_mean_sse_subsample_std",
        "posterior_cov_sse_subsample_mean",
        "posterior_cov_sse_subsample_std",
        "expected_sse_subsample_mean",
        "expected_sse_subsample_std",
        "trace_A_inv_subsample_mean",
        "trace_A_inv_subsample_std",
        "theta_diff_sq_subsample_mean",
        "theta_diff_sq_subsample_std",
        "logdet_A_subsample_mean",
        "logdet_A_subsample_std",
        "pac_score_1_subsample_mean",
        "pac_score_1_subsample_std",
        "bound_objective_subsample_mean",
        "bound_objective_subsample_std",
        "n_subsamples",
        "subsample_frac",
        "subsample_size",
        "n_pac_bucket",
        "prior_center",
        "status",
        "error_message",
        "posterior_path",
    ]
    results = results[[col for col in columns if col in results.columns]]
    results.to_csv(output_csv, index=False)

    if ok.any():
        top = results.loc[results["status"].eq("ok")].iloc[0]
        print(
            f"[PACTran BLR] top={top['theory']} bucket={top['maturity_bucket']} "
            f"pac_score_1_subsample_mean={top['pac_score_1_subsample_mean']:.6f}"
        )
    return results


def run_rolling_pactran_blr_scores(
    target_dataset: pd.DataFrame,
    checkpoint_specs: Iterable[dict],
    output_csv: Path,
    target_col: str = "mid_price",
    train_days: int = 20,
    test_days: int = 20,
    step_days: int = 1,
    only_put: bool = True,
    sigma2: float = 0.5,
    sigma_pi2: float = 0.01,
    prior_center: str = "pretrained",
    device: Optional[str] = None,
    posterior_dir: Optional[Path] = None,
    subsample_size: int = 10_000,
    subsample_frac: Optional[float] = None,
    n_subsamples: int = 5,
    subsample_seed: int = 123,
    use_maturity_buckets: bool = True,
    maturity_bins: Optional[list[float]] = None,
    maturity_labels: Optional[list[str]] = None,
) -> pd.DataFrame:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if posterior_dir is not None:
        posterior_dir = Path(posterior_dir)
        posterior_dir.mkdir(parents=True, exist_ok=True)

    data = target_dataset.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["_date"] = data["date"].dt.normalize()
    windows = _make_rolling_date_windows(
        target_dataset=data,
        train_days=train_days,
        test_days=test_days,
        step_days=step_days,
    )
    maturity_labels = MATURITY_LABELS if maturity_labels is None else maturity_labels

    rows = []
    for window in windows:
        train_dates = set(pd.to_datetime(window["train_dates"]).normalize())
        window_data = data.loc[data["_date"].isin(train_dates)].drop(columns=["_date"]).copy()
        if use_maturity_buckets:
            window_data = add_maturity_bucket(
                window_data,
                maturity_bins=maturity_bins,
                maturity_labels=maturity_labels,
            )
            bucket_items = [
                (
                    bucket,
                    window_data.loc[
                        window_data["maturity_bucket"].astype(str).eq(str(bucket))
                    ].copy(),
                )
                for bucket in maturity_labels
            ]
        else:
            bucket_items = [("all", window_data)]

        for bucket, bucket_data in bucket_items:
            for spec in checkpoint_specs:
                try:
                    row, posterior = score_checkpoint_blr_subsamples(
                        spec=spec,
                        target_dataset=bucket_data,
                        target_col=target_col,
                        date_before=None,
                        only_put=only_put,
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
                    row["window_id"] = window["window_id"]
                    row["train_start_date"] = window["train_start_date"]
                    row["train_end_date"] = window["train_end_date"]
                    row["test_start_date"] = window["test_start_date"]
                    row["test_end_date"] = window["test_end_date"]
                    row["train_days"] = window["train_days"]
                    row["test_days"] = window["test_days"]
                    row["step_days"] = window["step_days"]
                    row["maturity_bucket"] = str(bucket)
                    row["theory"] = theory
                    row["n_pac_bucket"] = row.pop("n_month_samples")
                    if posterior_dir is not None:
                        posterior_path = (
                            posterior_dir
                            / f"{theory}_{window['window_id']}_{bucket}_posterior.pt"
                        )
                        torch.save(posterior, posterior_path)
                        row["posterior_path"] = str(posterior_path)
                    else:
                        row["posterior_path"] = ""
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
                        "maturity_bucket": str(bucket),
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
                        "n_pac_bucket": int(len(bucket_data)),
                        "prior_center": prior_center,
                        "status": "failed",
                        "error_message": str(exc),
                        "posterior_path": "",
                    }
                rows.append(row)
        pd.DataFrame(rows).to_csv(output_csv, index=False)

    results = pd.DataFrame(rows)
    ok = results["status"].eq("ok")
    results.loc[ok, "pac_rank"] = results.loc[ok].groupby(
        ["window_id", "maturity_bucket"]
    )[
        "pac_score_1_subsample_mean"
    ].rank(method="first", ascending=True)
    results.loc[~ok, "pac_rank"] = np.nan
    results = results.sort_values(
        ["window_id", "maturity_bucket", "pac_rank", "theory"],
        na_position="last",
    ).reset_index(drop=True)
    columns = [
        "window_id",
        "train_start_date",
        "train_end_date",
        "test_start_date",
        "test_end_date",
        "train_days",
        "test_days",
        "step_days",
        "maturity_bucket",
        "pac_rank",
        "theory",
        "checkpoint_path",
        "expected_train_nll_subsample_mean",
        "kl_subsample_mean",
        "posterior_mean_sse_subsample_mean",
        "posterior_mean_sse_subsample_std",
        "posterior_cov_sse_subsample_mean",
        "posterior_cov_sse_subsample_std",
        "expected_sse_subsample_mean",
        "expected_sse_subsample_std",
        "trace_A_inv_subsample_mean",
        "trace_A_inv_subsample_std",
        "theta_diff_sq_subsample_mean",
        "theta_diff_sq_subsample_std",
        "logdet_A_subsample_mean",
        "logdet_A_subsample_std",
        "pac_score_1_subsample_mean",
        "pac_score_1_subsample_std",
        "bound_objective_subsample_mean",
        "bound_objective_subsample_std",
        "n_subsamples",
        "subsample_frac",
        "subsample_size",
        "n_pac_bucket",
        "prior_center",
        "status",
        "error_message",
        "posterior_path",
    ]
    results = results[[col for col in columns if col in results.columns]]
    results.to_csv(output_csv, index=False)
    return results


def align_pactran_finetune(
    finetune_results: pd.DataFrame,
    pactran_results: pd.DataFrame,
    output_csv: Path,
) -> pd.DataFrame:
    finetune = finetune_results.copy()
    pactran = pactran_results.copy()
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    finetune["maturity_bucket"] = finetune["maturity_bucket"].astype(str)
    finetune["theory"] = finetune["theory"].astype(str).str.upper()
    pactran["maturity_bucket"] = pactran["maturity_bucket"].astype(str)
    pactran["theory"] = pactran["theory"].astype(str).str.upper()

    pactran_cols = [
        "maturity_bucket",
        "theory",
        "pactran_scope",
        "pac_rank",
        "checkpoint_path",
        "expected_train_nll_subsample_mean",
        "kl_subsample_mean",
        "posterior_mean_sse_subsample_mean",
        "posterior_mean_sse_subsample_std",
        "posterior_cov_sse_subsample_mean",
        "posterior_cov_sse_subsample_std",
        "expected_sse_subsample_mean",
        "expected_sse_subsample_std",
        "trace_A_inv_subsample_mean",
        "trace_A_inv_subsample_std",
        "theta_diff_sq_subsample_mean",
        "theta_diff_sq_subsample_std",
        "logdet_A_subsample_mean",
        "logdet_A_subsample_std",
        "pac_score_1_subsample_mean",
        "pac_score_1_subsample_std",
        "bound_objective_subsample_mean",
        "bound_objective_subsample_std",
        "n_subsamples",
        "subsample_frac",
        "subsample_size",
        "n_pac_bucket",
        "n_pac_universal_train",
        "status",
        "error_message",
    ]
    pactran = pactran[[col for col in pactran_cols if col in pactran.columns]].rename(
        columns={
            "checkpoint_path": "pactran_checkpoint_path",
            "status": "pactran_status",
            "error_message": "pactran_error_message",
        }
    )
    if "pac_score_1_subsample_mean" in pactran.columns:
        pactran["pac_score"] = pactran["pac_score_1_subsample_mean"]

    aligned = finetune.merge(
        pactran,
        on=["theory", "maturity_bucket"],
        how="left",
        validate="many_to_one",
    )
    columns = [
        "holdout_seed",
        "draw_seed",
        "retry_idx",
        "bucket_type",
        "train_scope",
        "test_start_date",
        "train_start_date",
        "train_end_date",
        "global_train_start_date",
        "global_train_end_date",
        "test_end_date",
        "maturity_bucket",
        "theory",
        "method",
        "seed",
        "pactran_scope",
        "pac_rank",
        "pac_score",
        "pac_score_1_subsample_mean",
        "pac_score_1_subsample_std",
        "expected_train_nll_subsample_mean",
        "kl_subsample_mean",
        "posterior_mean_sse_subsample_mean",
        "posterior_mean_sse_subsample_std",
        "posterior_cov_sse_subsample_mean",
        "posterior_cov_sse_subsample_std",
        "expected_sse_subsample_mean",
        "expected_sse_subsample_std",
        "trace_A_inv_subsample_mean",
        "trace_A_inv_subsample_std",
        "theta_diff_sq_subsample_mean",
        "theta_diff_sq_subsample_std",
        "logdet_A_subsample_mean",
        "logdet_A_subsample_std",
        "bound_objective_subsample_mean",
        "bound_objective_subsample_std",
        "n_subsamples",
        "subsample_frac",
        "subsample_size",
        "n_pac_bucket",
        "n_pac_universal_train",
        "test_weighted_mae",
        "test_mae",
        "test_mse",
        "test_bsiv_median",
        "frac_nan_pred_iv",
        "frac_nan_mkt_iv",
        "n_used_bsiv",
        "n_trainval",
        "n_trainval_total",
        "n_train_total",
        "n_val_total",
        "n_test_total",
        "n_train",
        "n_val",
        "n_test",
        "batch_size",
        "lr",
        "best_val",
        "prediction_path",
        "finetune_checkpoint_path",
        "pactran_checkpoint_path",
        "status",
        "error_message",
        "pactran_status",
        "pactran_error_message",
    ]
    aligned = aligned[[col for col in columns if col in aligned.columns]]
    aligned.to_csv(output_csv, index=False)
    return aligned


def align_rolling_pactran_finetune(
    finetune_results: pd.DataFrame,
    pactran_results: pd.DataFrame,
    output_csv: Path,
) -> pd.DataFrame:
    finetune = finetune_results.copy()
    pactran = pactran_results.copy()
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    finetune["window_id"] = finetune["window_id"].astype(str)
    finetune["maturity_bucket"] = finetune["maturity_bucket"].astype(str)
    finetune["theory"] = finetune["theory"].astype(str).str.upper()
    pactran["window_id"] = pactran["window_id"].astype(str)
    pactran["maturity_bucket"] = pactran["maturity_bucket"].astype(str)
    pactran["theory"] = pactran["theory"].astype(str).str.upper()

    pactran_cols = [
        "window_id",
        "maturity_bucket",
        "theory",
        "pac_rank",
        "checkpoint_path",
        "expected_train_nll_subsample_mean",
        "kl_subsample_mean",
        "pac_score_1_subsample_mean",
        "pac_score_1_subsample_std",
        "bound_objective_subsample_mean",
        "bound_objective_subsample_std",
        "n_subsamples",
        "subsample_frac",
        "subsample_size",
        "n_pac_bucket",
        "status",
        "error_message",
    ]
    pactran = pactran[[col for col in pactran_cols if col in pactran.columns]].rename(
        columns={
            "checkpoint_path": "pactran_checkpoint_path",
            "status": "pactran_status",
            "error_message": "pactran_error_message",
        }
    )
    if "pac_score_1_subsample_mean" in pactran.columns:
        pactran["pac_score"] = pactran["pac_score_1_subsample_mean"]

    aligned = finetune.merge(
        pactran,
        on=["window_id", "theory", "maturity_bucket"],
        how="left",
        validate="many_to_one",
    )
    columns = [
        "window_id",
        "train_start_date",
        "train_end_date",
        "test_start_date",
        "test_end_date",
        "train_days",
        "test_days",
        "step_days",
        "maturity_bucket",
        "theory",
        "method",
        "seed",
        "pac_rank",
        "pac_score",
        "pac_score_1_subsample_mean",
        "pac_score_1_subsample_std",
        "expected_train_nll_subsample_mean",
        "kl_subsample_mean",
        "bound_objective_subsample_mean",
        "bound_objective_subsample_std",
        "n_subsamples",
        "subsample_frac",
        "subsample_size",
        "n_pac_bucket",
        "test_weighted_mae",
        "test_mae",
        "test_mse",
        "test_bsiv_median",
        "frac_nan_pred_iv",
        "frac_nan_mkt_iv",
        "n_used_bsiv",
        "n_trainval",
        "n_train",
        "n_val",
        "n_test",
        "best_val",
        "prediction_path",
        "finetune_checkpoint_path",
        "pactran_checkpoint_path",
        "status",
        "error_message",
        "pactran_status",
        "pactran_error_message",
    ]
    aligned = aligned[[col for col in columns if col in aligned.columns]]
    aligned.to_csv(output_csv, index=False)
    return aligned
