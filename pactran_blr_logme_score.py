"""
LogME-style empirical-Bayes scale optimization for PACTran top-layer BLR.

This module keeps the existing PACTran top-layer design:
    frozen ResMLP features Phi + Gaussian BLR final layer.

Difference from pactran_blr_score.py:
    pactran_blr_score.py takes sigma2 and sigma_pi2 as fixed hyperparameters.
    This module optimizes them by type-II maximum likelihood / evidence,
    using the fixed-point updates used by LogME.

For a nonzero prior center theta_pre, write w = theta_pre + u and optimize
alpha, beta for
    y - Phi theta_pre = Phi u + eps,
    u ~ N(0, alpha^{-1} I), eps ~ N(0, beta^{-1} I).
Then sigma_pi2 = 1 / alpha and sigma2 = 1 / beta.

Important ranking convention:
    When sigma2 and sigma_pi2 differ across checkpoints, the Gaussian constants
    are no longer common. Therefore the compatible selection score is the full
    negative log evidence / optimized PAC-Bayes objective:
        n E_Q[train NLL] + KL(Q || P) = -log p(y | Phi, theta_pre).
    For backward compatibility with the rest of the project, this value is also
    stored in pac_score_1. Lower is better.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch

from model import ResMLP
from pactran_blr_score import (
    MATURITY_LABELS,
    add_maturity_bucket,
    _feature_matrix,
    _make_model,
    _make_prior_center_theta,
    _make_rolling_date_windows,
    _mean_std,
    _path_safe_label,
    _prepare_target_data,
    _solve_with_jitter,
    _slogdet_with_jitter,
    _standardize_hidden_features,
)


def _positive_or_default(value: Optional[float], default: float, name: str) -> float:
    out = default if value is None else float(value)
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}.")
    return out


def _clamp_float(value: float, lo: float, hi: float) -> float:
    return float(min(max(float(value), float(lo)), float(hi)))


def _logme_fixed_point_optimize(
    Phi: torch.Tensor,
    y: torch.Tensor,
    theta_pre: torch.Tensor,
    init_sigma2: float,
    init_sigma_pi2: float,
    max_iter: int = 100,
    tol: float = 1e-6,
    min_sigma2: float = 1e-10,
    max_sigma2: float = 1e4,
    min_sigma_pi2: float = 1e-10,
    max_sigma_pi2: float = 1e4,
    eps: float = 1e-12,
) -> dict:
    """Optimize alpha=1/sigma_pi2 and beta=1/sigma2 by LogME fixed point."""
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if tol <= 0:
        raise ValueError("tol must be positive.")

    n_samples, top_layer_dim = Phi.shape
    if theta_pre.shape != (top_layer_dim, 1):
        raise ValueError(
            f"theta_pre shape must be {(top_layer_dim, 1)}, got {tuple(theta_pre.shape)}."
        )

    y_shifted = y - Phi @ theta_pre
    G = Phi.T @ Phi
    eigvals, eigvecs = torch.linalg.eigh(G)
    eigvals = torch.clamp(eigvals, min=0.0)
    z = Phi.T @ y_shifted
    eig_z = eigvecs.T @ z

    alpha_min = 1.0 / float(max_sigma_pi2)
    alpha_max = 1.0 / float(min_sigma_pi2)
    beta_min = 1.0 / float(max_sigma2)
    beta_max = 1.0 / float(min_sigma2)

    alpha = _clamp_float(1.0 / float(init_sigma_pi2), alpha_min, alpha_max)
    beta = _clamp_float(1.0 / float(init_sigma2), beta_min, beta_max)

    converged = False
    gamma_value = np.nan
    rss_value = np.nan
    u_norm_sq_value = np.nan
    n_iter = 0

    for n_iter in range(1, int(max_iter) + 1):
        denom = alpha + beta * eigvals
        denom = torch.clamp(denom, min=eps)

        u_bar = beta * (eigvecs @ (eig_z / denom.reshape(-1, 1)))
        residual = y_shifted - Phi @ u_bar
        rss = float((residual.T @ residual).squeeze().item())
        u_norm_sq = float((u_bar.T @ u_bar).squeeze().item())
        gamma = float(torch.sum(beta * eigvals / denom).item())

        if gamma <= eps or u_norm_sq <= eps:
            alpha_new = alpha_max
        else:
            alpha_new = gamma / u_norm_sq

        residual_df = max(float(n_samples) - gamma, eps)
        if rss <= eps:
            beta_new = beta_max
        else:
            beta_new = residual_df / rss

        alpha_new = _clamp_float(alpha_new, alpha_min, alpha_max)
        beta_new = _clamp_float(beta_new, beta_min, beta_max)

        rel_change = max(
            abs(np.log(alpha_new / alpha)),
            abs(np.log(beta_new / beta)),
        )
        alpha, beta = alpha_new, beta_new
        gamma_value = gamma
        rss_value = rss
        u_norm_sq_value = u_norm_sq

        if rel_change < tol:
            converged = True
            break

    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "sigma_pi2": float(1.0 / alpha),
        "sigma2": float(1.0 / beta),
        "gamma": float(gamma_value),
        "effective_dim": float(gamma_value),
        "residual_df": float(max(float(n_samples) - gamma_value, eps)),
        "rss_shifted": float(rss_value),
        "u_norm_sq": float(u_norm_sq_value),
        "n_iter": int(n_iter),
        "converged": bool(converged),
        "init_sigma2": float(init_sigma2),
        "init_sigma_pi2": float(init_sigma_pi2),
    }


def _score_precomputed_blr(
    Phi: torch.Tensor,
    y: torch.Tensor,
    theta_pre: torch.Tensor,
    sigma2: float,
    sigma_pi2: float,
    jitter: float,
) -> tuple[dict, dict]:
    """Score a standardized Phi/y/theta_pre with fixed scales."""
    n_samples, top_layer_dim = Phi.shape
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

    old_constant_part = (
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

    pac_score_without_constants = (
        0.5 * float(expected_sse.item()) / float(sigma2)
        + 0.5
        * (
            (trace_A_inv + theta_diff_sq) / float(sigma_pi2)
            + logdet_A_value
        )
    )

    row = {
        "n_samples": int(n_samples),
        "posterior_mean_sse": float(posterior_mean_sse.item()),
        "posterior_cov_sse": float(posterior_cov_sse.item()),
        "expected_sse": float(expected_sse.item()),
        "trace_A_inv": trace_A_inv,
        "theta_diff_sq": theta_diff_sq,
        "logdet_A": logdet_A_value,
        "constant_part": float(old_constant_part),
        "expected_train_nll_total": float(expected_train_nll_total),
        "kl": float(kl),
        "bound_objective_total": float(bound_objective_total),
        "neg_log_evidence_total": float(bound_objective_total),
        "log_evidence_total": float(-bound_objective_total),
        "neg_log_evidence_per_sample": float(bound_objective_total / max(n_samples, 1)),
        "logme_per_sample": float(-bound_objective_total / max(n_samples, 1)),
        "pac_score_without_constants": float(pac_score_without_constants),
        # Backward-compatible selection column. With optimized scales, constants
        # differ across checkpoints, so we use the full -log evidence here.
        "pac_score_1": float(bound_objective_total),
    }
    posterior = {
        "theta_bar": theta_bar,
        "A": A,
        "A_inv": A_inv,
        **row,
    }
    return row, posterior


def _score_blr_sample_logme(
    model: ResMLP,
    sample_df: pd.DataFrame,
    sample_id: str,
    feature_cols: list[str],
    target_col: str,
    theta_pre: torch.Tensor,
    init_sigma2: float,
    init_sigma_pi2: float,
    device: str,
    batch_size: int,
    jitter: float,
    optimization_max_iter: int = 100,
    optimization_tol: float = 1e-6,
    min_sigma2: float = 1e-10,
    max_sigma2: float = 1e4,
    min_sigma_pi2: float = 1e-10,
    max_sigma_pi2: float = 1e4,
) -> tuple[dict, dict]:
    Phi_raw = _feature_matrix(
        model=model,
        X_np=sample_df[feature_cols].to_numpy(np.float32),
        device=device,
        batch_size=batch_size,
    )
    y = torch.tensor(
        sample_df[target_col].to_numpy(np.float64).reshape(-1, 1),
        dtype=torch.float64,
    )

    n_samples, top_layer_dim = Phi_raw.shape
    if theta_pre.shape[0] != top_layer_dim:
        raise ValueError(
            f"Top-layer dim mismatch: theta_pre has {theta_pre.shape[0]}, "
            f"Phi has {top_layer_dim} columns."
        )

    Phi, theta_pre_scaled, phi_standardization = _standardize_hidden_features(
        Phi_raw,
        theta_pre,
    )
    opt = _logme_fixed_point_optimize(
        Phi=Phi,
        y=y,
        theta_pre=theta_pre_scaled,
        init_sigma2=init_sigma2,
        init_sigma_pi2=init_sigma_pi2,
        max_iter=optimization_max_iter,
        tol=optimization_tol,
        min_sigma2=min_sigma2,
        max_sigma2=max_sigma2,
        min_sigma_pi2=min_sigma_pi2,
        max_sigma_pi2=max_sigma_pi2,
    )
    row, posterior = _score_precomputed_blr(
        Phi=Phi,
        y=y,
        theta_pre=theta_pre_scaled,
        sigma2=opt["sigma2"],
        sigma_pi2=opt["sigma_pi2"],
        jitter=jitter,
    )
    row.update(
        {
            "sample_id": sample_id,
            "scale_optimization": "logme_fixed_point",
            "alpha_optimized": opt["alpha"],
            "beta_optimized": opt["beta"],
            "sigma2_optimized": opt["sigma2"],
            "sigma_pi2_optimized": opt["sigma_pi2"],
            "gamma_optimized": opt["gamma"],
            "effective_dim": opt["effective_dim"],
            "residual_df": opt["residual_df"],
            "scale_opt_n_iter": opt["n_iter"],
            "scale_opt_converged": opt["converged"],
            "init_sigma2": opt["init_sigma2"],
            "init_sigma_pi2": opt["init_sigma_pi2"],
        }
    )
    posterior.update(
        {
            "sample_id": sample_id,
            "theta_pre": theta_pre_scaled,
            "phi_standardization": phi_standardization,
            "scale_optimization": "logme_fixed_point",
            "scale_optimization_result": opt,
        }
    )
    return row, posterior


def _summarize_subsample_scores(subsample_scores: pd.DataFrame) -> dict:
    out = {
        "expected_train_nll_subsample_mean": float(
            subsample_scores["expected_train_nll_total"].mean()
        ),
        "kl_subsample_mean": float(subsample_scores["kl"].mean()),
        "pac_score_1_subsample_mean": float(subsample_scores["pac_score_1"].mean()),
        "pac_score_1_subsample_std": float(subsample_scores["pac_score_1"].std(ddof=0)),
        "bound_objective_subsample_mean": float(
            subsample_scores["bound_objective_total"].mean()
        ),
        "bound_objective_subsample_std": float(
            subsample_scores["bound_objective_total"].std(ddof=0)
        ),
    }
    for col in [
        "posterior_mean_sse",
        "posterior_cov_sse",
        "expected_sse",
        "trace_A_inv",
        "theta_diff_sq",
        "logdet_A",
        "pac_score_1",
        "pac_score_without_constants",
        "neg_log_evidence_total",
        "neg_log_evidence_per_sample",
        "logme_per_sample",
        "sigma2_optimized",
        "sigma_pi2_optimized",
        "alpha_optimized",
        "beta_optimized",
        "gamma_optimized",
        "effective_dim",
        "scale_opt_n_iter",
    ]:
        if col in subsample_scores.columns:
            mean_value, std_value = _mean_std(subsample_scores[col])
            out[f"{col}_subsample_mean"] = mean_value
            out[f"{col}_subsample_std"] = std_value
    if "scale_opt_converged" in subsample_scores.columns:
        out["scale_opt_converged_frac"] = float(subsample_scores["scale_opt_converged"].mean())
    return out


def score_checkpoint_blr_subsamples_logme(
    spec: dict,
    target_dataset: pd.DataFrame,
    target_col: str = "mid_price",
    date_before: Optional[str] = None,
    only_put: bool = True,
    sigma2: Optional[float] = None,
    sigma_pi2: Optional[float] = None,
    init_sigma2: Optional[float] = None,
    init_sigma_pi2: Optional[float] = None,
    prior_center: str = "pretrained",
    device: Optional[str] = None,
    batch_size: int = 8192,
    jitter: float = 1e-8,
    subsample_size: Optional[int] = 10_000,
    subsample_frac: Optional[float] = None,
    n_subsamples: int = 5,
    subsample_seed: int = 123,
    optimization_max_iter: int = 100,
    optimization_tol: float = 1e-6,
    min_sigma2: float = 1e-10,
    max_sigma2: float = 1e4,
    min_sigma_pi2: float = 1e-10,
    max_sigma_pi2: float = 1e4,
) -> tuple[dict, dict]:
    """PACTran-style score with LogME empirical-Bayes scale optimization."""
    init_sigma2 = _positive_or_default(init_sigma2, sigma2 if sigma2 is not None else 1.0, "init_sigma2")
    init_sigma_pi2 = _positive_or_default(
        init_sigma_pi2,
        sigma_pi2 if sigma_pi2 is not None else 1.0,
        "init_sigma_pi2",
    )
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
    if actual_subsample_size > len(df):
        raise ValueError(
            f"subsample_size={actual_subsample_size} exceeds available rows={len(df)}. "
            "Use subsample_frac=1 or a smaller subsample_size."
        )

    for subsample_idx in range(1, n_subsamples + 1):
        random_state = int(rng.integers(0, np.iinfo(np.int32).max))
        sample_df = df.sample(
            n=actual_subsample_size,
            replace=False,
            random_state=random_state,
        ).reset_index(drop=True)
        sample_row, sample_posterior = _score_blr_sample_logme(
            model=model,
            sample_df=sample_df,
            sample_id=f"subsample_{subsample_idx}",
            feature_cols=feature_cols,
            target_col=target_col,
            theta_pre=theta_pre,
            init_sigma2=init_sigma2,
            init_sigma_pi2=init_sigma_pi2,
            device=device,
            batch_size=batch_size,
            jitter=jitter,
            optimization_max_iter=optimization_max_iter,
            optimization_tol=optimization_tol,
            min_sigma2=min_sigma2,
            max_sigma2=max_sigma2,
            min_sigma_pi2=min_sigma_pi2,
            max_sigma_pi2=max_sigma_pi2,
        )
        sample_row["subsample_id"] = subsample_idx
        sample_row["subsample_seed"] = random_state
        sample_posterior["subsample_id"] = subsample_idx
        sample_posterior["subsample_seed"] = random_state
        subsample_rows.append(sample_row)
        subsample_posteriors.append(sample_posterior)

    subsample_scores = pd.DataFrame(subsample_rows)
    component_stats = _summarize_subsample_scores(subsample_scores)
    row = {
        "name": spec["name"],
        "checkpoint_path": str(spec["checkpoint_path"]),
        "n_subsamples": int(n_subsamples),
        "subsample_frac": np.nan if subsample_frac is None else float(subsample_frac),
        "subsample_size": int(actual_subsample_size),
        "n_month_samples": int(len(df)),
        "prior_center": prior_center,
        "scale_optimization": "logme_fixed_point",
        "init_sigma2": float(init_sigma2),
        "init_sigma_pi2": float(init_sigma_pi2),
        "status": "ok",
        "error_message": "",
    }
    row.update(component_stats)
    posterior = {
        "theta_pre": theta_pre,
        "prior_center": prior_center,
        "scale_optimization": "logme_fixed_point",
        "subsample_posteriors": subsample_posteriors,
        "subsample_scores": subsample_scores.to_dict(orient="records"),
        "feature_cols": feature_cols,
        "name": spec["name"],
        "subsample_frac": np.nan if subsample_frac is None else float(subsample_frac),
        "subsample_size": int(actual_subsample_size),
        "n_subsamples": int(n_subsamples),
        "n_month_samples": int(len(df)),
        "init_sigma2": float(init_sigma2),
        "init_sigma_pi2": float(init_sigma_pi2),
    }
    posterior.update(component_stats)
    return row, posterior


def run_pactran_blr_logme_scores(
    target_dataset: pd.DataFrame,
    checkpoint_specs: Iterable[dict],
    output_csv: Path,
    target_col: str = "mid_price",
    date_before: Optional[str] = "2020-01-01",
    only_put: bool = True,
    sigma2: float = 1.0,
    sigma_pi2: float = 1.0,
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
    optimization_max_iter: int = 100,
    optimization_tol: float = 1e-6,
    min_sigma2: float = 1e-10,
    max_sigma2: float = 1e4,
    min_sigma_pi2: float = 1e-10,
    max_sigma_pi2: float = 1e4,
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
            (bucket, data.loc[data["maturity_bucket"].astype(str).eq(str(bucket))].copy())
            for bucket in maturity_labels
        ]
    else:
        bucket_items = [("all", data)]

    rows = []
    for bucket, bucket_data in bucket_items:
        for spec in checkpoint_specs:
            try:
                row, posterior = score_checkpoint_blr_subsamples_logme(
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
                    optimization_max_iter=optimization_max_iter,
                    optimization_tol=optimization_tol,
                    min_sigma2=min_sigma2,
                    max_sigma2=max_sigma2,
                    min_sigma_pi2=min_sigma_pi2,
                    max_sigma_pi2=max_sigma_pi2,
                )
                theory = row.pop("name")
                row["test_start_date"] = date_before
                row["maturity_bucket"] = str(bucket)
                row["theory"] = theory
                row["n_pac_bucket"] = row.pop("n_month_samples")
                if posterior_dir is not None:
                    posterior_path = posterior_dir / f"{theory}_{_path_safe_label(bucket)}_posterior.pt"
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
                    "scale_optimization": "logme_fixed_point",
                    "status": "failed",
                    "error_message": str(exc),
                    "posterior_path": "",
                }
            rows.append(row)

    results = pd.DataFrame(rows)
    ok = results["status"].eq("ok")
    results.loc[ok, "pac_rank"] = results.loc[ok].groupby(["maturity_bucket"])[
        "pac_score_1_subsample_mean"
    ].rank(method="first", ascending=True)
    results.loc[~ok, "pac_rank"] = np.nan
    results = results.sort_values(
        ["maturity_bucket", "pac_rank", "theory"],
        na_position="last",
    ).reset_index(drop=True)
    results.to_csv(output_csv, index=False)
    if ok.any():
        top = results.loc[results["status"].eq("ok")].iloc[0]
        print(
            f"[PACTran BLR LogME] top={top['theory']} bucket={top['maturity_bucket']} "
            f"neg_log_evidence={top['pac_score_1_subsample_mean']:.6f}"
        )
    return results


def run_rolling_pactran_blr_logme_scores(
    target_dataset: pd.DataFrame,
    checkpoint_specs: Iterable[dict],
    output_csv: Path,
    target_col: str = "mid_price",
    train_days: int = 20,
    test_days: int = 20,
    step_days: int = 1,
    only_put: bool = True,
    sigma2: float = 1.0,
    sigma_pi2: float = 1.0,
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
    optimization_max_iter: int = 100,
    optimization_tol: float = 1e-6,
    min_sigma2: float = 1e-10,
    max_sigma2: float = 1e4,
    min_sigma_pi2: float = 1e-10,
    max_sigma_pi2: float = 1e4,
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
                    row, posterior = score_checkpoint_blr_subsamples_logme(
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
                        optimization_max_iter=optimization_max_iter,
                        optimization_tol=optimization_tol,
                        min_sigma2=min_sigma2,
                        max_sigma2=max_sigma2,
                        min_sigma_pi2=min_sigma_pi2,
                        max_sigma_pi2=max_sigma_pi2,
                    )
                    theory = row.pop("name")
                    row.update(
                        {
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
                            "n_pac_bucket": row.pop("n_month_samples"),
                        }
                    )
                    if posterior_dir is not None:
                        posterior_path = posterior_dir / f"{theory}_{window['window_id']}_{_path_safe_label(bucket)}_posterior.pt"
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
                        "scale_optimization": "logme_fixed_point",
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
    )["pac_score_1_subsample_mean"].rank(method="first", ascending=True)
    results.loc[~ok, "pac_rank"] = np.nan
    results = results.sort_values(
        ["window_id", "maturity_bucket", "pac_rank", "theory"],
        na_position="last",
    ).reset_index(drop=True)
    results.to_csv(output_csv, index=False)
    return results


# Short aliases used by patched repeated_random_holdout.py or ad-hoc scripts.
score_checkpoint_blr_subsamples_empirical_bayes = score_checkpoint_blr_subsamples_logme
run_pactran_blr_empirical_bayes_scores = run_pactran_blr_logme_scores
run_rolling_pactran_blr_empirical_bayes_scores = run_rolling_pactran_blr_logme_scores
