"""
Diagnose BLR scale choices for PACTran top-layer scoring.

This script freezes each pretrained network's feature extractor, builds the
same Phi matrix used by pactran_blr_score.py, fits a full-sample OLS top layer,
and reports rough scales for sigma2 and sigma_pi2.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from main import (
    EXPERIMENT_DIR,
    HESTON_DATASET_PATH,
    PACTRAN_TARGET_COL,
    checkpoint_specs,
)
from pactran_blr_score import (
    _feature_matrix,
    _make_model,
    _prepare_target_data,
    _standardize_hidden_features,
    _top_layer_theta,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_CSV = EXPERIMENT_DIR / "diagnostics" / "blr_scale_diagnostics.csv"


def _resolve(path: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _quantiles(prefix: str, values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float).reshape(-1)
    qs = {
        "q01": 0.01,
        "q05": 0.05,
        "q25": 0.25,
        "q50": 0.50,
        "q75": 0.75,
        "q95": 0.95,
        "q99": 0.99,
    }
    return {
        f"{prefix}_{name}": float(np.quantile(values, q))
        for name, q in qs.items()
    }


def _theta_stats(prefix: str, theta: torch.Tensor) -> dict:
    values = theta.detach().cpu().numpy().reshape(-1)
    abs_values = np.abs(values)
    out = {
        f"{prefix}_mean_abs": float(np.mean(abs_values)),
        f"{prefix}_std": float(np.std(values, ddof=0)),
        f"{prefix}_rms": float(np.sqrt(np.mean(values**2))),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
    }
    out.update(_quantiles(f"{prefix}_abs", abs_values))
    return out


def _vector_summary(prefix: str, values: torch.Tensor) -> dict:
    values_np = values.detach().cpu().numpy().reshape(-1)
    out = {
        f"{prefix}_mean": float(np.mean(values_np)),
        f"{prefix}_std": float(np.std(values_np, ddof=0)),
        f"{prefix}_min": float(np.min(values_np)),
        f"{prefix}_max": float(np.max(values_np)),
    }
    out.update(_quantiles(prefix, values_np))
    return out


def _per_column_rows(
    theory: str,
    hidden_mu: torch.Tensor,
    hidden_std_raw: torch.Tensor,
    hidden_std_used: torch.Tensor,
    hidden_scaled_std: torch.Tensor,
    theta_pre_raw: torch.Tensor,
    theta_pre_scaled: torch.Tensor,
) -> list[dict]:
    w_raw = theta_pre_raw[:-1].detach().cpu().numpy().reshape(-1)
    w_scaled = theta_pre_scaled[:-1].detach().cpu().numpy().reshape(-1)
    mu = hidden_mu.detach().cpu().numpy().reshape(-1)
    std_raw = hidden_std_raw.detach().cpu().numpy().reshape(-1)
    std_used = hidden_std_used.detach().cpu().numpy().reshape(-1)
    std_scaled = hidden_scaled_std.detach().cpu().numpy().reshape(-1)
    return [
        {
            "theory": theory,
            "hidden_column": int(i),
            "hidden_mu_raw": float(mu[i]),
            "hidden_std_raw": float(std_raw[i]),
            "hidden_std_used_for_scaling": float(std_used[i]),
            "hidden_mu_after_scaling": 0.0,
            "hidden_std_after_scaling": float(std_scaled[i]),
            "w_pre_raw": float(w_raw[i]),
            "w_pre_scaled": float(w_scaled[i]),
            "w_pre_scaled_minus_raw": float(w_scaled[i] - w_raw[i]),
            "w_pre_scale_ratio": (
                float(w_scaled[i] / w_raw[i])
                if abs(w_raw[i]) > 1e-12
                else np.nan
            ),
        }
        for i in range(len(mu))
    ]


def _fit_ols(Phi: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, str]:
    try:
        return torch.linalg.lstsq(Phi, y).solution, "torch.linalg.lstsq"
    except RuntimeError:
        G = Phi.T @ Phi
        rhs = Phi.T @ y
        eye = torch.eye(G.shape[0], dtype=G.dtype, device=G.device)
        return torch.linalg.solve(G + 1e-8 * eye, rhs), "normal_eq_jitter"


def diagnose_one(spec: dict, target_dataset: pd.DataFrame) -> tuple[dict, list[dict]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved_spec = dict(spec)
    resolved_spec["checkpoint_path"] = _resolve(resolved_spec["checkpoint_path"])

    model = _make_model(resolved_spec, device=device)
    theta_pre_raw = _top_layer_theta(model, dtype=torch.float64).reshape(-1, 1).cpu()
    feature_cols = list(resolved_spec["feature_cols"])

    df = _prepare_target_data(
        target_dataset=target_dataset,
        feature_cols=feature_cols,
        target_col=PACTRAN_TARGET_COL,
        date_before=None,
        only_put=True,
    )
    Phi_raw = _feature_matrix(
        model=model,
        X_np=df[feature_cols].to_numpy(np.float32),
        device=device,
        batch_size=8192,
    )
    hidden_raw = Phi_raw[:, :-1]
    hidden_mu_raw = hidden_raw.mean(dim=0)
    hidden_std_raw = hidden_raw.std(dim=0, unbiased=False)
    Phi, theta_pre, phi_standardization = _standardize_hidden_features(
        Phi_raw,
        theta_pre_raw,
    )
    hidden_scaled = Phi[:, :-1]
    hidden_scaled_mu = hidden_scaled.mean(dim=0)
    hidden_scaled_std = hidden_scaled.std(dim=0, unbiased=False)
    y = torch.tensor(
        df[PACTRAN_TARGET_COL].to_numpy(np.float64).reshape(-1, 1),
        dtype=torch.float64,
    )

    theta_hat, ols_method = _fit_ols(Phi, y)
    residual = y - Phi @ theta_hat
    n_samples, top_layer_dim = Phi.shape
    sse = float((residual.T @ residual).squeeze().item())
    sigma2_hat = sse / float(n_samples)
    sigma2_unbiased = (
        sse / float(n_samples - top_layer_dim)
        if n_samples > top_layer_dim
        else np.nan
    )

    theta_diff_pretrained = theta_hat - theta_pre
    sigma_pi2_zero = float(torch.mean(theta_hat**2).item())
    sigma_pi2_pretrained = float(torch.mean(theta_diff_pretrained**2).item())
    G = Phi.T @ Phi

    row = {
        "theory": resolved_spec["name"],
        "checkpoint_path": str(resolved_spec["checkpoint_path"]),
        "n_samples": int(n_samples),
        "top_layer_dim": int(top_layer_dim),
        "feature_cols": ",".join(feature_cols),
        "target_col": PACTRAN_TARGET_COL,
        "phi_scaling": "column_standardized_hidden_intercept_unchanged",
        "ols_method": ols_method,
        "ols_sse": sse,
        "ols_rmse": float(np.sqrt(sigma2_hat)),
        "sigma2_hat": float(sigma2_hat),
        "sigma2_unbiased": float(sigma2_unbiased),
        "sigma_hat": float(np.sqrt(sigma2_hat)),
        "sigma_unbiased": (
            float(np.sqrt(sigma2_unbiased))
            if np.isfinite(sigma2_unbiased)
            else np.nan
        ),
        "sigma_pi2_zero_hat": sigma_pi2_zero,
        "sigma_pi_zero_hat": float(np.sqrt(sigma_pi2_zero)),
        "sigma_pi2_pretrained_hat": sigma_pi2_pretrained,
        "sigma_pi_pretrained_hat": float(np.sqrt(sigma_pi2_pretrained)),
        "gram_condition_number": float(torch.linalg.cond(G).item()),
        "b_pre_raw": float(theta_pre_raw[-1].item()),
        "b_pre_scaled": float(theta_pre[-1].item()),
        "b_pre_scaled_minus_raw": float((theta_pre[-1] - theta_pre_raw[-1]).item()),
    }
    row.update(_vector_summary("hidden_mu_raw", hidden_mu_raw))
    row.update(_vector_summary("hidden_std_raw", hidden_std_raw))
    row.update(_vector_summary("hidden_std_used_for_scaling", phi_standardization["hidden_std"]))
    row.update(_vector_summary("hidden_mu_after_scaling", hidden_scaled_mu))
    row.update(_vector_summary("hidden_std_after_scaling", hidden_scaled_std))
    row.update(_theta_stats("theta_pre_raw", theta_pre_raw))
    row.update(_theta_stats("theta_pre_scaled", theta_pre))
    row.update(_theta_stats("theta_pre_scaled_minus_raw", theta_pre - theta_pre_raw))
    row.update(_theta_stats("theta_hat", theta_hat))
    row.update(_theta_stats("theta_hat_minus_theta_pre", theta_diff_pretrained))
    column_rows = _per_column_rows(
        theory=resolved_spec["name"],
        hidden_mu=hidden_mu_raw,
        hidden_std_raw=hidden_std_raw,
        hidden_std_used=phi_standardization["hidden_std"],
        hidden_scaled_std=hidden_scaled_std,
        theta_pre_raw=theta_pre_raw,
        theta_pre_scaled=theta_pre,
    )
    return row, column_rows


def main() -> None:
    dataset_path = _resolve(HESTON_DATASET_PATH)
    output_csv = _resolve(OUTPUT_CSV)
    target_dataset = pd.read_parquet(dataset_path)

    rows = []
    column_rows = []
    for spec in checkpoint_specs():
        row, spec_column_rows = diagnose_one(spec=spec, target_dataset=target_dataset)
        rows.append(row)
        column_rows.extend(spec_column_rows)
    results = pd.DataFrame(rows)
    column_results = pd.DataFrame(column_rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_csv, index=False)
    column_output_csv = output_csv.with_name("blr_scale_feature_diagnostics.csv")
    column_results.to_csv(column_output_csv, index=False)

    print(results.to_string(index=False))
    print("\nFeature scaling diagnostics:")
    print(column_results.to_string(index=False))
    print(f"\nSaved diagnostics to {output_csv}")
    print(f"Saved feature diagnostics to {column_output_csv}")


if __name__ == "__main__":
    main()
