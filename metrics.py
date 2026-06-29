import numpy as np
import pandas as pd


def _se(x: pd.Series) -> float:
    return x.std(ddof=1) / np.sqrt(len(x))


def _safe_corr(df: pd.DataFrame, x_col: str, y_col: str, method: str) -> float:
    """Correlation helper that returns NaN when the correlation is undefined."""
    cols = [x_col, y_col]
    if any(c not in df.columns for c in cols):
        return np.nan

    g = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(g) < 2:
        return np.nan
    if g[x_col].nunique() < 2 or g[y_col].nunique() < 2:
        return np.nan
    return float(g[x_col].corr(g[y_col], method=method))


def _safe_slope(df: pd.DataFrame, x_col: str, y_col: str) -> float:
    cols = [x_col, y_col]
    if any(c not in df.columns for c in cols):
        return np.nan

    g = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(g) < 2 or g[x_col].nunique() < 2:
        return np.nan
    return float(np.polyfit(g[x_col].to_numpy(), g[y_col].to_numpy(), deg=1)[0])


def _first_last(df: pd.DataFrame, value_col: str, x_col: str = "a_j"):
    if value_col not in df.columns:
        return np.nan, np.nan
    g = df.sort_values(x_col)[[x_col, value_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if g.empty:
        return np.nan, np.nan
    return float(g[value_col].iloc[0]), float(g[value_col].iloc[-1])


def summarize_multi_seed_results(all_results: pd.DataFrame):
    """Aggregate seed-level long-format results over seeds for each theory/method/a_j."""
    agg_spec = {
        "n_seeds": ("seed", "nunique"),
        "bound_mean": ("bound", "mean"),
        "bound_std": ("bound", "std"),
        "bound_se": ("bound", _se),
        "emp_loss_mean": ("emp_loss", "mean"),
        "emp_loss_std": ("emp_loss", "std"),
        "kl_mean": ("kl", "mean"),
        "kl_std": ("kl", "std"),
        "psi_hat_mean": ("psi_hat", "mean"),
        "psi_hat_std": ("psi_hat", "std"),
        "lambda_star_mean": ("lambda_star", "mean"),
        "input_dim": ("input_dim", "first"),
    }

    optional_cols = [
        "stage1_train_loss_at_best",
        "stage1_val_loss_at_best",
        "stage1_oos_mse",
        "stage1_final_train_loss",
        "stage1_final_val_loss",
        "stage1_best_epoch",
        "stage1_epochs",
    ]
    for col in optional_cols:
        if col in all_results.columns:
            agg_spec[f"{col}_mean"] = (col, "mean")
            agg_spec[f"{col}_std"] = (col, "std")
            agg_spec[f"{col}_se"] = (col, _se)

    return (
        all_results
        .groupby(["a_j", "theory", "method"], as_index=False)
        .agg(**agg_spec)
        .sort_values(["a_j", "theory", "method"])
        .reset_index(drop=True)
    )


def diagnose_monotonicity_by_seed(all_results: pd.DataFrame):
    """Bound monotonicity diagnostics by seed, theory, and method."""
    rows = []

    for (seed, theory, method), g in all_results.groupby(["seed", "theory", "method"]):
        g = g.sort_values("a_j").reset_index(drop=True)
        diffs = np.diff(g["bound"].to_numpy())

        rows.append({
            "seed": seed,
            "theory": theory,
            "method": method,
            "slope": _safe_slope(g, "a_j", "bound"),
            "spearman": _safe_corr(g, "a_j", "bound", "spearman"),
            "kendall": _safe_corr(g, "a_j", "bound", "kendall"),
            "num_decreases": int((diffs < 0).sum()),
            "first": float(g["bound"].iloc[0]),
            "last": float(g["bound"].iloc[-1]),
        })

    return pd.DataFrame(rows)


def diagnose_rank_correlations_by_seed(all_results: pd.DataFrame) -> pd.DataFrame:
    """Compute rank correlations with jump strength a_j by seed/theory/method."""
    targets = [
        ("stage1_train_loss_at_best", "stage1_train_loss_at_best"),
        ("stage1_oos_mse", "stage1_oos_mse"),
        ("stage1_final_train_loss", "stage1_final_train_loss"),
        ("stage2_bound", "bound"),
    ]

    rows = []
    for (seed, theory, method), g in all_results.groupby(["seed", "theory", "method"]):
        g = g.sort_values("a_j").reset_index(drop=True)
        for target, value_col in targets:
            if value_col not in g.columns:
                continue

            first_value, last_value = _first_last(g, value_col)
            valid = g[["a_j", value_col]].replace([np.inf, -np.inf], np.nan).dropna()

            rows.append({
                "seed": seed,
                "theory": theory,
                "method": method,
                "target": target,
                "value_col": value_col,
                "n_a_j": int(len(valid)),
                "spearman_a_j": _safe_corr(g, "a_j", value_col, "spearman"),
                "kendall_a_j": _safe_corr(g, "a_j", value_col, "kendall"),
                "pearson_a_j": _safe_corr(g, "a_j", value_col, "pearson"),
                "linear_slope_a_j": _safe_slope(g, "a_j", value_col),
                "first_a_j_value": first_value,
                "last_a_j_value": last_value,
            })

    return pd.DataFrame(rows)


def summarize_rank_correlations(corr_by_seed: pd.DataFrame) -> pd.DataFrame:
    """Average seed-level correlation diagnostics across seeds."""
    if corr_by_seed is None or corr_by_seed.empty:
        return pd.DataFrame()

    return (
        corr_by_seed
        .groupby(["theory", "method", "target", "value_col"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            spearman_mean=("spearman_a_j", "mean"),
            spearman_std=("spearman_a_j", "std"),
            spearman_se=("spearman_a_j", _se),
            kendall_mean=("kendall_a_j", "mean"),
            kendall_std=("kendall_a_j", "std"),
            kendall_se=("kendall_a_j", _se),
            pearson_mean=("pearson_a_j", "mean"),
            pearson_std=("pearson_a_j", "std"),
            pearson_se=("pearson_a_j", _se),
            slope_mean=("linear_slope_a_j", "mean"),
            slope_std=("linear_slope_a_j", "std"),
            slope_se=("linear_slope_a_j", _se),
        )
        .sort_values(["target", "theory", "method"])
        .reset_index(drop=True)
    )


def diagnose_rank_correlations_on_seed_means(summary: pd.DataFrame) -> pd.DataFrame:
    """Compute correlation with a_j after averaging variables across seeds."""
    targets = [
        ("stage1_train_loss_at_best_mean", "stage1_train_loss_at_best_mean"),
        ("stage1_oos_mse_mean", "stage1_oos_mse_mean"),
        ("stage1_final_train_loss_mean", "stage1_final_train_loss_mean"),
        ("stage2_bound_mean", "bound_mean"),
    ]

    rows = []
    for (theory, method), g in summary.groupby(["theory", "method"]):
        g = g.sort_values("a_j").reset_index(drop=True)
        for target, value_col in targets:
            if value_col not in g.columns:
                continue

            first_value, last_value = _first_last(g, value_col)
            valid = g[["a_j", value_col]].replace([np.inf, -np.inf], np.nan).dropna()
            rows.append({
                "theory": theory,
                "method": method,
                "target": target,
                "value_col": value_col,
                "n_a_j": int(len(valid)),
                "spearman_a_j": _safe_corr(g, "a_j", value_col, "spearman"),
                "kendall_a_j": _safe_corr(g, "a_j", value_col, "kendall"),
                "pearson_a_j": _safe_corr(g, "a_j", value_col, "pearson"),
                "linear_slope_a_j": _safe_slope(g, "a_j", value_col),
                "first_a_j_value": first_value,
                "last_a_j_value": last_value,
            })

    return pd.DataFrame(rows)
