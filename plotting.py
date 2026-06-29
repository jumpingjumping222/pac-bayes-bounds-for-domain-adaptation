from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_FIGURE_DIR = Path("outputs") / "figures"


DEFAULT_SELECTION_CRITERION_NAMES = {
    "pac_score": "Pac-score",
    "best_val": "Best-val",
    "expected_train_nll_subsample_mean": (
        r"$n\mathbb E_{\mathbf w\sim Q_k^*}"
        r"\widehat L_k^{\mathrm{nll}}(\mathbf w)$"
    ),
    "kl_subsample_mean": "KL",
    "posterior_mean_sse_subsample_mean": "Fitness",
    "posterior_cov_sse_subsample_mean": "Flatness",
    "theta_diff_sq_subsample_mean": "Adaptation",
    "logdet_A_subsample_mean": "Region width",
}


def _ensure_dir(save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def _safe_float_name(x: float):
    """Convert float to filename-safe string."""
    return f"{float(x):.6f}".replace(".", "p").replace("-", "minus")


def _save_and_maybe_show(fig, save_path: Path, show: bool = True, dpi: int = 300):
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print(f"[Saved figure] {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _load_frame(data) -> pd.DataFrame:
    if isinstance(data, (str, Path)):
        return pd.read_csv(data)
    return data.copy()


def _auto_group_cols(df: pd.DataFrame) -> list[str]:
    candidates = [
        "holdout_seed",
        "seed",
        "method",
        "maturity_bucket",
        "test_start_date",
    ]
    return [col for col in candidates if col in df.columns]


def _heston_minus_bs_wide(
    data,
    value_cols: Iterable[str],
    bucket: Optional[str] = "all",
    group_cols: Optional[list[str]] = None,
    theory_col: str = "theory",
    heston_name: str = "HESTON",
    bs_name: str = "BS",
) -> pd.DataFrame:
    df = _load_frame(data)
    value_cols = list(value_cols)
    required = [theory_col, *value_cols]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Data is missing required columns: {missing}")

    if bucket is not None:
        if "maturity_bucket" not in df.columns:
            raise ValueError("bucket filtering requires a 'maturity_bucket' column.")
        df = df.loc[df["maturity_bucket"].astype(str).eq(str(bucket))].copy()

    if df.empty:
        raise ValueError("No rows remain after filtering.")

    df[theory_col] = df[theory_col].astype(str).str.upper()
    heston_name = str(heston_name).upper()
    bs_name = str(bs_name).upper()
    df = df.loc[df[theory_col].isin([heston_name, bs_name])].copy()

    if group_cols is None:
        group_cols = _auto_group_cols(df)
    group_cols = [col for col in group_cols if col in df.columns and col != theory_col]

    id_cols = [*group_cols, theory_col]
    compact = (
        df[id_cols + value_cols]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=value_cols, how="all")
        .groupby(id_cols, as_index=False)
        .first()
    )
    wide = compact.pivot(index=group_cols, columns=theory_col, values=value_cols)
    if heston_name not in wide.columns.get_level_values(1):
        raise ValueError(f"No rows found for theory={heston_name!r}.")
    if bs_name not in wide.columns.get_level_values(1):
        raise ValueError(f"No rows found for theory={bs_name!r}.")

    out = wide.index.to_frame(index=False)
    for col in value_cols:
        out[f"{col}_heston"] = wide[(col, heston_name)].to_numpy()
        out[f"{col}_bs"] = wide[(col, bs_name)].to_numpy()
        out[f"{col}_diff"] = out[f"{col}_heston"] - out[f"{col}_bs"]

    sort_cols = [col for col in ["holdout_seed", "seed", "method", "maturity_bucket"] if col in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def plot_universal_overall_metric_differences(
    aligned_summary,
    oos_metric: str = "test_bsiv_median",
    selection_cols: Optional[list[str]] = None,
    selection_names: Optional[dict[str, str]] = None,
    bucket: str = "all",
    group_cols: Optional[list[str]] = None,
    theory_col: str = "theory",
    heston_name: str = "HESTON",
    bs_name: str = "BS",
    ncols: int = 2,
    figsize: tuple[float, float] = (14.0, 13.0),
    annotate_values: bool = False,
    save_dir=DEFAULT_FIGURE_DIR,
    filename: Optional[str] = None,
    show: bool = True,
    dpi: int = 300,
):
    selection_names = (
        DEFAULT_SELECTION_CRITERION_NAMES
        if selection_names is None
        else selection_names
    )
    if selection_cols is None:
        selection_cols = list(selection_names.keys())
    if len(selection_cols) == 0:
        raise ValueError("selection_cols must contain at least one column.")

    value_cols = [*selection_cols, oos_metric]
    diff_df = _heston_minus_bs_wide(
        aligned_summary,
        value_cols=value_cols,
        bucket=bucket,
        group_cols=group_cols,
        theory_col=theory_col,
        heston_name=heston_name,
        bs_name=bs_name,
    )

    x_col = "holdout_seed" if "holdout_seed" in diff_df.columns else diff_df.index.name
    if x_col is None:
        x = np.arange(len(diff_df))
        x_label = "Index"
    else:
        x = diff_df[x_col].to_numpy()
        x_label = x_col

    n_panels = len(selection_cols)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.reshape(-1)
    oos_diff_col = f"{oos_metric}_diff"

    for ax, selection_col in zip(axes_flat, selection_cols):
        criterion_diff_col = f"{selection_col}_diff"
        if criterion_diff_col not in diff_df.columns:
            raise ValueError(f"Missing computed diff column: {criterion_diff_col}")

        label = selection_names.get(selection_col, selection_col)
        y_left = diff_df[criterion_diff_col].to_numpy(dtype=float)
        y_right = diff_df[oos_diff_col].to_numpy(dtype=float)

        left_line = ax.plot(
            x,
            y_left,
            marker="o",
            linewidth=1.8,
            color="#1f77b4",
            label=f"{label} diff",
        )[0]
        ax.axhline(0.0, color="#6f6f6f", linewidth=0.9, linestyle="--")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel(x_label)
        ax.set_ylabel(f"{label}: HESTON - BS", color=left_line.get_color())
        ax.tick_params(axis="y", labelcolor=left_line.get_color())
        ax.grid(True, axis="y", linewidth=0.5, alpha=0.35)

        ax_right = ax.twinx()
        right_line = ax_right.plot(
            x,
            y_right,
            marker="s",
            linewidth=1.6,
            color="#d62728",
            label=f"{oos_metric} diff",
        )[0]
        ax_right.axhline(0.0, color="#6f6f6f", linewidth=0.9, linestyle=":")
        ax_right.set_ylabel(f"{oos_metric}: HESTON - BS", color=right_line.get_color())
        ax_right.tick_params(axis="y", labelcolor=right_line.get_color())

        if annotate_values:
            for xi, yi in zip(x, y_left):
                if np.isfinite(yi):
                    ax.annotate(f"{yi:.3g}", (xi, yi), textcoords="offset points", xytext=(0, 5),
                                ha="center", fontsize=7, color=left_line.get_color())
            for xi, yi in zip(x, y_right):
                if np.isfinite(yi):
                    ax_right.annotate(f"{yi:.3g}", (xi, yi), textcoords="offset points", xytext=(0, -10),
                                      ha="center", fontsize=7, color=right_line.get_color())

        ax.legend([left_line, right_line], [left_line.get_label(), right_line.get_label()],
                  frameon=False, fontsize=8, loc="best")

    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)

    title = (
        f"Overall HESTON - BS differences, bucket={bucket}, "
        f"OOS metric={oos_metric}"
    )
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if filename is None:
        filename = f"universal_overall_diff_{oos_metric}.png"
    if save_dir is not None:
        save_dir = _ensure_dir(save_dir)
        _save_and_maybe_show(fig, save_dir / filename, show=show, dpi=dpi)
    elif show:
        plt.show()

    return fig, diff_df


def plot_bounds(
    results_df: pd.DataFrame,
    save_dir=DEFAULT_FIGURE_DIR,
    filename: str = "study3_pacbayes_bounds.png",
    show: bool = True,
    dpi: int = 300,
):
    """Plot PAC-Bayes bounds against Bates jump strength a_j."""
    save_dir = _ensure_dir(save_dir)
    save_path = save_dir / filename

    plot_df = results_df.sort_values(["theory", "method", "a_j"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7, 5))

    for (theory, method), g in plot_df.groupby(["theory", "method"]):
        g = g.sort_values("a_j")
        linestyle = "-" if method == "TL" else "--"
        marker = "o" if theory == "BSM" else "s"
        ax.plot(
            g["a_j"],
            g["bound"],
            linewidth=2.3,
            linestyle=linestyle,
            marker=marker,
            label=f"{theory}-{method}",
        )

    ax.set_xlabel(r"Bates jump strength: $a_J$", fontsize=12)
    ax.set_ylabel("PAC-Bayes bound", fontsize=12)
    ax.set_title(r"Study 3 PAC-Bayes Bounds", fontsize=13)

    ax.set_xlim(plot_df["a_j"].min(), plot_df["a_j"].max())
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    _save_and_maybe_show(fig, save_path, show=show, dpi=dpi)


def collect_best_second_sgd_history(all_details: Dict):
    """Collect the second-SGD path only for the best lambda selected by the full bound."""
    frames = []
    for _, detail in all_details.items():
        a_j = float(detail.get("a_j"))
        theory = detail.get("theory")
        for method, key in [("TL", "TL_best"), ("DL", "DL_best")]:
            cand = detail[key]
            hist = cand["posterior"].history
            if hist is None or len(hist) == 0:
                continue
            h = hist.copy()
            h.insert(0, "method", method)
            h.insert(1, "theory", theory)
            h.insert(2, "a_j", a_j)
            h.insert(3, "lambda", cand["lambda"])
            h.insert(4, "bound", cand["bound"])
            frames.append(h)

    if len(frames) == 0:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def collect_all_second_sgd_history(all_details: Dict):
    """Collect the second-SGD path for every lambda candidate."""
    frames = []
    for _, detail in all_details.items():
        a_j = float(detail.get("a_j"))
        theory = detail.get("theory")
        for method, grid_key in [("TL", "TL_grid"), ("DL", "DL_grid")]:
            for cand in detail[grid_key]:
                hist = cand["posterior"].history
                if hist is None or len(hist) == 0:
                    continue
                h = hist.copy()
                h.insert(0, "method", method)
                h.insert(1, "theory", theory)
                h.insert(2, "a_j", a_j)
                h.insert(3, "lambda", cand["lambda"])
                h.insert(4, "bound", cand["bound"])
                frames.append(h)

    if len(frames) == 0:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def plot_second_sgd_history(
    history_df: pd.DataFrame,
    a_j_value: float,
    metric: str = "dist_to_prior_l2",
    lambda_value: Optional[float] = None,
    theory: Optional[str] = None,
    save_dir=DEFAULT_FIGURE_DIR,
    filename: Optional[str] = None,
    show: bool = True,
    dpi: int = 300,
):
    """
    Plot one diagnostic from the recorded second-SGD path.

    metric examples:
      - dist_to_prior_l2
      - dist_to_stage1_init_l2
      - kl
      - emp_loss
      - obj
    """
    save_dir = _ensure_dir(save_dir)

    if filename is None:
        a_j_name = _safe_float_name(a_j_value)
        theory_name = "all" if theory is None else str(theory).lower()
        if lambda_value is None:
            filename = f"second_sgd_a_j_{a_j_name}_{theory_name}_{metric}.png"
        else:
            lam_name = _safe_float_name(lambda_value)
            filename = f"second_sgd_a_j_{a_j_name}_{theory_name}_lambda_{lam_name}_{metric}.png"

    save_path = save_dir / filename

    plot_df = history_df[np.isclose(history_df["a_j"], float(a_j_value))].copy()
    if theory is not None:
        plot_df = plot_df[plot_df["theory"].astype(str) == str(theory)].copy()
    if lambda_value is not None:
        plot_df = plot_df[np.isclose(plot_df["lambda"], float(lambda_value))].copy()

    if plot_df.empty:
        print(
            f"[Warning] No history found for a_j={a_j_value}, "
            f"theory={theory}, lambda={lambda_value}, metric={metric}."
        )
        return

    if metric not in plot_df.columns:
        raise ValueError(
            f"Metric '{metric}' not found in history_df. "
            f"Available columns are: {list(plot_df.columns)}"
        )

    fig, ax = plt.subplots(figsize=(7, 5))

    for (theory_name, method), g in plot_df.groupby(["theory", "method"]):
        g = g.sort_values("step")
        ax.plot(g["step"], g[metric], linewidth=2.0, marker="o", label=f"{theory_name}-{method}")

    ax.set_xlabel("Second-SGD step", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f"Second-SGD path: a_J={a_j_value}, metric={metric}", fontsize=13)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    _save_and_maybe_show(fig, save_path, show=show, dpi=dpi)


def collect_stage1_sgd_history(all_details: Dict):
    """Collect the first-stage empirical SGD path for TL and DL."""
    frames = []
    for _, detail in all_details.items():
        a_j = float(detail.get("a_j"))
        theory = detail.get("theory")
        for method, key in [("TL", "TL_stage1_history"), ("DL", "DL_stage1_history")]:
            hist = detail.get(key)
            if hist is None or len(hist) == 0:
                continue
            h = hist.copy()
            h.insert(0, "method", method)
            h.insert(1, "theory", theory)
            h.insert(2, "a_j", a_j)
            frames.append(h)

    if len(frames) == 0:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def plot_stage1_sgd_history(
    history_df: pd.DataFrame,
    a_j_value: float,
    metric: str = "val_loss",
    theory: Optional[str] = None,
    save_dir=DEFAULT_FIGURE_DIR,
    filename: Optional[str] = None,
    show: bool = True,
    dpi: int = 300,
):
    """
    Plot one diagnostic from the recorded first-stage empirical SGD path.

    metric examples:
      - train_loss
      - val_loss
      - best_val
      - shift_from_init_l2
      - param_norm_l2
    """
    save_dir = _ensure_dir(save_dir)

    if filename is None:
        a_j_name = _safe_float_name(a_j_value)
        theory_name = "all" if theory is None else str(theory).lower()
        filename = f"stage1_sgd_a_j_{a_j_name}_{theory_name}_{metric}.png"

    save_path = save_dir / filename

    plot_df = history_df[np.isclose(history_df["a_j"], float(a_j_value))].copy()
    if theory is not None:
        plot_df = plot_df[plot_df["theory"].astype(str) == str(theory)].copy()

    if plot_df.empty:
        print(f"[Warning] No Stage-1 history found for a_j={a_j_value}, theory={theory}, metric={metric}.")
        return

    if metric not in plot_df.columns:
        raise ValueError(
            f"Metric '{metric}' not found in history_df. "
            f"Available columns are: {list(plot_df.columns)}"
        )

    fig, ax = plt.subplots(figsize=(7, 5))

    for (theory_name, method), g in plot_df.groupby(["theory", "method"]):
        g = g.sort_values("epoch")
        ax.plot(g["epoch"], g[metric], linewidth=2.0, marker="o", label=f"{theory_name}-{method}")

    ax.set_xlabel("Stage-1 epoch", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f"Stage-1 empirical SGD path: a_J={a_j_value}, metric={metric}", fontsize=13)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    _save_and_maybe_show(fig, save_path, show=show, dpi=dpi)


def plot_multi_seed_summary(
    summary: pd.DataFrame,
    save_dir=DEFAULT_FIGURE_DIR,
    filename: str = "study3_pacbayes_bounds_multi_seed_summary.png",
    show: bool = True,
    dpi: int = 300,
):
    """Plot mean bounds across seeds with +/- 1 standard error bands."""
    save_dir = _ensure_dir(save_dir)
    save_path = save_dir / filename

    plot_df = summary.sort_values(["theory", "method", "a_j"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7, 5))

    for (theory, method), g in plot_df.groupby(["theory", "method"]):
        g = g.sort_values("a_j")
        linestyle = "-" if method == "TL" else "--"
        marker = "o" if theory == "BSM" else "s"
        ax.plot(
            g["a_j"],
            g["bound_mean"],
            linewidth=2.3,
            linestyle=linestyle,
            marker=marker,
            label=f"{theory}-{method} mean",
        )
        ax.fill_between(
            g["a_j"],
            g["bound_mean"] - g["bound_se"],
            g["bound_mean"] + g["bound_se"],
            alpha=0.16,
        )

    ax.set_xlabel(r"Bates jump strength: $a_J$", fontsize=12)
    ax.set_ylabel("PAC-Bayes bound", fontsize=12)
    ax.set_title(r"Study 3 Multi-seed PAC-Bayes Bounds", fontsize=13)

    ax.set_xlim(plot_df["a_j"].min(), plot_df["a_j"].max())
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    _save_and_maybe_show(fig, save_path, show=show, dpi=dpi)
