from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_FIGURE_DIR = Path("outputs") / "figures"


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
