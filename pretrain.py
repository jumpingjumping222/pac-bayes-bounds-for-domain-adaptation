from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from hsv_bsm_bates_data import generate_bs_dataset, generate_heston_dataset
from model import ResMLP
from synthetic_data_generator import (
    DEFAULT_SYNTHETIC_DATA_ROOT,
    generate_bs_synthetic_dataset,
    load_or_generate_synthetic_dataset,
    make_bs_data_config,
)


FEATURE_COLUMNS_BSM = ["S", "K", "tau", "r", "d", "IV1"]
FEATURE_COLUMNS_HESTON = [
    "S",
    "K",
    "tau",
    "r",
    "d",
    "v0",
    "theta",
    "kappa",
    "xi",
    "rho",
]
TARGET_COLUMNS = ["y_put", "delta_put"]
DEFAULT_PRETRAIN_CHECKPOINT_ROOT = Path("output/checkpoints/pretrain")

BS_FEATURE_CONFIGS = {
    "raw_sk": ["S", "K", "tau", "r", "d", "IV1"],
    "sm": ["S", "m", "tau", "r", "d", "IV1"],
}

THEORY_CONFIGS = {
    "BS": {
        "feature_cols": FEATURE_COLUMNS_BSM,
        "generator": generate_bs_dataset,
        "checkpoint": Path("output/resmlp_source_pretrained_bs_put_no_arb.pt"),
        "history": Path("output/source_training_loss_bs_put.csv"),
    },
    "HESTON": {
        "feature_cols": FEATURE_COLUMNS_HESTON,
        "generator": generate_heston_dataset,
        "checkpoint": Path("output/resmlp_source_pretrained_heston_put_no_arb.pt"),
        "history": Path("output/source_training_loss_heston_put.csv"),
    },
}


class SourceDataset(Dataset):
    def __init__(self, X, y, delta, device):
        self.X = torch.tensor(X, dtype=torch.float32, device=device)
        self.y = torch.tensor(y.reshape(-1, 1), dtype=torch.float32, device=device)
        self.delta = torch.tensor(delta.reshape(-1, 1), dtype=torch.float32, device=device)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i], self.delta[i]


def normalize_theory(theory: str) -> str:
    key = str(theory).strip().upper()
    if key in {"BSM", "BLACK_SCHOLES", "BLACK-SCHOLES"}:
        key = "BS"
    if key not in THEORY_CONFIGS:
        raise ValueError(f"Unknown theory={theory!r}. Expected one of {sorted(THEORY_CONFIGS)}.")
    return key


def source_loss(
    model,
    X,
    y,
    delta_true,
    s_idx: int,
    eps: float = 1e-4,
    lambda_price: float = 10.0,
    lambda_delta: float = 2.0,
):
    Xreq = X.detach().clone().requires_grad_(True)
    price_pred = model(Xreq)

    grads = torch.autograd.grad(
        outputs=price_pred.sum(),
        inputs=Xreq,
        create_graph=True,
        retain_graph=True,
    )[0]

    delta_pred = grads[:, s_idx].reshape(-1, 1)
    w = 1.0 / torch.clamp(torch.abs(delta_true), min=eps)

    price_loss = torch.mean(w * torch.abs(price_pred - y))
    delta_loss = torch.mean(torch.abs(delta_pred - delta_true))
    loss = lambda_price * price_loss + lambda_delta * delta_loss

    return loss, (price_loss.detach(), delta_loss.detach())


def generate_theory_dataset(
    theory: str,
    n_samples: int = 800_000,
    seed: int = 123,
) -> pd.DataFrame:
    theory = normalize_theory(theory)
    if theory == "BS":
        data_config = make_bs_data_config(
            preset="bs_lm_m05_tau_u_sigma_u",
            n_samples=n_samples,
            seed=seed,
        )
        return generate_bs_synthetic_dataset(data_config).drop(
            columns=["data_id", "sampling_mode"],
            errors="ignore",
        )
    generator = THEORY_CONFIGS[theory]["generator"]
    return generator(n_samples=n_samples, seed=seed, cp_flag="P")


def load_or_generate_theory_dataset(
    theory: str,
    dataset_path: Path,
    generate_dataset: bool,
    n_samples: int = 800_000,
    seed: int = 123,
) -> pd.DataFrame:
    dataset_path = Path(dataset_path)
    if dataset_path.exists() and not generate_dataset:
        return pd.read_parquet(dataset_path)

    df = generate_theory_dataset(
        theory=theory,
        n_samples=n_samples,
        seed=seed,
    )
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dataset_path, index=False)
    return df


def train_source_model(
    theory: str,
    model,
    loader,
    s_idx: int,
    lr: float = 2e-4,
    epochs: int = 200,
    eps: float = 1e-4,
    lambda_price: float = 10.0,
    lambda_delta: float = 2.0,
):
    history = {
        "loss": [],
        "price_loss": [],
        "delta_loss": [],
    }

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(1, epochs + 1):
        total = 0.0
        price_total = 0.0
        delta_total = 0.0
        n = 0

        for X, y, delta in loader:
            opt.zero_grad(set_to_none=True)
            loss, (price_loss, delta_loss) = source_loss(
                model,
                X,
                y,
                delta,
                s_idx=s_idx,
                eps=eps,
                lambda_price=lambda_price,
                lambda_delta=lambda_delta,
            )
            loss.backward()
            opt.step()

            batch_size = X.shape[0]
            total += float(loss.detach().cpu()) * batch_size
            price_total += float(price_loss.cpu()) * batch_size
            delta_total += float(delta_loss.cpu()) * batch_size
            n += batch_size

        history["loss"].append(total / n)
        history["price_loss"].append(price_total / n)
        history["delta_loss"].append(delta_total / n)

        print(
            f"[{theory} source pretrain] epoch={epoch:03d}/{epochs} "
            f"loss={total / n:.6f} price={price_total / n:.6f} "
            f"delta={delta_total / n:.6f}"
        )

    return model, history


def _slug_float(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def model_id(model_config: dict) -> str:
    feature_id = str(model_config.get("feature_id", "features"))
    hidden_dim = int(model_config.get("hidden_dim", 22))
    num_hidden = int(model_config.get("num_hidden", 16))
    dropout = _slug_float(float(model_config.get("dropout", 0.0)))
    seed = int(model_config.get("seed", 123))
    return f"{feature_id}_h{hidden_dim}_l{num_hidden:02d}_d{dropout}_s{seed}"


def make_model_config(
    feature_id: str = "raw_sk",
    feature_cols: Optional[list[str]] = None,
    hidden_dim: int = 22,
    num_hidden: int = 16,
    dropout: float = 0.0,
    seed: int = 123,
) -> dict:
    if feature_cols is None:
        if feature_id not in BS_FEATURE_CONFIGS:
            raise ValueError(
                f"Unknown feature_id={feature_id!r}. Expected one of {sorted(BS_FEATURE_CONFIGS)}."
            )
        feature_cols = BS_FEATURE_CONFIGS[feature_id]
    return {
        "feature_id": feature_id,
        "feature_cols": list(feature_cols),
        "hidden_dim": int(hidden_dim),
        "num_hidden": int(num_hidden),
        "dropout": float(dropout),
        "seed": int(seed),
    }


def default_bs_model_configs(
    depths: tuple[int, ...] = (3, 5, 10, 16),
    feature_ids: tuple[str, ...] = ("raw_sk", "sm"),
    hidden_dim: int = 22,
    dropout: float = 0.0,
    seed: int = 123,
) -> list[dict]:
    configs = []
    for feature_id in feature_ids:
        for depth in depths:
            configs.append(
                make_model_config(
                    feature_id=feature_id,
                    hidden_dim=hidden_dim,
                    num_hidden=depth,
                    dropout=dropout,
                    seed=seed,
                )
            )
    return configs


def _resolve_model_config(
    default_feature_cols: list[str],
    model_config: Optional[dict],
    hidden_dim: int = 22,
    num_hidden: int = 16,
    dropout: float = 0.0,
    seed: int = 123,
) -> dict:
    config = {
        "feature_id": "default",
        "feature_cols": list(default_feature_cols),
        "hidden_dim": int(hidden_dim),
        "num_hidden": int(num_hidden),
        "dropout": float(dropout),
        "seed": int(seed),
    }
    if model_config:
        config.update(model_config)
        config["feature_cols"] = list(config.get("feature_cols", default_feature_cols))
        config["hidden_dim"] = int(config.get("hidden_dim", hidden_dim))
        config["num_hidden"] = int(config.get("num_hidden", num_hidden))
        config["dropout"] = float(config.get("dropout", dropout))
        config["seed"] = int(config.get("seed", seed))
    return config


def load_pretrained_model(
    checkpoint_path: Path,
    input_dim: int,
    device: str,
    model_config: Optional[dict] = None,
):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_model_config = checkpoint.get("model_config", {})
    checkpoint_feature_cols = checkpoint.get("feature_cols", ["x"] * input_dim)
    merged_model_config = dict(checkpoint_model_config)
    if model_config:
        merged_model_config.update(model_config)
    resolved_config = _resolve_model_config(
        default_feature_cols=checkpoint_feature_cols,
        model_config=merged_model_config,
    )
    model = ResMLP(
        input_dim=len(resolved_config["feature_cols"]),
        hidden_dim=resolved_config["hidden_dim"],
        num_hidden=resolved_config["num_hidden"],
        dropout=resolved_config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def run_theory_pretrain(
    theory: str,
    dataset_path: Path,
    checkpoint_path: Optional[Path] = None,
    history_path: Optional[Path] = None,
    data_config: Optional[dict] = None,
    model_config: Optional[dict] = None,
    generate_dataset: bool = False,
    retrain_model: bool = False,
    n_samples: int = 800_000,
    epochs: int = 200,
    batches_per_epoch: int = 600,
    learning_rate: float = 2e-4,
    lambda_price: float = 10.0,
    lambda_delta: float = 2.0,
    seed: int = 123,
    device: Optional[str] = None,
):
    theory = normalize_theory(theory)
    config = THEORY_CONFIGS[theory]
    resolved_model_config = _resolve_model_config(
        default_feature_cols=config["feature_cols"],
        model_config=model_config,
        seed=seed,
    )
    feature_cols = resolved_model_config["feature_cols"]
    seed = resolved_model_config["seed"]
    checkpoint_path = Path(checkpoint_path or config["checkpoint"])
    history_path = Path(history_path or config["history"])

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    df = None
    if generate_dataset or retrain_model or not checkpoint_path.exists():
        df = load_or_generate_theory_dataset(
            theory=theory,
            dataset_path=dataset_path,
            generate_dataset=generate_dataset,
            n_samples=n_samples,
            seed=seed,
        )

    if checkpoint_path.exists() and not retrain_model:
        return load_pretrained_model(
            checkpoint_path=checkpoint_path,
            input_dim=len(feature_cols),
            device=device,
            model_config=model_config,
        ), None

    missing = sorted(set(feature_cols + TARGET_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"{theory} pretrain dataset is missing required columns: {missing}")

    X = df[feature_cols].to_numpy(np.float32)
    y = df["y_put"].to_numpy(np.float32)
    delta = df["delta_put"].to_numpy(np.float32)

    batch_size = max(int(len(df) // batches_per_epoch), 1)
    ds = SourceDataset(X, y, delta, device=device)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model = ResMLP(
        input_dim=len(feature_cols),
        hidden_dim=resolved_model_config["hidden_dim"],
        num_hidden=resolved_model_config["num_hidden"],
        dropout=resolved_model_config["dropout"],
    ).to(device)
    model, history = train_source_model(
        theory=theory,
        model=model,
        loader=loader,
        s_idx=feature_cols.index("S"),
        lr=learning_rate,
        epochs=epochs,
        eps=1e-4,
        lambda_price=lambda_price,
        lambda_delta=lambda_delta,
    )

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "theory": theory,
            "feature_cols": feature_cols,
            "target_cols": TARGET_COLUMNS,
            "data_id": (data_config or {}).get("data_id"),
            "dataset_path": str(dataset_path),
            "data_config": data_config,
            "model_config": resolved_model_config,
            "train_config": {
                "n_samples": int(len(df)),
                "epochs": int(epochs),
                "batches_per_epoch": int(batches_per_epoch),
                "learning_rate": float(learning_rate),
                "seed": int(seed),
            },
            "loss_weights": {
                "price": lambda_price,
                "delta": lambda_delta,
            },
            "n_samples": int(len(df)),
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "seed": int(seed),
        },
        checkpoint_path,
    )

    history_df = pd.DataFrame(history)
    history_df["epoch"] = np.arange(1, len(history_df) + 1)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(history_path, index=False)

    print(f"[{theory} source pretrain] Saved checkpoint to {checkpoint_path}")
    print(f"[{theory} source pretrain] Saved history to {history_path}")
    return model, history_df


def run_bs_synthetic_pretrain_grid(
    data_preset: str = "bs_lm_m05_tau_u_sigma_u",
    n_samples: int = 800_000,
    data_seed: int = 123,
    model_configs: Optional[list[dict]] = None,
    data_id: Optional[str] = None,
    data_overrides: Optional[dict] = None,
    synthetic_data_root: Path = DEFAULT_SYNTHETIC_DATA_ROOT,
    checkpoint_root: Path = DEFAULT_PRETRAIN_CHECKPOINT_ROOT,
    regenerate_dataset: bool = False,
    retrain_models: bool = False,
    epochs: int = 200,
    batches_per_epoch: int = 600,
    learning_rate: float = 2e-4,
    lambda_price: float = 10.0,
    lambda_delta: float = 2.0,
    device: Optional[str] = None,
) -> dict:
    data_config = make_bs_data_config(
        preset=data_preset,
        n_samples=n_samples,
        seed=data_seed,
        data_id=data_id,
        overrides=data_overrides,
    )
    _, materialized_data_config = load_or_generate_synthetic_dataset(
        data_config=data_config,
        output_root=synthetic_data_root,
        regenerate=regenerate_dataset,
    )

    if model_configs is None:
        model_configs = default_bs_model_configs(seed=data_seed)

    data_id_value = materialized_data_config["data_id"]
    checkpoint_dir = Path(checkpoint_root) / "BS" / data_id_value
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    results = []
    manifest_rows = []
    for config in model_configs:
        resolved_model_config = _resolve_model_config(
            default_feature_cols=FEATURE_COLUMNS_BSM,
            model_config=config,
            seed=int(config.get("seed", data_seed)),
        )
        run_id = model_id(resolved_model_config)
        checkpoint_path = checkpoint_dir / f"{run_id}.pt"
        history_path = checkpoint_dir / f"{run_id}_history.csv"
        model, history = run_theory_pretrain(
            theory="BS",
            dataset_path=Path(materialized_data_config["dataset_path"]),
            checkpoint_path=checkpoint_path,
            history_path=history_path,
            data_config=materialized_data_config,
            model_config=resolved_model_config,
            generate_dataset=False,
            retrain_model=retrain_models,
            n_samples=n_samples,
            epochs=epochs,
            batches_per_epoch=batches_per_epoch,
            learning_rate=learning_rate,
            lambda_price=lambda_price,
            lambda_delta=lambda_delta,
            seed=resolved_model_config["seed"],
            device=device,
        )
        results.append(
            {
                "run_id": run_id,
                "checkpoint_path": str(checkpoint_path),
                "history_path": str(history_path),
                "model_config": resolved_model_config,
                "history": history,
                "model": model,
            }
        )
        manifest_rows.append(
            {
                "data_id": data_id_value,
                "run_id": run_id,
                "checkpoint_path": str(checkpoint_path),
                "history_path": str(history_path),
                "feature_id": resolved_model_config.get("feature_id"),
                "feature_cols": ",".join(resolved_model_config["feature_cols"]),
                "hidden_dim": resolved_model_config["hidden_dim"],
                "num_hidden": resolved_model_config["num_hidden"],
                "dropout": resolved_model_config["dropout"],
                "seed": resolved_model_config["seed"],
                "data_preset": materialized_data_config.get("preset"),
                "n_samples": materialized_data_config.get("n_samples"),
            }
        )

    manifest_path = checkpoint_dir / "manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    return {
        "data_config": materialized_data_config,
        "checkpoint_dir": checkpoint_dir,
        "manifest_path": manifest_path,
        "runs": results,
    }


def run_all_theory_pretrains(
    bs_dataset_path: Path,
    heston_dataset_path: Path,
    bs_checkpoint_path: Optional[Path] = None,
    heston_checkpoint_path: Optional[Path] = None,
    generate_datasets: bool = False,
    retrain_models: bool = False,
):
    bs_model, bs_history = run_theory_pretrain(
        theory="BS",
        dataset_path=bs_dataset_path,
        checkpoint_path=bs_checkpoint_path,
        generate_dataset=generate_datasets,
        retrain_model=retrain_models,
    )
    heston_model, heston_history = run_theory_pretrain(
        theory="HESTON",
        dataset_path=heston_dataset_path,
        checkpoint_path=heston_checkpoint_path,
        generate_dataset=generate_datasets,
        retrain_model=retrain_models,
    )
    return {
        "BS": {"model": bs_model, "history": bs_history},
        "HESTON": {"model": heston_model, "history": heston_history},
    }
