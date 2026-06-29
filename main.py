import json
from pathlib import Path

import pandas as pd

from calibration import get_heston_dataset
from finetune import run_finetune_experiments, run_rolling_finetune_experiments
from pactran_blr_score import (
    align_pactran_finetune,
    align_rolling_pactran_finetune,
    run_pactran_blr_scores,
    run_rolling_pactran_blr_scores,
)
from pretrain import run_all_theory_pretrains
from repeated_random_holdout import run_repeated_random_holdout_experiments


# ========= Data preparation
RUN_HESTON_CALIBRATION = False
RUN_THEORY_PRETRAIN = False

HESTON_DATASET_PATH = Path("data/heston_calibrated_dataset_raw.parquet")
BS_THEORY_DATASET_PATH = Path("data/synthetic_source_bs_put_800k_v2.parquet")
HESTON_THEORY_DATASET_PATH = Path("data/synthetic_source_heston_put_800k_v2.parquet")
BS_PRETRAIN_CHECKPOINT_PATH = Path("pt_checkpoints/resmlp_source_pretrained_bs_put_no_arb.pt")
HESTON_PRETRAIN_CHECKPOINT_PATH = Path("pt_checkpoints/resmlp_source_pretrained_heston_put_no_arb.pt")

# ========= Experiment design
OOS_MODE = "fixed"  # options: "fixed", "rolling"
RUN_FINETUNE = True
RUN_PACTRAN_SCORE = True

USE_MATURITY_BUCKETS = False
MATURITY_BINS = [0.0, 1.0 / 12.0, 3.0 / 12.0, 6.0 / 12.0, 1.0, 2.0, 3.0]
MATURITY_LABELS = ["0-1m", "1-3m", "3-6m", "6m-1y", "1y-2y", "2y-3y"]

# ========= Fixed OOS setting
FIXED_TEST_START_DATE = "2024-01-01"

# ========= Rolling OOS setting
ROLLING_TRAIN_DAYS = 20
ROLLING_TEST_DAYS = 20
ROLLING_STEP_DAYS = 20

# ========= Repeated random holdout OOS setting
RUN_REPEATED_RANDOM_HOLDOUT = True
HOLDOUT_TRAIN_SIZE = 10_0
HOLDOUT_TEST_SIZE = 5_0
HOLDOUT_BUCKET_TYPE = "maturity"  # options: "maturity", "delta"

# ========= Model / finetune setting
SEEDS = list(range(10))
RUN_THEORIES = ("BS", "HESTON")
RUN_METHODS = ("TL",)
FINETUNE_VAL_RATIO = 0.2
FINETUNE_TL_LR = 1e-4
FINETUNE_DL_LR = 1e-3
FINETUNE_MAX_EPOCHS = 200
FINETUNE_EARLY_STOP_CONSECUTIVE = 2

# ========= PACTran score setting
PACTRAN_TARGET_COL = "mid_price"
PACTRAN_SIGMA2 = 0.001
PACTRAN_SIGMA_PI2 = 0.06
PACTRAN_PRIOR_CENTER = "pretrained"  # options: "pretrained", "zero"
PACTRAN_SUBSAMPLE_SIZE = 100_000
PACTRAN_SUBSAMPLE_FRAC = 1
PACTRAN_N_SUBSAMPLES = 1
PACTRAN_SUBSAMPLE_SEED = 123

# ========= Output setting
OUTPUT_ROOT = Path("outputs")
EXPERIMENT_NAME = "repeated_holdout_maturity_100_50_seed_10_vix_15"
EXPERIMENT_DIR = OUTPUT_ROOT / EXPERIMENT_NAME

FINETUNE_OUTPUT_DIR = EXPERIMENT_DIR / "finetune"
PACTRAN_OUTPUT_CSV = EXPERIMENT_DIR / "pactran" / "scores.csv"
PACTRAN_POSTERIOR_DIR = EXPERIMENT_DIR / "pactran" / "posteriors"
ALIGNED_OUTPUT_CSV = EXPERIMENT_DIR / "aligned" / "pactran_finetune_aligned.csv"

FEATURE_COLUMNS_BSM = ["S", "K", "tau", "r", "d", "IV1"]
FEATURE_COLUMNS_HESTON = ["S", "K", "tau", "r", "d", "v0", "theta", "kappa", "xi", "rho"]


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def save_config() -> None:
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "run_heston_calibration": RUN_HESTON_CALIBRATION,
        "run_theory_pretrain": RUN_THEORY_PRETRAIN,
        "heston_dataset_path": HESTON_DATASET_PATH,
        "bs_theory_dataset_path": BS_THEORY_DATASET_PATH,
        "heston_theory_dataset_path": HESTON_THEORY_DATASET_PATH,
        "bs_pretrain_checkpoint_path": BS_PRETRAIN_CHECKPOINT_PATH,
        "heston_pretrain_checkpoint_path": HESTON_PRETRAIN_CHECKPOINT_PATH,
        "oos_mode": OOS_MODE,
        "run_finetune": RUN_FINETUNE,
        "run_pactran_score": RUN_PACTRAN_SCORE,
        "use_maturity_buckets": USE_MATURITY_BUCKETS,
        "maturity_bins": MATURITY_BINS,
        "maturity_labels": MATURITY_LABELS,
        "fixed_test_start_date": FIXED_TEST_START_DATE,
        "rolling_train_days": ROLLING_TRAIN_DAYS,
        "rolling_test_days": ROLLING_TEST_DAYS,
        "rolling_step_days": ROLLING_STEP_DAYS,
        "run_repeated_random_holdout": RUN_REPEATED_RANDOM_HOLDOUT,
        "holdout_train_size": HOLDOUT_TRAIN_SIZE,
        "holdout_test_size": HOLDOUT_TEST_SIZE,
        "holdout_bucket_type": HOLDOUT_BUCKET_TYPE,
        "seeds": SEEDS,
        "run_theories": RUN_THEORIES,
        "run_methods": RUN_METHODS,
        "finetune_val_ratio": FINETUNE_VAL_RATIO,
        "finetune_tl_lr": FINETUNE_TL_LR,
        "finetune_dl_lr": FINETUNE_DL_LR,
        "finetune_max_epochs": FINETUNE_MAX_EPOCHS,
        "finetune_early_stop_consecutive": FINETUNE_EARLY_STOP_CONSECUTIVE,
        "pactran_target_col": PACTRAN_TARGET_COL,
        "pactran_sigma2": PACTRAN_SIGMA2,
        "pactran_sigma_pi2": PACTRAN_SIGMA_PI2,
        "pactran_prior_center": PACTRAN_PRIOR_CENTER,
        "pactran_subsample_size": PACTRAN_SUBSAMPLE_SIZE,
        "pactran_subsample_frac": PACTRAN_SUBSAMPLE_FRAC,
        "pactran_n_subsamples": PACTRAN_N_SUBSAMPLES,
        "pactran_subsample_seed": PACTRAN_SUBSAMPLE_SEED,
        "output_root": OUTPUT_ROOT,
        "experiment_name": EXPERIMENT_NAME,
        "experiment_dir": EXPERIMENT_DIR,
    }
    with (EXPERIMENT_DIR / "config.json").open("w", encoding="utf-8") as f:
        json.dump({k: _jsonable(v) for k, v in config.items()}, f, indent=2)


def checkpoint_specs() -> list[dict]:
    return [
        {
            "name": "BS",
            "checkpoint_path": BS_PRETRAIN_CHECKPOINT_PATH,
            "input_dim": len(FEATURE_COLUMNS_BSM),
            "feature_cols": FEATURE_COLUMNS_BSM,
            "hidden_dim": 22,
            "num_hidden": 16,
            "dropout": 0.0,
        },
        {
            "name": "HESTON",
            "checkpoint_path": HESTON_PRETRAIN_CHECKPOINT_PATH,
            "input_dim": len(FEATURE_COLUMNS_HESTON),
            "feature_cols": FEATURE_COLUMNS_HESTON,
            "hidden_dim": 22,
            "num_hidden": 16,
            "dropout": 0.0,
        },
    ]


def run_fixed_oos(heston_dataset, specs):
    finetune_results = None
    pactran_results = None

    if RUN_FINETUNE:
        finetune_results = run_finetune_experiments(
            heston_dataset=heston_dataset,
            output_dir=FINETUNE_OUTPUT_DIR,
            bs_checkpoint_path=BS_PRETRAIN_CHECKPOINT_PATH,
            heston_checkpoint_path=HESTON_PRETRAIN_CHECKPOINT_PATH,
            seeds=SEEDS,
            theories=RUN_THEORIES,
            methods=RUN_METHODS,
            test_start_date=FIXED_TEST_START_DATE,
            use_maturity_buckets=USE_MATURITY_BUCKETS,
            maturity_bins=MATURITY_BINS,
            maturity_labels=MATURITY_LABELS,
            val_ratio=FINETUNE_VAL_RATIO,
            tl_lr=FINETUNE_TL_LR,
            dl_lr=FINETUNE_DL_LR,
            max_epochs=FINETUNE_MAX_EPOCHS,
            early_stop_consecutive=FINETUNE_EARLY_STOP_CONSECUTIVE,
        )

    if RUN_PACTRAN_SCORE:
        pactran_results = run_pactran_blr_scores(
            target_dataset=heston_dataset,
            checkpoint_specs=specs,
            output_csv=PACTRAN_OUTPUT_CSV,
            target_col=PACTRAN_TARGET_COL,
            date_before=FIXED_TEST_START_DATE,
            only_put=True,
            sigma2=PACTRAN_SIGMA2,
            sigma_pi2=PACTRAN_SIGMA_PI2,
            prior_center=PACTRAN_PRIOR_CENTER,
            posterior_dir=PACTRAN_POSTERIOR_DIR,
            subsample_size=PACTRAN_SUBSAMPLE_SIZE,
            subsample_frac=PACTRAN_SUBSAMPLE_FRAC,
            n_subsamples=PACTRAN_N_SUBSAMPLES,
            subsample_seed=PACTRAN_SUBSAMPLE_SEED,
            use_maturity_buckets=USE_MATURITY_BUCKETS,
            maturity_bins=MATURITY_BINS,
            maturity_labels=MATURITY_LABELS,
        )

    if finetune_results is None:
        finetune_results = pd.read_csv(FINETUNE_OUTPUT_DIR / "results.csv")
    if pactran_results is None:
        pactran_results = pd.read_csv(PACTRAN_OUTPUT_CSV)

    align_pactran_finetune(
        finetune_results=finetune_results,
        pactran_results=pactran_results,
        output_csv=ALIGNED_OUTPUT_CSV,
    )


def run_rolling_oos(heston_dataset, specs):
    finetune_results = None
    pactran_results = None

    if RUN_FINETUNE:
        finetune_results = run_rolling_finetune_experiments(
            heston_dataset=heston_dataset,
            output_dir=FINETUNE_OUTPUT_DIR,
            bs_checkpoint_path=BS_PRETRAIN_CHECKPOINT_PATH,
            heston_checkpoint_path=HESTON_PRETRAIN_CHECKPOINT_PATH,
            seeds=SEEDS,
            theories=RUN_THEORIES,
            methods=RUN_METHODS,
            train_days=ROLLING_TRAIN_DAYS,
            test_days=ROLLING_TEST_DAYS,
            step_days=ROLLING_STEP_DAYS,
            use_maturity_buckets=USE_MATURITY_BUCKETS,
            maturity_bins=MATURITY_BINS,
            maturity_labels=MATURITY_LABELS,
            val_ratio=FINETUNE_VAL_RATIO,
            tl_lr=FINETUNE_TL_LR,
            dl_lr=FINETUNE_DL_LR,
            max_epochs=FINETUNE_MAX_EPOCHS,
            early_stop_consecutive=FINETUNE_EARLY_STOP_CONSECUTIVE,
        )

    if RUN_PACTRAN_SCORE:
        pactran_results = run_rolling_pactran_blr_scores(
            target_dataset=heston_dataset,
            checkpoint_specs=specs,
            output_csv=PACTRAN_OUTPUT_CSV,
            target_col=PACTRAN_TARGET_COL,
            train_days=ROLLING_TRAIN_DAYS,
            test_days=ROLLING_TEST_DAYS,
            step_days=ROLLING_STEP_DAYS,
            only_put=True,
            use_maturity_buckets=USE_MATURITY_BUCKETS,
            maturity_bins=MATURITY_BINS,
            maturity_labels=MATURITY_LABELS,
            sigma2=PACTRAN_SIGMA2,
            sigma_pi2=PACTRAN_SIGMA_PI2,
            prior_center=PACTRAN_PRIOR_CENTER,
            posterior_dir=PACTRAN_POSTERIOR_DIR,
            subsample_size=PACTRAN_SUBSAMPLE_SIZE,
            subsample_frac=PACTRAN_SUBSAMPLE_FRAC,
            n_subsamples=PACTRAN_N_SUBSAMPLES,
            subsample_seed=PACTRAN_SUBSAMPLE_SEED,
        )

    if finetune_results is None:
        finetune_results = pd.read_csv(FINETUNE_OUTPUT_DIR / "rolling_results.csv")
    if pactran_results is None:
        pactran_results = pd.read_csv(PACTRAN_OUTPUT_CSV)

    align_rolling_pactran_finetune(
        finetune_results=finetune_results,
        pactran_results=pactran_results,
        output_csv=ALIGNED_OUTPUT_CSV,
    )


def run_repeated_random_holdout_oos(heston_dataset, specs):
    run_repeated_random_holdout_experiments(
        heston_dataset=heston_dataset,
        checkpoint_specs=specs,
        output_root=EXPERIMENT_DIR / "repeated_random_holdout",
        bs_checkpoint_path=BS_PRETRAIN_CHECKPOINT_PATH,
        heston_checkpoint_path=HESTON_PRETRAIN_CHECKPOINT_PATH,
        seeds=SEEDS,
        train_size=HOLDOUT_TRAIN_SIZE,
        test_size=HOLDOUT_TEST_SIZE,
        bucket_type=HOLDOUT_BUCKET_TYPE,
        theories=RUN_THEORIES,
        methods=RUN_METHODS,
        val_ratio=FINETUNE_VAL_RATIO,
        tl_lr=FINETUNE_TL_LR,
        dl_lr=FINETUNE_DL_LR,
        max_epochs=FINETUNE_MAX_EPOCHS,
        early_stop_consecutive=FINETUNE_EARLY_STOP_CONSECUTIVE,
        run_finetune=RUN_FINETUNE,
        run_pactran_score=RUN_PACTRAN_SCORE,
        pactran_target_col=PACTRAN_TARGET_COL,
        pactran_sigma2=PACTRAN_SIGMA2,
        pactran_sigma_pi2=PACTRAN_SIGMA_PI2,
        pactran_prior_center=PACTRAN_PRIOR_CENTER,
        pactran_subsample_size=PACTRAN_SUBSAMPLE_SIZE,
        pactran_subsample_frac=PACTRAN_SUBSAMPLE_FRAC,
        pactran_n_subsamples=PACTRAN_N_SUBSAMPLES,
        pactran_subsample_seed=PACTRAN_SUBSAMPLE_SEED,
        maturity_bins=MATURITY_BINS,
        maturity_labels=MATURITY_LABELS,
    )


def main():
    save_config()
    heston_dataset = get_heston_dataset(
        heston_dataset_path=HESTON_DATASET_PATH,
        run_heston_calibration=RUN_HESTON_CALIBRATION,
    )
    vix = pd.read_csv('./data/VIX.csv')
    vix['Date'] = pd.to_datetime(vix['Date'])
    heston_dataset['date'] = pd.to_datetime(heston_dataset['date'])
    turb_dates = vix[vix.vix < 15].Date.unique().tolist()
    heston_dataset = heston_dataset[heston_dataset.date.isin(turb_dates)]
    # heston_dataset = heston_dataset[heston_dataset.tau > 14 / 365]
    print(heston_dataset.shape)

    if RUN_THEORY_PRETRAIN:
        run_all_theory_pretrains(
            bs_dataset_path=BS_THEORY_DATASET_PATH,
            heston_dataset_path=HESTON_THEORY_DATASET_PATH,
            bs_checkpoint_path=BS_PRETRAIN_CHECKPOINT_PATH,
            heston_checkpoint_path=HESTON_PRETRAIN_CHECKPOINT_PATH,
        )

    mode = str(OOS_MODE).lower()
    specs = checkpoint_specs()
    if RUN_REPEATED_RANDOM_HOLDOUT:
        run_repeated_random_holdout_oos(heston_dataset, specs)
    elif mode == "fixed":
        run_fixed_oos(heston_dataset, specs)
    elif mode == "rolling":
        run_rolling_oos(heston_dataset, specs)

if __name__ == "__main__":
    main()
