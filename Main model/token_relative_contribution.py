# -*- coding: utf-8 -*-
"""
Three-token relative contribution analysis
for the no-CLS + mean-pooling modal-token residual network.

Goal
----
Quantify the relative contribution of:
    1. Geom
    2. Mode0
    3. Mode1

to the final predictions of:
    eta_0
    eta_-1

Method
------
For each token, randomly shuffle that token across test samples while keeping
the other two tokens unchanged. Then measure how much the FINAL prediction
changes relative to the original prediction.

For eta_0:
    I_token = mean_i |eta0_full(i) - eta0_shuffle_token(i)|

For eta_-1:
    I_token = mean_i |etam1_full(i) - etam1_shuffle_token(i)|

Then normalize the three importance values for each output:

    C_token = I_token / (I_Geom + I_Mode0 + I_Mode1) * 100%

Thus, for each diffraction order:
    C_Geom + C_Mode0 + C_Mode1 = 100%

Important
---------
This is a MODEL-LEVEL relative contribution analysis. It reflects how strongly
the trained model prediction depends on each token. It should not be described
as a strict causal physical contribution to the real diffraction efficiency.

No retraining is performed.
"""

import importlib.util
import json
import os
import random
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# 1. Configuration
# ============================================================

ROOT = Path(__file__).resolve().parent

CSV_PATH = ROOT / "RCWA_TE_Littrow_LHS_10000.csv"

# Support two common project layouts.
MODEL_CANDIDATES = [
    ROOT / "best_modal_residual_net_10000.pth",
    ROOT / "proposed_method_results" / "models" / "best_modal_residual_net_10000.pth",
]

SCALER_CANDIDATES = [
    ROOT / "scalers_10000.pkl",
    ROOT / "proposed_method_results" / "models" / "scalers_10000.pkl",
]

SPLIT_CANDIDATES = [
    ROOT / "split_indices_10000.npz",
    ROOT / "proposed_method_results" / "split_indices_10000.npz",
]

MODEL_CODE_CANDIDATES = [
    ROOT / "proposed_method.py",
    ROOT.parent / "proposed_method.py",
]

OUTPUT_DIR = ROOT / "token_relative_contribution_results"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

SEED = 42
N_REPEATS = 50
BATCH_SIZE = 256

# Must match the trained mean-pooling model.
D_MODEL = 96
N_HEADS = 2
N_LAYERS = 2
DROPOUT = 0.08
MLP_HIDDEN = 64

TORCH_NUM_THREADS = min(8, os.cpu_count() or 1)

TOKEN_ORDER = ["Geom", "Mode0", "Mode1"]


# ============================================================
# 2. Utilities
# ============================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_dirs_and_style() -> None:
    for p in [OUTPUT_DIR, FIG_DIR, TABLE_DIR]:
        p.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "axes.unicode_minus": False,
        "axes.grid": False,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "savefig.dpi": 900,
        "svg.fonttype": "none",
    })


def save_fig(fig, name: str) -> None:
    fig.patch.set_facecolor("white")
    for ax in fig.axes:
        ax.set_facecolor("white")
        ax.grid(False)

    fig.savefig(
        FIG_DIR / f"{name}.png",
        dpi=900,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    fig.savefig(
        FIG_DIR / f"{name}.svg",
        format="svg",
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)


def first_existing(candidates, description: str) -> Path:
    for path in candidates:
        if path.exists():
            return path

    searched = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"找不到{description}。已检查以下路径：\n{searched}"
    )


def save_json(obj, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


# ============================================================
# 3. Data
# ============================================================

def load_dataframe() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"找不到数据集：{CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    if "status" in df.columns:
        df = df[df["status"] == "OK"].reset_index(drop=True)

    required_cols = [
        "Lambda_m", "f", "h_m", "lambda_m", "theta_deg",
        "neff0_TE", "sin_phi0_TE", "cos_phi0_TE",
        "neff1_TE", "sin_phi1_TE", "cos_phi1_TE",
        "eta0_SMM_TE", "etam1_SMM_TE",
        "eta0_RCWA_TE", "etam1_RCWA_TE",
        "res_eta0_TE", "res_etam1_TE",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少必要列：{missing}")

    return df


def build_features(df: pd.DataFrame) -> dict:
    mode = np.stack([
        df[["neff0_TE", "sin_phi0_TE", "cos_phi0_TE"]].values,
        df[["neff1_TE", "sin_phi1_TE", "cos_phi1_TE"]].values,
    ], axis=1).astype(np.float32)

    geom = np.stack([
        df["Lambda_m"].values / df["lambda_m"].values,
        df["f"].values,
        df["h_m"].values / df["lambda_m"].values,
        df["theta_deg"].values / 90.0,
    ], axis=1).astype(np.float32)

    smm = df[
        ["eta0_SMM_TE", "etam1_SMM_TE"]
    ].values.astype(np.float32)

    rcwa = df[
        ["eta0_RCWA_TE", "etam1_RCWA_TE"]
    ].values.astype(np.float32)

    return {
        "mode": mode,
        "geom": geom,
        "smm": smm,
        "rcwa": rcwa,
    }


def standardize_features(x: dict, scalers: dict) -> dict:
    mode = x["mode"]
    geom = x["geom"]

    n, t, d = mode.shape

    mode_std = scalers["mode"].transform(
        mode.reshape(-1, d)
    ).reshape(n, t, d).astype(np.float32)

    geom_std = scalers["geom"].transform(
        geom
    ).astype(np.float32)

    return {
        "mode": mode_std,
        "geom": geom_std,
    }


# ============================================================
# 4. Load main model code and trained assets
#    Reuse proposed_method.py directly to guarantee consistency
# ============================================================

def load_model_module():
    """Load the current main code proposed_method.py.

    The contribution script intentionally reuses the model class and
    feature-building function from the main code instead of maintaining
    a duplicate network definition. This prevents architecture drift.
    """
    for code_path in MODEL_CODE_CANDIDATES:
        if code_path.exists():
            spec = importlib.util.spec_from_file_location(
                "proposed_method_main",
                code_path,
            )
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            print("Model code:", code_path)
            return module

    searched = "\n".join(str(p) for p in MODEL_CODE_CANDIDATES)
    raise FileNotFoundError(
        "找不到主代码 proposed_method.py。已检查以下路径：\n" + searched
    )


def load_assets(device):
    pm = load_model_module()

    model_path = first_existing(
        MODEL_CANDIDATES,
        "模型文件 best_modal_residual_net_10000.pth"
    )

    scaler_path = first_existing(
        SCALER_CANDIDATES,
        "标准化器 scalers_10000.pkl"
    )

    split_path = first_existing(
        SPLIT_CANDIDATES,
        "数据划分 split_indices_10000.npz"
    )

    print("Model :", model_path)
    print("Scaler:", scaler_path)
    print("Split :", split_path)

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        config = checkpoint.get("config", {})
    else:
        state_dict = checkpoint
        config = {}

    # Use the hyperparameters saved with the trained model before
    # instantiating the class from proposed_method.py.
    for key in ["D_MODEL", "N_HEADS", "N_LAYERS", "DROPOUT", "MLP_HIDDEN"]:
        if key in config:
            setattr(pm, key, config[key])

    # Prevent accidental use of an old CLS-token checkpoint.
    if any("cls_token" in key for key in state_dict.keys()):
        raise RuntimeError(
            "当前 .pth 仍然是旧的 CLS Token 模型。\n"
            "请加载去掉 CLS、采用 mean pooling 后重新训练得到的模型。"
        )

    model = pm.ModalResidualNet().to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    scalers = joblib.load(scaler_path)
    missing_scalers = {"mode", "geom", "res"} - set(scalers)
    if missing_scalers:
        raise KeyError(f"标准化器缺少必要键：{sorted(missing_scalers)}")

    split_npz = np.load(split_path)

    return model, scalers, split_npz, pm


# ============================================================
# 6. Prediction
# ============================================================

@torch.no_grad()
def predict_final_efficiency(
    model,
    mode_std,
    geom_std,
    smm,
    res_scaler,
    device,
):
    """
    Return final predicted diffraction efficiencies:
        eta_pred = clip(eta_SMM + residual_NN, 0, 1)

    This matches the prediction logic in the original training code.
    """
    model.eval()

    pred_scaled_all = []

    for start in range(
        0,
        len(mode_std),
        BATCH_SIZE
    ):
        end = min(
            start + BATCH_SIZE,
            len(mode_std)
        )

        mode_t = torch.from_numpy(
            mode_std[start:end]
        ).to(device)

        geom_t = torch.from_numpy(
            geom_std[start:end]
        ).to(device)

        pred_scaled = model(
            mode_t,
            geom_t
        )

        pred_scaled_all.append(
            pred_scaled.cpu().numpy()
        )

    pred_scaled_all = np.vstack(
        pred_scaled_all
    )

    pred_res = res_scaler.inverse_transform(
        pred_scaled_all
    )

    pred_eff_raw = (
        smm + pred_res
    )

    pred_eff = np.clip(
        pred_eff_raw,
        0.0,
        1.0
    )

    return pred_eff


# ============================================================
# 7. Token shuffle
# ============================================================

def shuffle_one_token(
    mode_test,
    geom_test,
    token_name,
    rng,
):
    """
    Shuffle exactly ONE token across test samples.

    Geom:
        Shuffle the complete 4-dimensional geometry token as one unit.

    Mode0:
        Shuffle the complete 3-dimensional Mode0 token as one unit.

    Mode1:
        Shuffle the complete 3-dimensional Mode1 token as one unit.

    Other tokens remain unchanged.
    """
    mode_shuffled = mode_test.copy()
    geom_shuffled = geom_test.copy()

    n = len(geom_test)
    perm = rng.permutation(n)

    if token_name == "Geom":
        geom_shuffled[:, :] = (
            geom_test[perm, :]
        )

    elif token_name == "Mode0":
        mode_shuffled[:, 0, :] = (
            mode_test[perm, 0, :]
        )

    elif token_name == "Mode1":
        mode_shuffled[:, 1, :] = (
            mode_test[perm, 1, :]
        )

    else:
        raise ValueError(
            f"Unknown token: {token_name}"
        )

    return (
        mode_shuffled,
        geom_shuffled,
    )


# ============================================================
# 8. Relative contribution calculation
# ============================================================

def run_relative_contribution_analysis(
    model,
    mode_test,
    geom_test,
    smm_test,
    res_scaler,
    device,
):
    # --------------------------------------------------------
    # Original full-model prediction
    # --------------------------------------------------------
    full_pred = predict_final_efficiency(
        model=model,
        mode_std=mode_test,
        geom_std=geom_test,
        smm=smm_test,
        res_scaler=res_scaler,
        device=device,
    )

    records = []

    # --------------------------------------------------------
    # Repeated token shuffling
    # --------------------------------------------------------
    for token_idx, token_name in enumerate(TOKEN_ORDER):

        print(
            f"\nAnalyzing {token_name} "
            f"({N_REPEATS} repeated shuffles)..."
        )

        for repeat in range(N_REPEATS):

            rng = np.random.default_rng(
                SEED
                + token_idx * 10000
                + repeat
            )

            (
                mode_shuffled,
                geom_shuffled,
            ) = shuffle_one_token(
                mode_test=mode_test,
                geom_test=geom_test,
                token_name=token_name,
                rng=rng,
            )

            shuffled_pred = predict_final_efficiency(
                model=model,
                mode_std=mode_shuffled,
                geom_std=geom_shuffled,
                smm=smm_test,
                res_scaler=res_scaler,
                device=device,
            )

            # ------------------------------------------------
            # Importance = change in final model prediction
            # ------------------------------------------------
            abs_change = np.abs(
                shuffled_pred - full_pred
            )

            eta0_importance = float(
                abs_change[:, 0].mean()
            )

            etam1_importance = float(
                abs_change[:, 1].mean()
            )

            overall_importance = float(
                abs_change.mean()
            )

            records.append({
                "Token": token_name,
                "Repeat": repeat + 1,
                "Eta0_prediction_change": eta0_importance,
                "Etam1_prediction_change": etam1_importance,
                "Overall_prediction_change": overall_importance,
            })

    raw_df = pd.DataFrame(
        records
    )

    raw_df.to_csv(
        TABLE_DIR
        / "token_contribution_all_repeats.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Average raw importance over repeated shuffles
    # --------------------------------------------------------
    summary_rows = []

    for token_name in TOKEN_ORDER:

        sub = raw_df[
            raw_df["Token"] == token_name
        ]

        summary_rows.append({
            "Token": token_name,

            "Eta0_importance_mean":
                float(
                    sub[
                        "Eta0_prediction_change"
                    ].mean()
                ),

            "Eta0_importance_std":
                float(
                    sub[
                        "Eta0_prediction_change"
                    ].std(ddof=1)
                ),

            "Etam1_importance_mean":
                float(
                    sub[
                        "Etam1_prediction_change"
                    ].mean()
                ),

            "Etam1_importance_std":
                float(
                    sub[
                        "Etam1_prediction_change"
                    ].std(ddof=1)
                ),

            "Overall_importance_mean":
                float(
                    sub[
                        "Overall_prediction_change"
                    ].mean()
                ),

            "Overall_importance_std":
                float(
                    sub[
                        "Overall_prediction_change"
                    ].std(ddof=1)
                ),
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    # --------------------------------------------------------
    # Normalize to 100% separately for eta0 and eta-1
    # --------------------------------------------------------
    eta0_total = summary_df[
        "Eta0_importance_mean"
    ].sum()

    etam1_total = summary_df[
        "Etam1_importance_mean"
    ].sum()

    overall_total = summary_df[
        "Overall_importance_mean"
    ].sum()

    if eta0_total <= 0:
        raise RuntimeError(
            "eta0 importance total is zero."
        )

    if etam1_total <= 0:
        raise RuntimeError(
            "eta-1 importance total is zero."
        )

    summary_df[
        "Eta0_relative_contribution_percent"
    ] = (
        summary_df[
            "Eta0_importance_mean"
        ]
        / eta0_total
        * 100.0
    )

    summary_df[
        "Etam1_relative_contribution_percent"
    ] = (
        summary_df[
            "Etam1_importance_mean"
        ]
        / etam1_total
        * 100.0
    )

    summary_df[
        "Overall_relative_contribution_percent"
    ] = (
        summary_df[
            "Overall_importance_mean"
        ]
        / overall_total
        * 100.0
    )

    summary_df.to_csv(
        TABLE_DIR
        / "token_relative_contribution_summary.csv",
        index=False,
    )

    return (
        full_pred,
        raw_df,
        summary_df,
    )


# ============================================================
# 9. Figures
# ============================================================

def plot_two_output_relative_contribution(
    summary_df,
):
    """
    Main paper figure:
    contribution of Geom / Mode0 / Mode1
    to eta0 and eta-1.
    """
    token_names = summary_df[
        "Token"
    ].tolist()

    eta0_contrib = summary_df[
        "Eta0_relative_contribution_percent"
    ].values

    etam1_contrib = summary_df[
        "Etam1_relative_contribution_percent"
    ].values

    x = np.arange(
        len(token_names)
    )

    width = 0.34

    fig, ax = plt.subplots(
        figsize=(7.6, 5.2)
    )

    bars0 = ax.bar(
        x - width / 2,
        eta0_contrib,
        width,
        label=r"$\eta_0$",
    )

    bars1 = ax.bar(
        x + width / 2,
        etam1_contrib,
        width,
        label=r"$\eta_{-1}$",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        token_names
    )

    ax.set_ylabel(
        "Relative contribution (%)"
    )

    ax.set_title(
        "Relative Contribution of Physical Tokens"
    )

    ax.legend()

    max_value = max(
        eta0_contrib.max(),
        etam1_contrib.max(),
    )

    ax.set_ylim(
        0,
        max_value * 1.22
    )

    for bar in bars0:
        h = bar.get_height()
        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            h + max_value * 0.025,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    for bar in bars1:
        h = bar.get_height()
        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            h + max_value * 0.025,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()

    save_fig(
        fig,
        "token_relative_contribution_two_outputs"
    )


def plot_eta0_contribution(
    summary_df,
):
    token_names = summary_df[
        "Token"
    ].tolist()

    values = summary_df[
        "Eta0_relative_contribution_percent"
    ].values

    fig, ax = plt.subplots(
        figsize=(6.6, 4.8)
    )

    bars = ax.bar(
        token_names,
        values,
    )

    ax.set_ylabel(
        "Relative contribution (%)"
    )

    ax.set_title(
        r"Token Contribution to $\eta_0$ Prediction"
    )

    ax.set_ylim(
        0,
        values.max() * 1.22
    )

    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            h + values.max() * 0.025,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()

    save_fig(
        fig,
        "token_relative_contribution_eta0"
    )


def plot_etam1_contribution(
    summary_df,
):
    token_names = summary_df[
        "Token"
    ].tolist()

    values = summary_df[
        "Etam1_relative_contribution_percent"
    ].values

    fig, ax = plt.subplots(
        figsize=(6.6, 4.8)
    )

    bars = ax.bar(
        token_names,
        values,
    )

    ax.set_ylabel(
        "Relative contribution (%)"
    )

    ax.set_title(
        r"Token Contribution to $\eta_{-1}$ Prediction"
    )

    ax.set_ylim(
        0,
        values.max() * 1.22
    )

    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            h + values.max() * 0.025,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()

    save_fig(
        fig,
        "token_relative_contribution_etam1"
    )


# ============================================================
# 10. Main
# ============================================================

def main():
    set_seed(
        SEED
    )

    setup_dirs_and_style()

    torch.set_num_threads(
        TORCH_NUM_THREADS
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 80)
    print("Three-token Relative Contribution Analysis")
    print("Device:", device)
    print("Repeated shuffles:", N_REPEATS)
    print("=" * 80)

    df = load_dataframe()

    model, scalers, split_npz, pm = load_assets(
        device
    )

    # Use exactly the same feature construction as the main code.
    x = pm.build_features(
        df
    )

    if "test_indices" not in split_npz.files:
        raise KeyError(
            "split_indices_10000.npz 中找不到 test_indices。"
        )

    test_idx = split_npz[
        "test_indices"
    ]

    xs = standardize_features(
        x,
        scalers,
    )

    mode_test = xs[
        "mode"
    ][test_idx]

    geom_test = xs[
        "geom"
    ][test_idx]

    smm_test = x[
        "smm"
    ][test_idx]

    print(
        f"Test samples: {len(test_idx)}"
    )

    (
        full_pred,
        raw_df,
        summary_df,
    ) = run_relative_contribution_analysis(
        model=model,
        mode_test=mode_test,
        geom_test=geom_test,
        smm_test=smm_test,
        res_scaler=scalers["res"],
        device=device,
    )

    # Main figure
    plot_two_output_relative_contribution(
        summary_df
    )

    # Separate figures
    plot_eta0_contribution(
        summary_df
    )

    plot_etam1_contribution(
        summary_df
    )

    # Save a concise JSON summary
    result_dict = {}

    for _, row in summary_df.iterrows():
        token = row["Token"]

        result_dict[token] = {
            "eta0_relative_contribution_percent":
                float(
                    row[
                        "Eta0_relative_contribution_percent"
                    ]
                ),

            "etam1_relative_contribution_percent":
                float(
                    row[
                        "Etam1_relative_contribution_percent"
                    ]
                ),

            "overall_relative_contribution_percent":
                float(
                    row[
                        "Overall_relative_contribution_percent"
                    ]
                ),
        }

    save_json(
        result_dict,
        OUTPUT_DIR
        / "token_relative_contribution.json",
    )

    print("\n" + "=" * 80)
    print("Relative contribution results")
    print("=" * 80)

    display_df = summary_df[
        [
            "Token",
            "Eta0_relative_contribution_percent",
            "Etam1_relative_contribution_percent",
            "Overall_relative_contribution_percent",
        ]
    ].copy()

    print(
        display_df.to_string(
            index=False,
            formatters={
                "Eta0_relative_contribution_percent":
                    lambda x: f"{x:.2f}%",
                "Etam1_relative_contribution_percent":
                    lambda x: f"{x:.2f}%",
                "Overall_relative_contribution_percent":
                    lambda x: f"{x:.2f}%",
            }
        )
    )

    print("\nCheck:")
    print(
        "eta0 total =",
        f"{summary_df['Eta0_relative_contribution_percent'].sum():.2f}%"
    )
    print(
        "eta-1 total =",
        f"{summary_df['Etam1_relative_contribution_percent'].sum():.2f}%"
    )

    print("\nAnalysis completed.")
    print(
        "Results folder:",
        OUTPUT_DIR.resolve()
    )

    print("\nKey outputs:")
    print(
        "  1. tables/token_relative_contribution_summary.csv"
    )
    print(
        "  2. tables/token_contribution_all_repeats.csv"
    )
    print(
        "  3. figures/token_relative_contribution_two_outputs.png/.svg"
    )
    print(
        "  4. figures/token_relative_contribution_eta0.png/.svg"
    )
    print(
        "  5. figures/token_relative_contribution_etam1.png/.svg"
    )
    print(
        "  6. token_relative_contribution.json"
    )


if __name__ == "__main__":
    main()
