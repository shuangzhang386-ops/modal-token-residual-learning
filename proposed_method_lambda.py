import json
import os
import random
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# 1. Configuration
# ============================================================

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "RCWA_TE_Littrow_LHS_10000.csv"
OUTPUT_DIR = ROOT / "proposed_method_lambda_results"
FIG_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"

SEED = 42
SPLIT_MODE = "lambda_block"          # random / h_block / lambda_block / f_block
BLOCK_TEST_SIDE = "high"       # high / low, only used for block split
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1

BATCH_SIZE = 256
MAX_EPOCHS = 1000
PATIENCE = 100
LR = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0

D_MODEL = 96
N_HEADS = 2
N_LAYERS = 2
DROPOUT = 0.08
MLP_HIDDEN = 64

NUM_WORKERS = 0
TORCH_NUM_THREADS = min(8, os.cpu_count() or 1)
REL_ERR_MIN_DENOM = 0.01

REQUIRED_COLS = [
    "Lambda_m", "f", "h_m", "lambda_m", "theta_deg",
    "neff0_TE", "sin_phi0_TE", "cos_phi0_TE",
    "neff1_TE", "sin_phi1_TE", "cos_phi1_TE",
    "eta0_SMM_TE", "etam1_SMM_TE",
    "eta0_RCWA_TE", "etam1_RCWA_TE",
    "res_eta0_TE", "res_etam1_TE",
]

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
    for p in [OUTPUT_DIR, FIG_DIR, MODEL_DIR]:
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
        "figure.dpi": 150,
        "savefig.dpi": 900,
        "svg.fonttype": "none",
        "text.antialiased": True,
    })


def to_device(batch, device):
    return [x.to(device, non_blocking=True) if torch.is_tensor(x) else x for x in batch]


def save_json(obj, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def add_bottom_panel_label(ax, label: str) -> None:
    ax.text(0.5, -0.115, label, transform=ax.transAxes,
            ha="center", va="top", fontsize=13, fontweight="bold", clip_on=False)


def save_paper_fig(fig, name: str) -> None:
    fig.patch.set_facecolor("white")
    for ax in fig.axes:
        ax.set_facecolor("white")
        ax.grid(False)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=900, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(FIG_DIR / f"{name}.svg", format="svg", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def metric_dict(y_true, y_pred, rel_min: float = REL_ERR_MIN_DENOM) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    abs_err = np.abs(y_pred - y_true)
    mask = np.abs(y_true) >= rel_min
    rel_err = float(np.mean(abs_err[mask] / np.abs(y_true[mask])) * 100) if np.any(mask) else None
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "MaxAbsError": float(np.max(abs_err)),
        f"MeanRelativeErrorPercent_absTrue_ge_{rel_min}": rel_err,
    }


def success_rate(y_true, y_pred, threshold: float) -> float:
    return float(np.mean(np.all(np.abs(y_pred - y_true) < threshold, axis=1)) * 100)

# ============================================================
# 3. Data
# ============================================================

def load_dataframe() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"找不到 CSV 文件：{CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    if "status" in df.columns:
        df = df[df["status"] == "OK"].reset_index(drop=True)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 文件缺少必要列：{missing}")
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

    return {
        "mode": mode,
        "geom": geom,
        "smm": df[["eta0_SMM_TE", "etam1_SMM_TE"]].values.astype(np.float32),
        "rcwa": df[["eta0_RCWA_TE", "etam1_RCWA_TE"]].values.astype(np.float32),
        "res": df[["res_eta0_TE", "res_etam1_TE"]].values.astype(np.float32),
    }


def make_split(df: pd.DataFrame):
    idx = np.arange(len(df))
    if SPLIT_MODE == "random":
        train_idx, temp_idx = train_test_split(idx, test_size=1 - TRAIN_RATIO, random_state=SEED, shuffle=True)
        val_ratio_in_temp = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
        val_idx, test_idx = train_test_split(temp_idx, test_size=1 - val_ratio_in_temp, random_state=SEED, shuffle=True)
        return train_idx, val_idx, test_idx, {"split_mode": "random", "description": "随机划分，主要测试插值能力。"}

    if SPLIT_MODE == "h_block":
        value, name = df["h_m"].values / df["lambda_m"].values, "h/lambda"
    elif SPLIT_MODE == "lambda_block":
        value, name = df["Lambda_m"].values / df["lambda_m"].values, "Lambda/lambda"
    elif SPLIT_MODE == "f_block":
        value, name = df["f"].values, "f"
    else:
        raise ValueError(f"未知 SPLIT_MODE: {SPLIT_MODE}")

    if BLOCK_TEST_SIDE == "high":
        th = np.quantile(value, 1 - TEST_RATIO)
        test_idx, train_val_idx = idx[value >= th], idx[value < th]
        train_region, test_region = f"{name} < {th:.6f}", f"{name} >= {th:.6f}"
    elif BLOCK_TEST_SIDE == "low":
        th = np.quantile(value, TEST_RATIO)
        test_idx, train_val_idx = idx[value <= th], idx[value > th]
        train_region, test_region = f"{name} > {th:.6f}", f"{name} <= {th:.6f}"
    else:
        raise ValueError("BLOCK_TEST_SIDE 必须是 'high' 或 'low'")

    val_size = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=val_size, random_state=SEED, shuffle=True)
    info = {
        "split_mode": SPLIT_MODE,
        "split_parameter": name,
        "block_test_side": BLOCK_TEST_SIDE,
        "threshold": float(th),
        "train_region": train_region,
        "test_region": test_region,
        "description": "物理区间划分，用于测试模型外推能力。",
    }
    return train_idx, val_idx, test_idx, info


def fit_transform_features(x: dict, train_idx: np.ndarray):
    scalers = {
        "mode": StandardScaler(),
        "geom": StandardScaler(),
        "res": StandardScaler(),
    }

    n, t, d = x["mode"].shape
    scalers["mode"].fit(x["mode"][train_idx].reshape(-1, d))
    scalers["geom"].fit(x["geom"][train_idx])
    scalers["res"].fit(x["res"][train_idx])

    xs = {
        "mode": scalers["mode"].transform(x["mode"].reshape(-1, d)).reshape(n, t, d).astype(np.float32),
        "geom": scalers["geom"].transform(x["geom"]).astype(np.float32),
        "res_scaled": scalers["res"].transform(x["res"]).astype(np.float32),
    }
    return xs, scalers


def make_loader(xs: dict, x: dict, indices: np.ndarray, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(xs["mode"][indices]),
        torch.from_numpy(xs["geom"][indices]),
        torch.from_numpy(xs["res_scaled"][indices]),
        torch.from_numpy(x["smm"][indices]),
        torch.from_numpy(x["rcwa"][indices]),
        torch.from_numpy(x["res"][indices]),
        torch.from_numpy(indices.astype(np.int64)),
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=NUM_WORKERS,
                      pin_memory=torch.cuda.is_available())

# ============================================================
# 4. Model
# ============================================================

class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.ffn(self.norm2(x))


class ModalResidualNet(nn.Module):
    """
    Clean modal-token residual network.

    Network input tokens:
        [CLS, Geometry, Mode0, Mode1]

    Not used as network input:
        delta modal-difference features
        SMM diffraction efficiencies
        overlap integrals

    Final prediction:
        eta_pred = eta_SMM + residual_NN
    """
    def __init__(self):
        super().__init__()
        self.mode_embed = nn.Sequential(nn.Linear(3, D_MODEL), nn.LayerNorm(D_MODEL), nn.GELU())
        self.geom_embed = nn.Sequential(nn.Linear(4, D_MODEL), nn.LayerNorm(D_MODEL), nn.GELU())

        self.cls_token = nn.Parameter(torch.zeros(1, 1, D_MODEL))
        self.type_embedding = nn.Embedding(4, D_MODEL)
        self.input_dropout = nn.Dropout(DROPOUT)
        self.encoder = nn.ModuleList([EncoderBlock(D_MODEL, N_HEADS, DROPOUT) for _ in range(N_LAYERS)])

        # Lightweight MLP regression head: CLS_out(96) -> 64 -> 2.
        # No residual connection is used in this head.
        self.head = nn.Sequential(
            nn.LayerNorm(D_MODEL),
            nn.Linear(D_MODEL, MLP_HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(MLP_HIDDEN, 2),
        )
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, mode, geom):
        b = mode.shape[0]
        tokens = torch.cat([
            self.cls_token.expand(b, -1, -1),
            self.geom_embed(geom).unsqueeze(1),
            self.mode_embed(mode[:, 0]).unsqueeze(1),
            self.mode_embed(mode[:, 1]).unsqueeze(1),
        ], dim=1)

        type_ids = torch.arange(4, device=tokens.device).unsqueeze(0).expand(b, -1)
        z = self.input_dropout(tokens + self.type_embedding(type_ids))
        for block in self.encoder:
            z = block(z)

        cls_out = z[:, 0]
        return self.head(cls_out)

# ============================================================
# 5. Training and prediction
# ============================================================

def unpack(batch, device):
    mode, geom, target, smm, rcwa, true_res, idx = to_device(batch, device)
    return mode, geom, target, smm, rcwa, true_res, idx


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total = 0.0

    for batch in loader:
        mode, geom, target, *_ = unpack(batch, device)
        with torch.set_grad_enabled(is_train):
            pred = model(mode, geom)
            loss = loss_fn(pred, target)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()
        total += loss.item() * mode.shape[0]
    return total / len(loader.dataset)


def train_model(model, train_loader, val_loader, device):
    loss_fn = nn.SmoothL1Loss(beta=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-6)

    best_state, best_val, best_epoch, wait = None, float("inf"), 0, 0
    history = []
    t0 = time.time()
    print("\n开始训练...\n")

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss = run_epoch(model, val_loader, loss_fn, device)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr})

        if val_loss < best_val - 1e-7:
            best_val, best_epoch, wait = val_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1

        if epoch == 1 or epoch % 20 == 0:
            print(f"Epoch {epoch:04d} | train={train_loss:.6f} | val={val_loss:.6f} | lr={lr:.2e}")
        if wait >= PATIENCE:
            print(f"\n早停：连续 {PATIENCE} 轮验证集没有提升。")
            break

    if best_state is None:
        raise RuntimeError("训练失败：没有保存到最佳模型。")

    model.load_state_dict(best_state)
    print(f"\n最佳 epoch = {best_epoch}, best val loss = {best_val:.6f}")
    print(f"训练耗时 = {time.time() - t0:.1f} s")
    return history, best_epoch, best_val


@torch.no_grad()
def predict(model, loader, res_scaler, device):
    model.eval()
    pred_s, true_s, smm_all, rcwa_all, res_all, idx_all = [], [], [], [], [], []
    for batch in loader:
        mode, geom, target, smm, rcwa, true_res, idx = unpack(batch, device)
        pred_s.append(model(mode, geom).cpu().numpy())
        true_s.append(target.cpu().numpy())
        smm_all.append(smm.cpu().numpy())
        rcwa_all.append(rcwa.cpu().numpy())
        res_all.append(true_res.cpu().numpy())
        idx_all.append(idx.cpu().numpy())

    pred_res = res_scaler.inverse_transform(np.vstack(pred_s))
    true_res = res_scaler.inverse_transform(np.vstack(true_s))
    smm = np.vstack(smm_all)
    rcwa = np.vstack(rcwa_all)
    pred_eff_raw = smm + pred_res
    pred_eff = np.clip(pred_eff_raw, 0.0, 1.0)
    return {
        "pred_res": pred_res,
        "true_res": true_res,
        "true_res_direct": np.vstack(res_all),
        "smm": smm,
        "rcwa": rcwa,
        "pred_eff_raw": pred_eff_raw,
        "pred_eff": pred_eff,
        "idx": np.concatenate(idx_all),
    }

# ============================================================
# 6. Save results and figures
# ============================================================

def build_summary(df, train_idx, val_idx, test_idx, split_info, history, best_epoch, best_val, metrics):
    return {
        "dataset": {
            "csv_path": str(CSV_PATH),
            "valid_samples": int(len(df)),
            "train_samples": int(len(train_idx)),
            "val_samples": int(len(val_idx)),
            "test_samples": int(len(test_idx)),
            "use_overlap_as_input": False,
            "use_delta_as_network_input": False,
            "use_smm_efficiency_as_network_input": False,
        },
        "split_info": split_info,
        "training": {
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val),
            "epochs_ran": int(len(history)),
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "learning_rate_initial": LR,
            "weight_decay": WEIGHT_DECAY,
        },
        "model_config": {
            "D_MODEL": D_MODEL,
            "N_HEADS": N_HEADS,
            "N_LAYERS": N_LAYERS,
            "DROPOUT": DROPOUT,
            "MLP_HIDDEN": MLP_HIDDEN,
            "HEAD_TYPE": "light_mlp_96_to_64_to_2",
            "USE_DELTA_AS_NETWORK_INPUT": False,
            "USE_SMM_EFFICIENCY_AS_NETWORK_INPUT": False,
        },
        "metrics": metrics,
    }


def save_model_and_scalers(model, scalers, split_info):
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "D_MODEL": D_MODEL,
            "N_HEADS": N_HEADS,
            "N_LAYERS": N_LAYERS,
            "DROPOUT": DROPOUT,
            "MLP_HIDDEN": MLP_HIDDEN,
            "HEAD_TYPE": "light_mlp_96_to_64_to_2",
            "SPLIT_MODE": SPLIT_MODE,
            "BLOCK_TEST_SIDE": BLOCK_TEST_SIDE,
            "TRAIN_RATIO": TRAIN_RATIO,
            "VAL_RATIO": VAL_RATIO,
            "TEST_RATIO": TEST_RATIO,
            "SEED": SEED,
            "USE_OVERLAP_AS_INPUT": False,
            "USE_DELTA_AS_NETWORK_INPUT": False,
            "USE_SMM_EFFICIENCY_AS_NETWORK_INPUT": False,
        },
        "split_info": split_info,
    }, MODEL_DIR / "best_modal_residual_net_10000.pth")
    joblib.dump(scalers, MODEL_DIR / "scalers_10000.pkl")


def save_prediction_csv(df, test):
    idx = test["idx"]
    pred_df = df.iloc[idx].copy()
    pred_df.insert(0, "original_index", idx)

    rcwa, nn_pred, pred_res = test["rcwa"], test["pred_eff"], test["pred_res"]
    pred_df["pred_res_eta0_TE"] = pred_res[:, 0]
    pred_df["pred_res_etam1_TE"] = pred_res[:, 1]
    pred_df["pred_eta0_TE"] = nn_pred[:, 0]
    pred_df["pred_etam1_TE"] = nn_pred[:, 1]
    pred_df["pred_eta0_raw_TE"] = test["pred_eff_raw"][:, 0]
    pred_df["pred_etam1_raw_TE"] = test["pred_eff_raw"][:, 1]
    pred_df["err_eta0_TE"] = nn_pred[:, 0] - rcwa[:, 0]
    pred_df["err_etam1_TE"] = nn_pred[:, 1] - rcwa[:, 1]
    pred_df["abs_err_eta0_TE"] = np.abs(pred_df["err_eta0_TE"].values)
    pred_df["abs_err_etam1_TE"] = np.abs(pred_df["err_etam1_TE"].values)
    pred_df["abs_err_max_two_channels_TE"] = np.maximum(pred_df["abs_err_eta0_TE"].values, pred_df["abs_err_etam1_TE"].values)
    pred_df.to_csv(OUTPUT_DIR / "test_predictions_10000.csv", index=False)


def plot_loss(history):
    h = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(h["epoch"], h["train_loss"], label="Train")
    ax.plot(h["epoch"], h["val_loss"], label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("SmoothL1 Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    fig.tight_layout()
    save_paper_fig(fig, "loss_curve_10000")


def plot_efficiency(rcwa_true, nn_pred):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.95))
    for ax, col, title, panel in zip(
        axes, [0, 1], [r"$\eta_{0}$ Efficiency Prediction", r"$\eta_{-1}$ Efficiency Prediction"], ["(a)", "(b)"]
    ):
        ax.scatter(rcwa_true[:, col], nn_pred[:, col], s=16, color="C0", alpha=0.70, label="Predicted samples", zorder=2)
        ax.plot([0, 1], [0, 1], color="red", linestyle="--", linewidth=2.4, label="Ideal prediction", zorder=3)
        ax.set_xlabel("True RCWA efficiency")
        ax.set_ylabel("Predicted efficiency: SMM + NN residual")
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(frameon=True, loc="upper left")
        add_bottom_panel_label(ax, panel)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.13, wspace=0.10)
    save_paper_fig(fig, "efficiency_prediction_combined_ab_10000")


def plot_residual_error(true_res, pred_res):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.25))
    for ax, col, title, panel in zip(
        axes, [0, 1], [r"$\eta_{0}$ residual error", r"$\eta_{-1}$ residual error"], ["(a)", "(b)"]
    ):
        true_r = true_res[:, col]
        err = pred_res[:, col] - true_r
        x_min, x_max = true_r.min(), true_r.max()
        x_pad = 0.04 * (x_max - x_min) if x_max > x_min else 0.01
        ax.axhspan(-0.01, 0.01, color="C0", alpha=0.12, label=r"$\pm 0.01$ band", zorder=0)
        ax.scatter(true_r, err, s=14, color="C0", alpha=0.65, label="Predicted samples", zorder=2)
        ax.axhline(0, color="red", linestyle="--", linewidth=2.4, label="Zero error", zorder=3)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_xlabel("True residual: RCWA - SMM")
        ax.set_ylabel("Residual prediction error")
        ax.set_title(title)
        ax.legend(frameon=True, loc="upper right")
        add_bottom_panel_label(ax, panel)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.14, wspace=0.12)
    save_paper_fig(fig, "residual_error_combined_ab_10000")


def plot_residual_prediction(true_res, pred_res):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.25))
    for ax, col, title, panel in zip(
        axes, [0, 1], [r"$\eta_{0}$ residual prediction", r"$\eta_{-1}$ residual prediction"], ["(a)", "(b)"]
    ):
        true_r, pred_r = true_res[:, col], pred_res[:, col]
        min_v, max_v = min(true_r.min(), pred_r.min()), max(true_r.max(), pred_r.max())
        pad = 0.04 * (max_v - min_v) if max_v > min_v else 0.01
        ax.scatter(true_r, pred_r, s=14, color="C0", alpha=0.65, label="Predicted samples", zorder=2)
        ax.plot([min_v, max_v], [min_v, max_v], color="red", linestyle="--", linewidth=2.4, label="Ideal prediction", zorder=3)
        ax.set_xlim(min_v - pad, max_v + pad)
        ax.set_ylim(min_v - pad, max_v + pad)
        ax.set_xlabel("True residual: RCWA - SMM")
        ax.set_ylabel("Predicted residual")
        ax.set_title(title)
        ax.legend(frameon=True, loc="upper left")
        add_bottom_panel_label(ax, panel)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.14, wspace=0.12)
    save_paper_fig(fig, "residual_prediction_combined_ab_10000")

# ============================================================
# 7. Main
# ============================================================

def main():
    set_seed(SEED)
    setup_dirs_and_style()
    torch.set_num_threads(TORCH_NUM_THREADS)
    try:
        torch.set_num_interop_threads(max(1, min(2, TORCH_NUM_THREADS // 2)))
    except RuntimeError:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("SMM-guided Modal Token Residual Network - 10000 samples")
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("CSV_PATH:", CSV_PATH)
    print("OUTPUT_DIR:", OUTPUT_DIR)
    print("SPLIT_MODE:", SPLIT_MODE)
    print("=" * 80)

    df = load_dataframe()
    x = build_features(df)
    train_idx, val_idx, test_idx, split_info = make_split(df)
    xs, scalers = fit_transform_features(x, train_idx)

    np.savez(OUTPUT_DIR / "split_indices_10000.npz", train_indices=train_idx, val_indices=val_idx, test_indices=test_idx)
    train_loader = make_loader(xs, x, train_idx, shuffle=True)
    val_loader = make_loader(xs, x, val_idx, shuffle=False)
    test_loader = make_loader(xs, x, test_idx, shuffle=False)

    print("\n数据划分：")
    print(json.dumps(split_info, indent=4, ensure_ascii=False))
    print(f"训练集: {len(train_idx)} | 验证集: {len(val_idx)} | 测试集: {len(test_idx)}")

    model = ModalResidualNet().to(device)
    print("\n模型参数量:", sum(p.numel() for p in model.parameters()))

    history, best_epoch, best_val = train_model(model, train_loader, val_loader, device)
    pd.DataFrame(history).to_csv(OUTPUT_DIR / "training_history_10000.csv", index=False)

    test = predict(model, test_loader, scalers["res"], device)
    rcwa_true, smm_pred, nn_pred = test["rcwa"], test["smm"], test["pred_eff"]
    true_res, pred_res = test["true_res"], test["pred_res"]

    smm_m = metric_dict(rcwa_true, smm_pred)
    nn_m = metric_dict(rcwa_true, nn_pred)
    improve = (smm_m["MAE"] - nn_m["MAE"]) / smm_m["MAE"] * 100

    metrics = {
        "SMM_overall": smm_m,
        "NN_corrected_overall": nn_m,
        "NN_eta0_efficiency": metric_dict(rcwa_true[:, 0], nn_pred[:, 0]),
        "NN_etam1_efficiency": metric_dict(rcwa_true[:, 1], nn_pred[:, 1]),
        "Residual_eta0": metric_dict(true_res[:, 0], pred_res[:, 0]),
        "Residual_etam1": metric_dict(true_res[:, 1], pred_res[:, 1]),
        "MAE_improvement_percent": float(improve),
        "success_rate_all_channels_error_lt_0.01_percent": success_rate(rcwa_true, nn_pred, 0.01),
        "NN_raw_unclipped_overall": metric_dict(rcwa_true, test["pred_eff_raw"]),
    }

    save_json(build_summary(df, train_idx, val_idx, test_idx, split_info, history, best_epoch, best_val, metrics),
              OUTPUT_DIR / "summary_metrics_10000.json")
    save_model_and_scalers(model, scalers, split_info)
    save_prediction_csv(df, test)
    plot_loss(history)
    plot_efficiency(rcwa_true, nn_pred)
    plot_residual_error(true_res, pred_res)
    plot_residual_prediction(true_res, pred_res)

    print("\n" + "=" * 80)
    print("核心测试结果")
    print("=" * 80)
    print(f"划分方式: {SPLIT_MODE}")
    if SPLIT_MODE != "random":
        print(f"测试区域: {split_info['test_region']}")
    print(f"SMM MAE:       {smm_m['MAE']:.6f}")
    print(f"NN 修正 MAE:  {nn_m['MAE']:.6f}")
    print(f"MAE 降低比例: {improve:.2f}%")
    print(f"NN R2:        {nn_m['R2']:.6f}")
    print(f"NN MaxError:  {nn_m['MaxAbsError']:.6f}")
    print(f"成功率 |error|<0.01: {metrics['success_rate_all_channels_error_lt_0.01_percent']:.2f}%")
    print("=" * 80)

    print("\n训练、评估与绘图全部完成。")
    print("结果文件夹:", OUTPUT_DIR.resolve())
    print("关键输出：")
    print("  1. summary_metrics_10000.json")
    print("  2. training_history_10000.csv")
    print("  3. test_predictions_10000.csv")
    print("  4. split_indices_10000.npz")
    print("  5. figures/*.png 和 figures/*.svg")
    print("  6. models/best_modal_residual_net_10000.pth")
    print("  7. models/scalers_10000.pkl")


if __name__ == "__main__":
    main()
