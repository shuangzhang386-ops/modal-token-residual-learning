import json
import os
import random
import time
from pathlib import Path

import joblib
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

OUTPUT_DIR = ROOT / "mlp_baseline_lambda_results"
MODEL_DIR = OUTPUT_DIR / "models"

REUSE_MAIN_SPLIT_IF_AVAILABLE = True
MAIN_SPLIT_PATH = ROOT / "10000proposed_method_results" / "split_indices_10000.npz"

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

MLP_HIDDEN_DIMS = [64, 128, 64]
DROPOUT = 0.08
USE_BATCH_NORM = False

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


def setup_dirs() -> None:
    for p in [OUTPUT_DIR, MODEL_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def to_device(batch, device):
    return [x.to(device, non_blocking=True) if torch.is_tensor(x) else x for x in batch]


def save_json(obj, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


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
    geom = np.stack([
        df["Lambda_m"].values / df["lambda_m"].values,
        df["f"].values,
        df["h_m"].values / df["lambda_m"].values,
        df["theta_deg"].values / 90.0,
    ], axis=1).astype(np.float32)

    mode = np.stack([
        df[["neff0_TE", "sin_phi0_TE", "cos_phi0_TE"]].values,
        df[["neff1_TE", "sin_phi1_TE", "cos_phi1_TE"]].values,
    ], axis=1).astype(np.float32)

    # 展平后的 MLP 输入，共 10 维。
    # 注意：这里不包含 eta0_SMM_TE / etam1_SMM_TE，不让 MLP 直接看到 SMM 效率。
    feat = np.concatenate([geom, mode.reshape(len(df), -1)], axis=1).astype(np.float32)

    return {
        "feat": feat,
        "geom": geom,
        "mode": mode,
        "smm": df[["eta0_SMM_TE", "etam1_SMM_TE"]].values.astype(np.float32),
        "rcwa": df[["eta0_RCWA_TE", "etam1_RCWA_TE"]].values.astype(np.float32),
        "res": df[["res_eta0_TE", "res_etam1_TE"]].values.astype(np.float32),
    }


def make_split(df: pd.DataFrame):
    idx = np.arange(len(df))

    if REUSE_MAIN_SPLIT_IF_AVAILABLE and MAIN_SPLIT_PATH.exists():
        split = np.load(MAIN_SPLIT_PATH)
        train_idx = split["train_indices"].astype(np.int64)
        val_idx = split["val_indices"].astype(np.int64)
        test_idx = split["test_indices"].astype(np.int64)
        info = {
            "split_mode": "reuse_main_split",
            "source": str(MAIN_SPLIT_PATH),
            "description": "复用主模型 split_indices.npz，保证 MLP baseline 与主模型测试样本完全一致。",
        }
        return train_idx, val_idx, test_idx, info

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
        "feat": StandardScaler(),
        "res": StandardScaler(),
    }
    scalers["feat"].fit(x["feat"][train_idx])
    scalers["res"].fit(x["res"][train_idx])

    xs = {
        "feat": scalers["feat"].transform(x["feat"]).astype(np.float32),
        "res_scaled": scalers["res"].transform(x["res"]).astype(np.float32),
    }
    return xs, scalers


def make_loader(xs: dict, x: dict, indices: np.ndarray, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(xs["feat"][indices]),
        torch.from_numpy(xs["res_scaled"][indices]),
        torch.from_numpy(x["smm"][indices]),
        torch.from_numpy(x["rcwa"][indices]),
        torch.from_numpy(x["res"][indices]),
        torch.from_numpy(indices.astype(np.int64)),
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=NUM_WORKERS,
                      pin_memory=torch.cuda.is_available())

# ============================================================
# 4. MLP baseline model
# ============================================================

class MLPResidualBaseline(nn.Module):
    """
    MLP baseline for residual learning.

    Input:
        flattened geometry + two modal parameters, 10 dimensions in total.

    Not used as network input:
        SMM diffraction efficiencies
        delta modal-difference features
        overlap integrals

    Final prediction outside the network:
        eta_pred = eta_SMM + residual_MLP
    """
    def __init__(self, input_dim: int = 10, hidden_dims=None, dropout: float = 0.08):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128, 64]

        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            if USE_BATCH_NORM:
                layers.append(nn.BatchNorm1d(h))
            else:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 2))
        self.net = nn.Sequential(*layers)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, feat):
        return self.net(feat)

# ============================================================
# 5. Training and prediction
# ============================================================

def unpack(batch, device):
    feat, target, smm, rcwa, true_res, idx = to_device(batch, device)
    return feat, target, smm, rcwa, true_res, idx


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total = 0.0

    for batch in loader:
        feat, target, *_ = unpack(batch, device)
        with torch.set_grad_enabled(is_train):
            pred = model(feat)
            loss = loss_fn(pred, target)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()
        total += loss.item() * feat.shape[0]
    return total / len(loader.dataset)


def train_model(model, train_loader, val_loader, device):
    loss_fn = nn.SmoothL1Loss(beta=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-6)

    best_state, best_val, best_epoch, wait = None, float("inf"), 0, 0
    history = []
    t0 = time.time()
    print("\n开始训练 MLP baseline...\n")

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
        feat, target, smm, rcwa, true_res, idx = unpack(batch, device)
        pred_s.append(model(feat).cpu().numpy())
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

def build_summary(df, train_idx, val_idx, test_idx, split_info, history, best_epoch, best_val, metrics, model):
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
            "feature_dim": 10,
            "feature_order": [
                "Lambda/lambda", "f", "h/lambda", "theta/90",
                "neff0_TE", "sin_phi0_TE", "cos_phi0_TE",
                "neff1_TE", "sin_phi1_TE", "cos_phi1_TE",
            ],
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
            "MODEL_TYPE": "MLPResidualBaseline",
            "MLP_HIDDEN_DIMS": MLP_HIDDEN_DIMS,
            "DROPOUT": DROPOUT,
            "USE_BATCH_NORM": USE_BATCH_NORM,
            "PARAMETERS": int(sum(p.numel() for p in model.parameters())),
            "USE_DELTA_AS_NETWORK_INPUT": False,
            "USE_SMM_EFFICIENCY_AS_NETWORK_INPUT": False,
        },
        "metrics": metrics,
    }


def save_model_and_scalers(model, scalers, split_info):
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "MODEL_TYPE": "MLPResidualBaseline",
            "INPUT_DIM": 10,
            "MLP_HIDDEN_DIMS": MLP_HIDDEN_DIMS,
            "DROPOUT": DROPOUT,
            "USE_BATCH_NORM": USE_BATCH_NORM,
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
    }, MODEL_DIR / "best_mlp_residual_baseline_10000.pth")
    joblib.dump(scalers, MODEL_DIR / "scalers_mlp_baseline_10000.pkl")


def save_prediction_csv(df, test):
    idx = test["idx"]
    pred_df = df.iloc[idx].copy()
    pred_df.insert(0, "original_index", idx)

    rcwa, mlp_pred, pred_res = test["rcwa"], test["pred_eff"], test["pred_res"]
    pred_df["mlp_pred_res_eta0_TE"] = pred_res[:, 0]
    pred_df["mlp_pred_res_etam1_TE"] = pred_res[:, 1]
    pred_df["mlp_pred_eta0_TE"] = mlp_pred[:, 0]
    pred_df["mlp_pred_etam1_TE"] = mlp_pred[:, 1]
    pred_df["mlp_pred_eta0_raw_TE"] = test["pred_eff_raw"][:, 0]
    pred_df["mlp_pred_etam1_raw_TE"] = test["pred_eff_raw"][:, 1]
    pred_df["mlp_err_eta0_TE"] = mlp_pred[:, 0] - rcwa[:, 0]
    pred_df["mlp_err_etam1_TE"] = mlp_pred[:, 1] - rcwa[:, 1]
    pred_df["mlp_abs_err_eta0_TE"] = np.abs(pred_df["mlp_err_eta0_TE"].values)
    pred_df["mlp_abs_err_etam1_TE"] = np.abs(pred_df["mlp_err_etam1_TE"].values)
    pred_df["mlp_abs_err_max_two_channels_TE"] = np.maximum(
        pred_df["mlp_abs_err_eta0_TE"].values,
        pred_df["mlp_abs_err_etam1_TE"].values,
    )
    pred_df.to_csv(OUTPUT_DIR / "test_predictions_mlp_baseline_10000.csv", index=False)


# ============================================================
# 7. Main
# ============================================================

def main():
    set_seed(SEED)
    setup_dirs()
    torch.set_num_threads(TORCH_NUM_THREADS)
    try:
        torch.set_num_interop_threads(max(1, min(2, TORCH_NUM_THREADS // 2)))
    except RuntimeError:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("MLP Residual Baseline - TE Littrow - 10000 samples")
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("CSV_PATH:", CSV_PATH)
    print("OUTPUT_DIR:", OUTPUT_DIR)
    print("SPLIT_MODE:", SPLIT_MODE)
    print("REUSE_MAIN_SPLIT_IF_AVAILABLE:", REUSE_MAIN_SPLIT_IF_AVAILABLE)
    print("=" * 80)

    df = load_dataframe()
    x = build_features(df)
    train_idx, val_idx, test_idx, split_info = make_split(df)
    xs, scalers = fit_transform_features(x, train_idx)

    np.savez(OUTPUT_DIR / "split_indices_mlp_baseline_10000.npz",
             train_indices=train_idx, val_indices=val_idx, test_indices=test_idx)
    train_loader = make_loader(xs, x, train_idx, shuffle=True)
    val_loader = make_loader(xs, x, val_idx, shuffle=False)
    test_loader = make_loader(xs, x, test_idx, shuffle=False)

    print("\n数据划分：")
    print(json.dumps(split_info, indent=4, ensure_ascii=False))
    print(f"训练集: {len(train_idx)} | 验证集: {len(val_idx)} | 测试集: {len(test_idx)}")

    model = MLPResidualBaseline(input_dim=10, hidden_dims=MLP_HIDDEN_DIMS, dropout=DROPOUT).to(device)
    print("\nMLP baseline 参数量:", sum(p.numel() for p in model.parameters()))

    history, best_epoch, best_val = train_model(model, train_loader, val_loader, device)
    pd.DataFrame(history).to_csv(OUTPUT_DIR / "training_history_mlp_baseline_10000.csv", index=False)

    test = predict(model, test_loader, scalers["res"], device)
    rcwa_true, smm_pred, mlp_pred = test["rcwa"], test["smm"], test["pred_eff"]
    true_res, pred_res = test["true_res"], test["pred_res"]

    smm_m = metric_dict(rcwa_true, smm_pred)
    mlp_m = metric_dict(rcwa_true, mlp_pred)
    improve = (smm_m["MAE"] - mlp_m["MAE"]) / smm_m["MAE"] * 100

    metrics = {
        "SMM_overall": smm_m,
        "MLP_corrected_overall": mlp_m,
        "MLP_eta0_efficiency": metric_dict(rcwa_true[:, 0], mlp_pred[:, 0]),
        "MLP_etam1_efficiency": metric_dict(rcwa_true[:, 1], mlp_pred[:, 1]),
        "Residual_eta0": metric_dict(true_res[:, 0], pred_res[:, 0]),
        "Residual_etam1": metric_dict(true_res[:, 1], pred_res[:, 1]),
        "MAE_improvement_percent": float(improve),
        "success_rate_all_channels_error_lt_0.01_percent": success_rate(rcwa_true, mlp_pred, 0.01),
        "MLP_raw_unclipped_overall": metric_dict(rcwa_true, test["pred_eff_raw"]),
    }

    save_json(build_summary(df, train_idx, val_idx, test_idx, split_info, history, best_epoch, best_val, metrics, model),
              OUTPUT_DIR / "summary_metrics_mlp_baseline_10000.json")
    save_model_and_scalers(model, scalers, split_info)
    save_prediction_csv(df, test)

    print("\n" + "=" * 80)
    print("MLP baseline 核心测试结果")
    print("=" * 80)
    print(f"划分方式: {split_info.get('split_mode', SPLIT_MODE)}")
    if "test_region" in split_info:
        print(f"测试区域: {split_info['test_region']}")
    print(f"SMM MAE:        {smm_m['MAE']:.6f}")
    print(f"MLP 修正 MAE:   {mlp_m['MAE']:.6f}")
    print(f"MAE 降低比例:   {improve:.2f}%")
    print(f"MLP R2:         {mlp_m['R2']:.6f}")
    print(f"MLP MaxError:   {mlp_m['MaxAbsError']:.6f}")
    print(f"成功率 |error|<0.01: {metrics['success_rate_all_channels_error_lt_0.01_percent']:.2f}%")
    print("=" * 80)

    print("\n训练与评估全部完成。")
    print("结果文件夹:", OUTPUT_DIR.resolve())
    print("关键输出：")
    print("  1. summary_metrics_mlp_baseline_10000.json")
    print("  2. training_history_mlp_baseline_10000.csv")
    print("  3. test_predictions_mlp_baseline_10000.csv")
    print("  4. split_indices_mlp_baseline_10000.npz")
    print("  5. models/best_mlp_residual_baseline_10000.pth")
    print("  6. models/scalers_mlp_baseline_10000.pkl")


if __name__ == "__main__":
    main()
