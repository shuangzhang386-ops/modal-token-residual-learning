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

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "RCWA_TE_Littrow_LHS_10000.csv"
OUTPUT_DIR = ROOT / "only_geometrics_results"
MODEL_DIR = OUTPUT_DIR / "models"
EXPERIMENT_NAME = "geometry_scalar_token_residual_10000"

SEED = 42
SPLIT_MODE = "random"          # random / h_block / lambda_block / f_block
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
    "eta0_SMM_TE", "etam1_SMM_TE",
    "eta0_RCWA_TE", "etam1_RCWA_TE",
    "res_eta0_TE", "res_etam1_TE",
]

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
    return {
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
        "geom": StandardScaler(),
        "res": StandardScaler(),
    }
    scalers["geom"].fit(x["geom"][train_idx])
    scalers["res"].fit(x["res"][train_idx])
    xs = {
        "geom": scalers["geom"].transform(x["geom"]).astype(np.float32),
        "target_scaled": scalers["res"].transform(x["res"]).astype(np.float32),
    }
    return xs, scalers

def make_loader(xs: dict, x: dict, indices: np.ndarray, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(xs["geom"][indices]),
        torch.from_numpy(xs["target_scaled"][indices]),
        torch.from_numpy(x["smm"][indices]),
        torch.from_numpy(x["rcwa"][indices]),
        torch.from_numpy(x["res"][indices]),
        torch.from_numpy(indices.astype(np.int64)),
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=NUM_WORKERS,
                      pin_memory=torch.cuda.is_available())

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

class GeometryScalarTokenResidualNet(nn.Module):
    """
    Geometry-only ablation model with meaningful Transformer input.

    Removed from the main model:
        Mode0 token
        Mode1 token

    Geometry input representation:
        Instead of using [CLS, Geometry], the four geometry variables are treated
        as four separate scalar tokens:
            [Lambda/lambda], [f], [h/lambda], [theta/90]

    Why this version:
        With only [CLS, Geometry], the Transformer has essentially no meaningful
        token-token interaction. Here, self-attention can model interactions
        among period, fill factor, depth, and incident angle. The final readout
        uses flattening instead of mean pooling, preserving the identity-specific
        information of all four geometry tokens.

    Output:
        scaled residual [RCWA - SMM]

    Final prediction:
        eta_pred = eta_SMM + residual_NN
    """
    def __init__(self):
        super().__init__()
        self.scalar_embed = nn.Sequential(
            nn.Linear(1, D_MODEL),
            nn.LayerNorm(D_MODEL),
            nn.GELU(),
        )
        self.feature_embedding = nn.Embedding(4, D_MODEL)
        self.input_dropout = nn.Dropout(DROPOUT)
        self.encoder = nn.ModuleList([EncoderBlock(D_MODEL, N_HEADS, DROPOUT) for _ in range(N_LAYERS)])
        self.head = nn.Sequential(
            nn.LayerNorm(4 * D_MODEL),
            nn.Linear(4 * D_MODEL, MLP_HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(MLP_HIDDEN, 2),
        )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, geom):
        # geom: [B, 4]
        b = geom.shape[0]
        tokens = self.scalar_embed(geom.unsqueeze(-1))  # [B, 4, D_MODEL]
        feature_ids = torch.arange(4, device=geom.device).unsqueeze(0).expand(b, -1)
        z = self.input_dropout(tokens + self.feature_embedding(feature_ids))
        for block in self.encoder:
            z = block(z)

        # No CLS token and no pooling are used.
        # Keep all four geometry-token representations and concatenate them.
        z_flat = z.reshape(b, 4 * D_MODEL)
        return self.head(z_flat)

def unpack(batch, device):
    geom, target, smm, rcwa, true_res, idx = to_device(batch, device)
    return geom, target, smm, rcwa, true_res, idx

def run_epoch(model, loader, loss_fn, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total = 0.0
    for batch in loader:
        geom, target, *_ = unpack(batch, device)
        with torch.set_grad_enabled(is_train):
            pred = model(geom)
            loss = loss_fn(pred, target)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()
        total += loss.item() * geom.shape[0]
    return total / len(loader.dataset)

def train_model(model, train_loader, val_loader, device):
    loss_fn = nn.SmoothL1Loss(beta=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-6)
    best_state, best_val, best_epoch, wait = None, float("inf"), 0, 0
    history = []
    t0 = time.time()
    print("\n开始训练：Geometry scalar-token flatten residual ablation...\n")
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
    pred_s, target_s, smm_all, rcwa_all, res_all, idx_all = [], [], [], [], [], []
    for batch in loader:
        geom, target, smm, rcwa, true_res, idx = unpack(batch, device)
        pred_s.append(model(geom).cpu().numpy())
        target_s.append(target.cpu().numpy())
        smm_all.append(smm.cpu().numpy())
        rcwa_all.append(rcwa.cpu().numpy())
        res_all.append(true_res.cpu().numpy())
        idx_all.append(idx.cpu().numpy())
    pred_res = res_scaler.inverse_transform(np.vstack(pred_s))
    true_res = res_scaler.inverse_transform(np.vstack(target_s))
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
    pred_df.to_csv(OUTPUT_DIR / "test_predictions_only_geometrics_10000.csv", index=False)

def build_summary(df, train_idx, val_idx, test_idx, split_info, history, best_epoch, best_val, metrics):
    return {
        "experiment_name": EXPERIMENT_NAME,
        "dataset": {"csv_path": str(CSV_PATH), "valid_samples": int(len(df)),
                    "train_samples": int(len(train_idx)), "val_samples": int(len(val_idx)), "test_samples": int(len(test_idx))},
        "split_info": split_info,
        "learning_target": "scaled_residual_RCWA_minus_SMM",
        "final_prediction_form": "eta_pred = eta_SMM + residual_NN",
        "use_smm_as_network_input": False,
        "use_smm_final_addition": True,
        "model_config": {"MODEL": "GeometryScalarTokenResidualNet", "INPUT_TOKENS": "[Lambda/lambda, f, h/lambda, theta/90]",
                         "REMOVED_TOKENS": "Mode0, Mode1, CLS", "D_MODEL": D_MODEL, "N_HEADS": N_HEADS,
                         "N_LAYERS": N_LAYERS, "DROPOUT": DROPOUT, "MLP_HIDDEN": MLP_HIDDEN},
        "training": {"best_epoch": int(best_epoch), "best_val_loss": float(best_val),
                     "epochs_ran": int(len(history)), "batch_size": BATCH_SIZE,
                     "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
                     "learning_rate_initial": LR, "weight_decay": WEIGHT_DECAY},
        "metrics": metrics,
    }

def save_model_and_scalers(model, scalers, split_info):
    torch.save({
        "model_state_dict": model.state_dict(),
        "experiment_name": EXPERIMENT_NAME,
        "config": {"D_MODEL": D_MODEL, "N_HEADS": N_HEADS, "N_LAYERS": N_LAYERS,
                   "DROPOUT": DROPOUT, "MLP_HIDDEN": MLP_HIDDEN, "SPLIT_MODE": SPLIT_MODE,
                   "BLOCK_TEST_SIDE": BLOCK_TEST_SIDE, "SEED": SEED},
        "split_info": split_info,
    }, MODEL_DIR / "best_geometry_scalar_token_residual_net_10000.pth")
    joblib.dump(scalers, MODEL_DIR / "scalers_geometry_scalar_token_residual_10000.pkl")

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
    print("Geometry-only residual ablation - 10000 samples")
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
    np.savez(OUTPUT_DIR / "split_indices_only_geometrics_10000.npz",
             train_indices=train_idx, val_indices=val_idx, test_indices=test_idx)
    train_loader = make_loader(xs, x, train_idx, shuffle=True)
    val_loader = make_loader(xs, x, val_idx, shuffle=False)
    test_loader = make_loader(xs, x, test_idx, shuffle=False)
    print("\n数据划分：")
    print(json.dumps(split_info, indent=4, ensure_ascii=False))
    print(f"训练集: {len(train_idx)} | 验证集: {len(val_idx)} | 测试集: {len(test_idx)}")
    model = GeometryScalarTokenResidualNet().to(device)
    print("\n模型参数量:", sum(p.numel() for p in model.parameters()))
    history, best_epoch, best_val = train_model(model, train_loader, val_loader, device)
    pd.DataFrame(history).to_csv(OUTPUT_DIR / "training_history_only_geometrics_10000.csv", index=False)
    test = predict(model, test_loader, scalers["res"], device)
    rcwa_true, smm_pred, nn_pred = test["rcwa"], test["smm"], test["pred_eff"]
    true_res, pred_res = test["true_res"], test["pred_res"]
    smm_m = metric_dict(rcwa_true, smm_pred)
    nn_m = metric_dict(rcwa_true, nn_pred)
    improve = (smm_m["MAE"] - nn_m["MAE"]) / smm_m["MAE"] * 100
    metrics = {
        "SMM_overall": smm_m,
        "Geometry_only_corrected_overall": nn_m,
        "Geometry_only_eta0_efficiency": metric_dict(rcwa_true[:, 0], nn_pred[:, 0]),
        "Geometry_only_etam1_efficiency": metric_dict(rcwa_true[:, 1], nn_pred[:, 1]),
        "Residual_eta0": metric_dict(true_res[:, 0], pred_res[:, 0]),
        "Residual_etam1": metric_dict(true_res[:, 1], pred_res[:, 1]),
        "MAE_improvement_percent_vs_SMM": float(improve),
        "success_rate_all_channels_error_lt_0.01_percent": success_rate(rcwa_true, nn_pred, 0.01),
        "Geometry_only_raw_unclipped_overall": metric_dict(rcwa_true, test["pred_eff_raw"]),
    }
    save_json(build_summary(df, train_idx, val_idx, test_idx, split_info, history, best_epoch, best_val, metrics),
              OUTPUT_DIR / "summary_metrics_only_geometrics_10000.json")
    save_model_and_scalers(model, scalers, split_info)
    save_prediction_csv(df, test)
    print("\n" + "=" * 80)
    print("核心测试结果")
    print("=" * 80)
    print(f"划分方式: {SPLIT_MODE}")
    if SPLIT_MODE != "random":
        print(f"测试区域: {split_info['test_region']}")
    print(f"SMM MAE:        {smm_m['MAE']:.6f}")
    print(f"Geometry-only 修正 MAE: {nn_m['MAE']:.6f}")
    print(f"MAE 降低比例:  {improve:.2f}%")
    print(f"Geometry-only R2:       {nn_m['R2']:.6f}")
    print(f"Geometry-only MaxError: {nn_m['MaxAbsError']:.6f}")
    print(f"成功率 |error|<0.01: {metrics['success_rate_all_channels_error_lt_0.01_percent']:.2f}%")
    print("结果文件夹:", OUTPUT_DIR.resolve())

if __name__ == "__main__":
    main()
