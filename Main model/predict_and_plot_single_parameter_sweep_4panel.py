import importlib.util
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent

RESULT_DIR = ROOT / "single_parameter_sweep_results"
FIG_DIR = RESULT_DIR / "figures"
PRED_DIR = RESULT_DIR / "predictions"

FIG_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

SWEEP_FILES = [
    ROOT / "sweep_h_over_lambda_for_nn.csv",
    ROOT / "sweep_Lambda_over_lambda_for_nn.csv",
]

MODEL_SCALER_CANDIDATES = [
    (
        ROOT / "proposed_method_results" / "models" / "best_modal_residual_net_10000.pth",
        ROOT / "proposed_method_results" / "models" / "scalers_10000.pkl",
    ),
    (
        ROOT / "models" / "best_modal_residual_net_10000.pth",
        ROOT / "models" / "scalers_10000.pkl",
    ),
]

MODEL_CODE_CANDIDATES = [
    ROOT / "proposed_method.py",
    ROOT.parent / "proposed_method.py",
]


SMM_COLOR = "#1f77b4"
RCWA_COLOR = "black"
PROPOSED_COLOR = "#d62728"
SMM_LW = 2.0
RCWA_LW = 1.5
PROPOSED_LW = 2.0
COMMON_DASH_PATTERN = (4.0, 2.2)
LEGEND_LOC = "upper left"
LEGEND_BBOX = (0.02, 1.01)
LEGEND_FONTSIZE = 9.0
LEGEND_HANDLELENGTH = 2.5
LEGEND_HANDLETEXTPAD = 0.7
LEGEND_BORDERPAD = 0.45
LEGEND_LABELSPACING = 0.35


def set_plot_style():
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "font.size": 20,
        "axes.linewidth": 1.2,
        "axes.unicode_minus": False,
        "axes.grid": False,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        "savefig.dpi": 900,
        "svg.fonttype": "none",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.major.width": 1.1,
        "ytick.major.width": 1.1,
    })


def load_model_module():
    for code_path in MODEL_CODE_CANDIDATES:
        if code_path.exists():
            spec = importlib.util.spec_from_file_location("proposed_method_10000", code_path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError(
        "找不到 10000proposed_method.py，请将本脚本与主模型代码放在同一目录。"
    )


def find_model_and_scaler():
    for model_path, scaler_path in MODEL_SCALER_CANDIDATES:
        if model_path.exists() and scaler_path.exists():
            return model_path, scaler_path

    msg = ["Cannot find trained model and scaler. Please check one of these pairs:"]
    for model_path, scaler_path in MODEL_SCALER_CANDIDATES:
        msg.append(f"  model:  {model_path}")
        msg.append(f"  scaler: {scaler_path}")
    raise FileNotFoundError("\n".join(msg))


def load_model_and_scalers(device):
    pm = load_model_module()
    model_path, scaler_path = find_model_and_scaler()

    print("Using model:", model_path)
    print("Using scalers:", scaler_path)

    checkpoint = torch.load(model_path, map_location=device)

    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    for key in ["D_MODEL", "N_HEADS", "N_LAYERS", "DROPOUT", "MLP_HIDDEN"]:
        if key in config:
            setattr(pm, key, config[key])

    model = pm.ModalResidualNet().to(device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    scalers = joblib.load(scaler_path)
    missing_scalers = {"mode", "geom", "res"} - set(scalers)
    if missing_scalers:
        raise KeyError(f"标准化器缺少必要键：{sorted(missing_scalers)}")
    return model, scalers, pm


# ============================================================
# 4. Prediction
# ============================================================
def build_scaled_features(df, scalers, pm):
    x = pm.build_features(df)

    mode = x["mode"]
    geom = x["geom"]

    n, t, d = mode.shape

    mode_scaled = scalers["mode"].transform(
        mode.reshape(-1, d)
    ).reshape(n, t, d).astype(np.float32)

    geom_scaled = scalers["geom"].transform(geom).astype(np.float32)

    return x, mode_scaled, geom_scaled


@torch.no_grad()
def predict_dataframe(df, model, scalers, pm, device, batch_size=512):
    x, mode_scaled, geom_scaled = build_scaled_features(df, scalers, pm)

    pred_res_scaled_list = []

    for start in range(0, len(df), batch_size):
        end = min(start + batch_size, len(df))

        mode_tensor = torch.from_numpy(mode_scaled[start:end]).to(device)
        geom_tensor = torch.from_numpy(geom_scaled[start:end]).to(device)

        pred_res_scaled = model(mode_tensor, geom_tensor).cpu().numpy()
        pred_res_scaled_list.append(pred_res_scaled)

    pred_res_scaled = np.vstack(pred_res_scaled_list)
    pred_res = scalers["res"].inverse_transform(pred_res_scaled)

    smm = x["smm"]
    rcwa = x["rcwa"]

    pred_eff_raw = smm + pred_res
    pred_eff = np.clip(pred_eff_raw, 0.0, 1.0)

    out = df.copy()

    out["pred_res_eta0_TE"] = pred_res[:, 0]
    out["pred_res_etam1_TE"] = pred_res[:, 1]

    out["pred_eta0_TE"] = pred_eff[:, 0]
    out["pred_etam1_TE"] = pred_eff[:, 1]

    out["pred_eta0_raw_TE"] = pred_eff_raw[:, 0]
    out["pred_etam1_raw_TE"] = pred_eff_raw[:, 1]

    out["err_eta0_TE"] = pred_eff[:, 0] - rcwa[:, 0]
    out["err_etam1_TE"] = pred_eff[:, 1] - rcwa[:, 1]

    out["abs_err_eta0_TE"] = np.abs(out["err_eta0_TE"].to_numpy())
    out["abs_err_etam1_TE"] = np.abs(out["err_etam1_TE"].to_numpy())

    return out


# ============================================================
# 5. Physical x-axis conversion
# ============================================================
def get_depth_um(df):
    if "h_m" in df.columns:
        return df["h_m"].to_numpy(dtype=float) * 1e6

    if "h_over_lambda" in df.columns and "lambda_m" in df.columns:
        return (
            df["h_over_lambda"].to_numpy(dtype=float)
            * df["lambda_m"].to_numpy(dtype=float)
            * 1e6
        )

    raise KeyError(
        "Cannot build depth axis. Need either 'h_m' or both "
        "'h_over_lambda' and 'lambda_m'."
    )


def get_period_nm(df):
    if "Lambda_m" in df.columns:
        return df["Lambda_m"].to_numpy(dtype=float) * 1e9

    if "Lambda_over_lambda" in df.columns and "lambda_m" in df.columns:
        return (
            df["Lambda_over_lambda"].to_numpy(dtype=float)
            * df["lambda_m"].to_numpy(dtype=float)
            * 1e9
        )

    raise KeyError(
        "Cannot build period axis. Need either 'Lambda_m' or both "
        "'Lambda_over_lambda' and 'lambda_m'."
    )


# ============================================================
# 6. Data preparation for plotting
# ============================================================
def prepare_sweep_data(df, x_values):
    if "status" in df.columns:
        ok = df["status"].astype(str).eq("OK").to_numpy()
    else:
        ok = np.ones(len(df), dtype=bool)

    data = df.loc[ok].copy()
    x_values = np.asarray(x_values, dtype=float)[ok]

    data["_x_plot"] = x_values

    required_plot_cols = [
        "_x_plot",
        "eta0_RCWA_TE",
        "eta0_SMM_TE",
        "pred_eta0_TE",
        "etam1_RCWA_TE",
        "etam1_SMM_TE",
        "pred_etam1_TE",
    ]

    data = data.replace([np.inf, -np.inf], np.nan).dropna(
        subset=required_plot_cols
    )
    data = data.sort_values("_x_plot")

    return {
        "x": data["_x_plot"].to_numpy(dtype=float),
        "eta0_rcwa": data["eta0_RCWA_TE"].to_numpy(dtype=float),
        "eta0_smm": data["eta0_SMM_TE"].to_numpy(dtype=float),
        "eta0_pred": data["pred_eta0_TE"].to_numpy(dtype=float),
        "etam1_rcwa": data["etam1_RCWA_TE"].to_numpy(dtype=float),
        "etam1_smm": data["etam1_SMM_TE"].to_numpy(dtype=float),
        "etam1_pred": data["pred_etam1_TE"].to_numpy(dtype=float),
    }


# ============================================================
# 7. Four-panel combined plotting
# ============================================================
def plot_combined_sweeps(depth_data, period_data):
    set_plot_style()

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13.5, 10.0),
        dpi=300,
        facecolor="white",
    )
    fig.patch.set_facecolor("white")

    def panel(ax, x, rcwa, smm, pred, x_label, title, panel_label):
        ax.set_facecolor("white")

        line_smm, = ax.plot(
            x,
            smm,
            color=SMM_COLOR,
            linewidth=SMM_LW,
            linestyle="-",
            label="SMM",
            zorder=2,
            dash_capstyle="butt",
        )
        line_smm.set_dashes(COMMON_DASH_PATTERN)

        ax.plot(
            x,
            rcwa,
            color=RCWA_COLOR,
            linewidth=RCWA_LW,
            linestyle="-",
            label="RCWA",
            zorder=4,
            solid_capstyle="butt",
        )

        line_proposed, = ax.plot(
            x,
            pred,
            color=PROPOSED_COLOR,
            linewidth=PROPOSED_LW,
            linestyle="-",
            label="NN",
            zorder=6,
            dash_capstyle="butt",
        )
        line_proposed.set_dashes(COMMON_DASH_PATTERN)

        ax.set_xlabel(x_label)
        ax.set_ylabel("Diffraction efficiency")
        ax.set_title(title, pad=10)

        ax.set_ylim(-0.02, 1.02)
        ax.set_yticks(np.linspace(0, 1.0, 6))

        ax.grid(False)

        legend = ax.legend(
            loc=LEGEND_LOC,
            bbox_to_anchor=LEGEND_BBOX,
            fontsize=LEGEND_FONTSIZE,
            frameon=True,
            fancybox=False,
            framealpha=0.92,
            edgecolor="0.70",
            facecolor="white",
            handlelength=LEGEND_HANDLELENGTH,
            handletextpad=LEGEND_HANDLETEXTPAD,
            borderpad=LEGEND_BORDERPAD,
            labelspacing=LEGEND_LABELSPACING,
        )
        legend.get_frame().set_facecolor("white")

        ax.text(
            0.5,
            -0.27,
            panel_label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=17,
            fontweight="bold",
        )

    # Top row: depth sweep
    panel(
        axes[0, 0],
        depth_data["x"],
        depth_data["eta0_rcwa"],
        depth_data["eta0_smm"],
        depth_data["eta0_pred"],
        r"Grating depth $h$ ($\mu$m)",
        r"$\eta_{0}$ Efficiency Comparison",
        "(a)",
    )

    panel(
        axes[0, 1],
        depth_data["x"],
        depth_data["etam1_rcwa"],
        depth_data["etam1_smm"],
        depth_data["etam1_pred"],
        r"Grating depth $h$ ($\mu$m)",
        r"$\eta_{-1}$ Efficiency Comparison",
        "(b)",
    )

    # Bottom row: period sweep
    panel(
        axes[1, 0],
        period_data["x"],
        period_data["eta0_rcwa"],
        period_data["eta0_smm"],
        period_data["eta0_pred"],
        r"Period $\Lambda$ (nm)",
        r"$\eta_{0}$ Efficiency Comparison",
        "(c)",
    )

    panel(
        axes[1, 1],
        period_data["x"],
        period_data["etam1_rcwa"],
        period_data["etam1_smm"],
        period_data["etam1_pred"],
        r"Period $\Lambda$ (nm)",
        r"$\eta_{-1}$ Efficiency Comparison",
        "(d)",
    )

    plt.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.105,
        top=0.955,
        wspace=0.28,
        hspace=0.52,
    )

    save_prefix = "combined_depth_period_sweep_rcwa_smm_corrected_10000"
    png_path = FIG_DIR / f"{save_prefix}.png"
    svg_path = FIG_DIR / f"{save_prefix}.svg"

    plt.savefig(
        png_path,
        dpi=900,
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
    )
    plt.savefig(
        svg_path,
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
    )
    plt.close(fig)

    print("Saved combined figure:", png_path)
    print("Saved combined figure:", svg_path)


# ============================================================
# 8. Main
# ============================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model, scalers, pm = load_model_and_scalers(device)

    sweep_plot_data = {}

    for csv_path in SWEEP_FILES:
        if not csv_path.exists():
            print("Skip missing file:", csv_path)
            continue

        print("\nProcessing:", csv_path)

        df = pd.read_csv(csv_path)
        pred_df = predict_dataframe(df, model, scalers, pm, device)

        pred_csv_path = PRED_DIR / csv_path.name.replace(
            "_for_nn.csv",
            "_with_prediction_10000.csv",
        )
        pred_df.to_csv(pred_csv_path, index=False)
        print("Saved prediction CSV:", pred_csv_path)

        lower_name = csv_path.name.lower()

        if "sweep_h" in lower_name:
            x_values = get_depth_um(pred_df)
            sweep_plot_data["depth"] = prepare_sweep_data(pred_df, x_values)

        elif "sweep_lambda" in lower_name:
            x_values = get_period_nm(pred_df)
            sweep_plot_data["period"] = prepare_sweep_data(pred_df, x_values)

    missing = [name for name in ("depth", "period") if name not in sweep_plot_data]

    if missing:
        print(
            "\nCombined figure was not generated because the following sweep "
            "data are missing:",
            ", ".join(missing),
        )
    else:
        plot_combined_sweeps(
            depth_data=sweep_plot_data["depth"],
            period_data=sweep_plot_data["period"],
        )

    print("\nAll sweep predictions and figures finished.")
    print("Output directory:", RESULT_DIR.resolve())


if __name__ == "__main__":
    main()
