# path: backend/var_macro.py
# VAR(1) Monte Carlo untuk EXOG (macro) — simpan data & plot
# ---------------------------------------------------------
# pip install fastapi uvicorn statsmodels pandas numpy matplotlib

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.tseries.offsets import MonthBegin
from statsmodels.tsa.api import VAR
from statsmodels.tsa.ar_model import AutoReg  # fallback jika hanya 1 kolom

# (ASGI) impor aman—kalau fastapi belum terpasang, mode CLI tetap bisa jalan
try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import JSONResponse, FileResponse
    _FASTAPI_AVAILABLE = True
except Exception:
    _FASTAPI_AVAILABLE = False

# --------- Konstanta path dasar ---------
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"

# --------- Helpers I/O ---------
def _paths(dataset_id: str) -> Dict[str, Path]:
    ddir = DATA_DIR / dataset_id
    out = ddir / "var1_macro"
    return {
        "ddir": ddir,
        "csv": ddir / "timeseries_clean.csv",
        "sel": ddir / "selection.json",
        "out": out,
        "params_json": out / "var1_params.json",
        "sims_npy": out / "sims_macro.npy",
        "summary_csv": out / "summary_percentiles.csv",
        "fan_grid_png": out / "fan_chart_grid.png",
        "fan_overlay_png": out / "historis_plus_fan.png",
        "manifest_json": out / "manifest.json",
    }

def _load_clean_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"timeseries_clean.csv tidak ditemukan di: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError("CSV tidak memiliki cukup kolom.")
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if df[date_col].isna().all():
        raise ValueError(f"Kolom tanggal '{date_col}' gagal diparse.")
    df = df.set_index(date_col).sort_index()
    return df

def _read_selection(sel_path: Path) -> Dict[str, Any]:
    if not sel_path.exists():
        raise FileNotFoundError("selection.json tidak ditemukan. Jalankan pemilihan variabel dulu.")
    data = json.loads(sel_path.read_text(encoding="utf-8"))
    # Normalisasi penamaan kunci (fallback kompat lama)
    if "exog" not in data and "macro_cols" in data:
        data["exog"] = data.get("macro_cols", [])
    if "targets" not in data and "asset_cols" in data:
        data["targets"] = data.get("asset_cols", [])
    data["exog"] = list(data.get("exog", []))
    data["targets"] = list(data.get("targets", []))
    return data

# --------- Utils Matrix ---------
def _cholesky_pd(Sigma: np.ndarray) -> np.ndarray:
    """Cholesky robust: tambah jitter bertahap sampai matriks PD."""
    jitter = 1e-12
    I = np.eye(Sigma.shape[0])
    for _ in range(12):  # 1e-12 .. 1e-1 ~ 12 langkah
        try:
            return np.linalg.cholesky(Sigma + I * jitter)
        except np.linalg.LinAlgError:
            jitter *= 10.0
    raise np.linalg.LinAlgError("Sigma is not positive definite even after jitter.")

# --------- Core VAR(1) + Monte Carlo ---------
def fit_var1(df_exog: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit model:
      - Jika k==1 → AutoReg(AR(1)) fallback
      - Jika k>=2 → VAR(1)
    Return: Phi (k×k), c (k,), Sigma (k×k), L (k×k, cholesky)
    """
    df_clean = df_exog.dropna()
    if df_clean.shape[0] < 5:
        raise ValueError("Observasi terlalu sedikit setelah dropna() untuk fit (min ~5).")

    k = df_clean.shape[1]
    if k == 1:
        # Fallback AR(1)
        y = df_clean.iloc[:, 0]
        ar = AutoReg(y, lags=1, old_names=False).fit()
        # Ambil koef lag & konstanta
        # Nama param bisa "y.L1" atau serupa; fallback ke filter(like=".L1")
        phi1 = float(ar.params.get("y.L1", ar.params.filter(like=".L1").iloc[0] if ar.params.filter(like=".L1").size else 0.0))
        const = float(ar.params.get("const", 0.0))
        sigma2 = float(getattr(ar, "sigma2", np.var(ar.resid, ddof=1)))
        Phi = np.array([[phi1]], dtype=float)
        c = np.array([const], dtype=float)
        Sigma = np.array([[sigma2]], dtype=float)
        L = np.array([[np.sqrt(max(sigma2, 0.0))]], dtype=float)
        return Phi, c, Sigma, L
    else:
        # VAR(1)
        model = VAR(df_clean)
        res = model.fit(1)
        Phi = res.coefs[0]            # (k×k)
        c = res.intercept             # (k,)
        Sigma = res.sigma_u           # (k×k)
        L = _cholesky_pd(Sigma)
        return Phi, c, Sigma, L

def simulate_var_mc(
    Phi: np.ndarray, c: np.ndarray, L: np.ndarray, start_vals: np.ndarray,
    horizon: int = 60, n_sim: int = 1000, seed: int = 42
) -> np.ndarray:
    """Simulasi Monte Carlo VAR(1). Output (n_sim, horizon+1, k) termasuk t=0."""
    k = start_vals.shape[0]
    sims = np.zeros((n_sim, horizon + 1, k), dtype=float)
    sims[:, 0, :] = start_vals
    rng = np.random.default_rng(seed)
    for s in range(n_sim):
        x = start_vals.copy()
        for t in range(1, horizon + 1):
            eps = rng.standard_normal(k)
            x = c + Phi.dot(x) + L.dot(eps)
            sims[s, t, :] = x
    return sims

def summarize_sims(sims: np.ndarray) -> Dict[str, np.ndarray]:
    """Ringkas: mean, p10, p25, p75, p90, min, max — semua (T,k)."""
    mean = sims.mean(axis=0)
    p10, p25 = np.percentile(sims, [10, 25], axis=0)
    p75, p90 = np.percentile(sims, [75, 90], axis=0)
    minv, maxv = sims.min(axis=0), sims.max(axis=0)
    return {"mean": mean, "p10": p10, "p25": p25, "p75": p75, "p90": p90, "min": minv, "max": maxv}

def build_proj_dates(last_index: pd.Index, horizon: int) -> pd.DatetimeIndex:
    last_date = last_index[-1]
    if isinstance(last_index, pd.PeriodIndex):
        last_date = last_date.to_timestamp()
    return pd.date_range(start=last_date + MonthBegin(1), periods=horizon + 1, freq="MS")

# --------- Saving ---------
# --- PATCH: ganti fungsi ini di backend/var_macro.py ---

def save_params(paths, cols, Phi, c, Sigma) -> None:
    import numpy as _np

    def _as_list(x):
        return _np.asarray(x).tolist()  # aman untuk ndarray, Series, DataFrame, scalar

    params = {
        "k": len(cols),
        "columns": list(cols),
        "Phi": _as_list(Phi),
        "c": _as_list(c),
        "Sigma": _as_list(Sigma),
    }
    paths["params_json"].write_text(json.dumps(params, indent=2), encoding="utf-8")


def save_sims(paths: Dict[str, Path], sims: np.ndarray) -> None:
    np.save(paths["sims_npy"], sims)

def save_summary_csv(
    paths: Dict[str, Path],
    summary: Dict[str, np.ndarray],
    proj_dates: pd.DatetimeIndex,
    cols: List[str]
) -> pd.DataFrame:
    """Simpan CSV lebar MultiIndex kolom: (series, stat)."""
    T, k = summary["mean"].shape
    idx = proj_dates[:T]
    frames = []
    for j, col in enumerate(cols):
        dfj = pd.DataFrame({
            (col, "mean"): summary["mean"][:, j],
            (col, "p10"): summary["p10"][:, j],
            (col, "p25"): summary["p25"][:, j],
            (col, "p75"): summary["p75"][:, j],
            (col, "p90"): summary["p90"][:, j],
            (col, "min"): summary["min"][:, j],
            (col, "max"): summary["max"][:, j],
        }, index=idx)
        frames.append(dfj)
    out_df = pd.concat(frames, axis=1)
    out_df.index.name = "Date"
    out_df.to_csv(paths["summary_csv"])
    return out_df

# --------- Plotting ---------
def plot_fan_grid(
    paths: Dict[str, Path],
    hist_df: pd.DataFrame,
    proj_dates: pd.DatetimeIndex,
    summary: Dict[str, np.ndarray],
    exog_cols: List[str],
    exclude_cols: List[str] | set[str] = ()
) -> None:
    plot_cols = [c for c in exog_cols if c not in set(exclude_cols) and c in hist_df.columns]
    if not plot_cols:
        raise ValueError("Tidak ada kolom untuk diplot. Periksa exog/exclude/df.columns.")

    ncols = 3
    nrows = int(np.ceil(len(plot_cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.6 * nrows), sharex=False)
    axes = np.atleast_1d(axes).ravel()

    for i, col in enumerate(plot_cols):
        j = exog_cols.index(col)
        ax = axes[i]
        # Historis
        y_hist = hist_df[col].dropna()
        x_hist = y_hist.index if not isinstance(hist_df.index, pd.PeriodIndex) else y_hist.index.to_timestamp()
        ax.plot(x_hist, y_hist.values, "k", lw=2.0, label="Historis")
        # Fan chart
        ax.fill_between(proj_dates, summary["p25"][:, j], summary["p75"][:, j], alpha=0.20, label="25–75%")
        ax.fill_between(proj_dates, summary["p10"][:, j], summary["p90"][:, j], alpha=0.10, label="10–90%")
        ax.plot(proj_dates, summary["mean"][:, j], lw=1.8, label="Mean")
        ax.set_title(col)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8, loc="upper left")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Hapus axes kosong
    for k_ in range(len(plot_cols), len(axes)):
        fig.delaxes(axes[k_])

    fig.suptitle("Historis + Proyeksi (VAR(1), Monte Carlo)", fontsize=14)
    plt.tight_layout()
    fig.savefig(paths["fan_grid_png"], dpi=160)
    plt.close(fig)

# === REPLACE fungsi lama plot_overlay_all(...) DENGAN INI ===
def plot_fan_only_grid(
    paths: Dict[str, Path],
    proj_dates: pd.DatetimeIndex,
    summary: Dict[str, np.ndarray],
    exog_cols: List[str],
    exclude_cols: List[str] | set[str] = (),
) -> None:
    """
    Grid fan chart TANPA historis.
    Style:
      - Min–Max: area abu-abu
      - P10–P90: garis hijau putus-putus
      - P25–P75: garis biru titik-titik
      - Mean   : garis oranye tebal
    """
    plot_cols = [c for c in exog_cols if c not in set(exclude_cols)]
    if not plot_cols:
        raise ValueError("Tidak ada kolom untuk diplot. Periksa exog/exclude.")

    ncols = 3
    nrows = int(np.ceil(len(plot_cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.6 * nrows), sharex=False)
    axes = np.atleast_1d(axes).ravel()

    for i, col in enumerate(plot_cols):
        j = exog_cols.index(col)
        ax = axes[i]

        # FAN saja (tanpa historis)
        ax.fill_between(proj_dates, summary["min"][:, j], summary["max"][:, j],
                        color="gray", alpha=0.30, label="Min–Max")
        ax.plot(proj_dates, summary["p90"][:, j], "g--", linewidth=1.2)
        ax.plot(proj_dates, summary["p10"][:, j], "g--", linewidth=1.2, label="P10–P90")
        ax.plot(proj_dates, summary["p75"][:, j], "b:", linewidth=1.2)
        ax.plot(proj_dates, summary["p25"][:, j], "b:", linewidth=1.2, label="P25–P75")
        ax.plot(proj_dates, summary["mean"][:, j], color="orange", linewidth=2.0, label="Mean")

        ax.set_title(col)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # hapus slot kosong
    for k_ in range(len(plot_cols), len(axes)):
        fig.delaxes(axes[k_])

    fig.suptitle("Proyeksi (VAR(1), Monte Carlo) — Fan Chart", fontsize=14)
    plt.tight_layout()
    # simpan ke nama yang sama agar kompatibel:
    fig.savefig(paths["fan_overlay_png"], dpi=160)
    plt.close(fig)


# --------- Orchestrator ---------
def run_var_macro(
    dataset_id: str,
    horizon: int = 60,
    n_sim: int = 1000,
    exclude_cols: List[str] | None = ["aa10y"],  # default skip 'aa10y' di plot
    seed: int = 42,
    show_history: bool = True,
) -> Dict[str, Any]:
    """
    Jalankan pipeline VAR(1)+Monte Carlo untuk EXOG.
    Return: {"summary_df": DataFrame, "manifest": dict, "paths": dict-of-Path}
    """
    p = _paths(dataset_id)
    p["out"].mkdir(parents=True, exist_ok=True)

    # Load data & selection
    df = _load_clean_csv(p["csv"])
    sel = _read_selection(p["sel"])

    # Coerce exog ke numerik terlebih dulu agar lolos dtype check (nama dengan '-' tetap aman)
    for c in sel.get("exog", []):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Exog hanya kolom numerik yang ada di df
    exog_cols = [c for c in sel.get("exog", []) if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not exog_cols:
        raise ValueError("Tidak ada kolom exog terpilih (atau tidak numerik). Buat selection dulu.")

    # Fit
    data_var = df[exog_cols].dropna()
    if data_var.shape[0] < 5:
        raise ValueError(f"Data exog terlalu pendek setelah dropna(): {data_var.shape[0]} baris.")
    # DEBUG prints
    print(f"[DEBUG] exog_cols: {exog_cols}")
    print(f"[DEBUG] data_var shape: {data_var.shape}  (rows x exog)")

    Phi, c, Sigma, L = fit_var1(data_var)

    # Simulasi
    start_vals = data_var.iloc[-1][exog_cols].to_numpy()
    sims = simulate_var_mc(Phi, c, L, start_vals, horizon=horizon, n_sim=n_sim, seed=seed)
    summary = summarize_sims(sims)
    proj_dates = build_proj_dates(data_var.index, horizon)

    # Save artefak
    save_params(p, exog_cols, Phi, c, Sigma)
    save_sims(p, sims)
    out_df = save_summary_csv(p, summary, proj_dates, exog_cols)

    # Plots
    ex_set = exclude_cols or []
    plot_fan_grid(p, df, proj_dates, summary, exog_cols, exclude_cols=ex_set)
    plot_fan_only_grid(p, proj_dates, summary, exog_cols, exclude_cols=ex_set)

    manifest = {
        "dataset_id": dataset_id,
        "exog_cols": exog_cols,
        "horizon": horizon,
        "n_sim": n_sim,
        "seed": seed,
        "outputs": {
            "params_json": str(p["params_json"].relative_to(p["ddir"])),
            "sims_npy": str(p["sims_npy"].relative_to(p["ddir"])),
            "summary_csv": str(p["summary_csv"].relative_to(p["ddir"])),
            "fan_chart_grid_png": str(p["fan_grid_png"].relative_to(p["ddir"])),
            "historis_plus_fan_png": str(p["fan_overlay_png"].relative_to(p["ddir"])),
        },
    }
    p["manifest_json"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"summary_df": out_df, "manifest": manifest, "paths": p}

# --------- CLI ---------
def _parse_args():
    p = argparse.ArgumentParser(description="VAR(1) Monte Carlo untuk EXOG (macro) — simpan data & plot")
    p.add_argument("--dataset_id", required=True, type=str, help="ID dataset (folder di data/)")
    p.add_argument("--horizon", type=int, default=60, help="Horizon bulan (default: 60)")
    p.add_argument("--n_sim", type=int, default=1000, help="Jumlah simulasi Monte Carlo (default: 1000)")
    p.add_argument("--exclude", type=str, default="", help="Daftar kolom dipisah koma untuk di-skip pada plot (default skip 'aa10y')")
    p.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    return p.parse_args()

# ===== (ASGI) FastAPI app — opsional jalankan via uvicorn =====
if _FASTAPI_AVAILABLE:
    app = FastAPI(title="PRISM VAR(1) Macro API", version="1.0")

    @app.post("/api/forecast/macro/{dataset_id}")
    def forecast_macro(
        dataset_id: str,
        horizon: int = Query(60, ge=1, le=120),
        n_sim: int = Query(1000, ge=10, le=5000),
        exclude: str = Query("", description="Daftar kolom dipisah koma untuk di-skip pada plot; default skip 'aa10y'"),
        seed: int = Query(42),
    ):
        try:
            exclude_cols = [s.strip() for s in exclude.split(",") if s.strip()] if exclude else ["aa10y"]
            res = run_var_macro(dataset_id, horizon=horizon, n_sim=n_sim, exclude_cols=exclude_cols, seed=seed)
            return JSONResponse(res["manifest"])
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/forecast/macro/{dataset_id}/plot")
    def get_plot(dataset_id: str):
        p = _paths(dataset_id)
        png = p["fan_overlay_png"]
        if not png.exists():
            raise HTTPException(status_code=404, detail="Plot belum tersedia. Jalankan forecast dulu.")
        return FileResponse(png, media_type="image/png", filename="historis_plus_fan.png")

    @app.get("/api/forecast/macro/{dataset_id}/summary")
    def get_summary_csv(dataset_id: str):
        p = _paths(dataset_id)
        csv = p["summary_csv"]
        if not csv.exists():
            raise HTTPException(status_code=404, detail="CSV belum tersedia. Jalankan forecast dulu.")
        return FileResponse(csv, media_type="text/csv", filename="summary_percentiles.csv")

# ===== Entry point CLI =====
if __name__ == "__main__":
    args = _parse_args()
    exclude_cols = [s.strip() for s in args.exclude.split(",") if s.strip()] if args.exclude else ["aa10y"]
    res = run_var_macro(
        dataset_id=args.dataset_id,
        horizon=args.horizon,
        n_sim=args.n_sim,
        exclude_cols=exclude_cols,
        seed=args.seed,
    )
    print("=== VAR(1) Macro Done ===")
    print(json.dumps(res["manifest"], indent=2))
