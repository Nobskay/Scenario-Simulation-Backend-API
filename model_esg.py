# path: backend/model_esg.py
# ESG Scenario Generator (Mapping + Monte Carlo ESG + Plots) + SoA Exports
# -----------------------------------------------------------------------
# pip install fastapi uvicorn statsmodels pandas numpy matplotlib

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Sequence, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
from pandas.tseries.offsets import MonthBegin

# (ASGI optional)
try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import JSONResponse, FileResponse
    _FASTAPI_AVAILABLE = True
except Exception:
    _FASTAPI_AVAILABLE = False

# ---------------- Paths ----------------
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"

def _paths(dataset_id: str) -> Dict[str, Path]:
    ddir = DATA_DIR / dataset_id
    var_dir = ddir / "var1_macro"
    rec_dir = ddir / "resesi"
    esg_dir = ddir / "esg"
    return {
        "ddir": ddir,
        "csv": ddir / "timeseries_clean.csv",
        "sel": ddir / "selection.json",
        # from var_macro.py
        "var_params_json": var_dir / "var1_params.json",
        "var_summary_csv": var_dir / "summary_percentiles.csv",
        # from prob_resesi.py
        "logit_params_json": rec_dir / "logit_params.json",
        # outputs for this module
        "out": esg_dir,
        "mapping_csv": esg_dir / "mapping.csv",
        "mapping_json": esg_dir / "mapping_params.json",
        "sims_esg_npy": esg_dir / "sims_esg.npy",
        "summary_esg_csv": esg_dir / "summary_percentiles_esg.csv",
        "plot_mc_png": esg_dir / "esg_mc_grid.png",
        "plot_combo_png": esg_dir / "historis_plus_proyeksi_esg.png",
        "manifest_json": esg_dir / "manifest.json",
        # ---------------- SoA outputs ----------------
        "soa_dir": esg_dir / "soa",
        "soa_mapping_metrics_csv": esg_dir / "soa" / "mapping_metrics_per_asset.csv",
        "soa_mapping_validation_json": esg_dir / "soa" / "mapping_validation.json",
        "soa_zero_shock_macro_csv": esg_dir / "soa" / "zero_shock_macro.csv",
        "soa_one_step_stats_csv": esg_dir / "soa" / "one_step_resid_stats.csv",
        "soa_rec_prob_png": esg_dir / "soa" / "recession_prob_mean_path.png",
        "soa_rec_prob_csv": esg_dir / "soa" / "recession_prob_mean_path.csv",
        "soa_residuals_summary_json": esg_dir / "soa" / "residuals_summary.json",
    }

# ---------------- Loaders ----------------
def _load_clean_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"timeseries_clean.csv tidak ditemukan: {csv_path}")
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
    # Kompat nama lama
    if "exog" not in data and "macro_cols" in data:
        data["exog"] = data.get("macro_cols", [])
    if "targets" not in data and "asset_cols" in data:
        data["targets"] = data.get("asset_cols", [])
    data["exog"] = list(data.get("exog", []))
    data["targets"] = list(data.get("targets", []))
    return data

def _load_var_params(params_json: Path, exog_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not params_json.exists():
        raise FileNotFoundError("var1_params.json tidak ditemukan. Jalankan var_macro.py terlebih dahulu.")
    params = json.loads(params_json.read_text(encoding="utf-8"))
    Phi = np.asarray(params["Phi"], dtype=float)
    c   = np.asarray(params["c"], dtype=float).reshape(-1)
    Sigma = np.asarray(params["Sigma"], dtype=float)
    k = len(exog_cols)
    if Phi.shape != (k, k) or c.shape != (k,) or Sigma.shape != (k, k):
        raise ValueError(f"Dimensi VAR tidak cocok dengan exog: Phi {Phi.shape}, c {c.shape}, Sigma {Sigma.shape}, k={k}")
    # Cholesky robust (dari var_macro.py style)
    L = _chol_pd(Sigma)
    return Phi, c, L

def _load_logit_params(logit_json: Path) -> Dict[str, Any]:
    if not logit_json.exists():
        raise FileNotFoundError("logit_params.json tidak ditemukan. Jalankan prob_resesi.py terlebih dahulu.")
    return json.loads(logit_json.read_text(encoding="utf-8"))

# ---------------- Utils ----------------
def _chol_pd(Sigma: np.ndarray) -> np.ndarray:
    jitter = 1e-12
    I = np.eye(Sigma.shape[0])
    for _ in range(12):
        try:
            return np.linalg.cholesky(Sigma + I * jitter)
        except np.linalg.LinAlgError:
            jitter *= 10.0
    raise np.linalg.LinAlgError("Sigma not PD even after jitter.")

def _as_1d(a) -> np.ndarray:
    x = np.asarray(a, dtype=float)
    return x.reshape(-1)

def _ensure_numeric(df: pd.DataFrame, cols: Sequence[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def _build_proj_dates(last_index: pd.Index, horizon: int) -> pd.DatetimeIndex:
    last_date = last_index[-1]
    if isinstance(last_index, pd.PeriodIndex):
        last_date = last_date.to_timestamp()
    return pd.date_range(start=last_date + MonthBegin(1), periods=horizon + 1, freq="MS")

# ---------------- Step-8: Mapping aset–makro–resesi ----------------
def _ar1_rho(e: np.ndarray) -> float:
    e = np.asarray(e, dtype=float)
    e = e[np.isfinite(e)]
    if e.size < 3:
        return np.nan
    return float(np.corrcoef(e[1:], e[:-1])[0, 1])

def _fit_mapping(
    df: pd.DataFrame, exog_cols: List[str], asset_cols: List[str]
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    """
    Replikasi struktur Step-8: OLS per aset terhadap:
      Intercept + AR(1,2) aset + makro lag0/1/2  (+ kolom resesi hanya untuk pemisahan residu)
    Return:
      - map_df: DataFrame parameter dan metrik per aset
      - map_reg: matriks koef per aset (N × (3+3k)) urutan [Intercept, AR1, AR2, macro lag0, lag1, lag2]
      - regime_params: dict s_norm, s_rec, rho_norm, rho_rec, sp_all, sp_rec
      - normal_resid, recess_resid: residu historis per rezim (kolom = aset)
    """
    if "recession" not in df.columns:
        raise KeyError("Kolom 'recession' tidak ditemukan di timeseries_clean.csv")

    normal_resid = pd.DataFrame()
    recess_resid = pd.DataFrame()

    k = len(exog_cols)
    rows = []
    for asset in asset_cols:
        tmp = df[exog_cols + [asset, "recession"]].copy()
        # lag aset
        tmp[f"{asset}_lag1"] = tmp[asset].shift(1)
        tmp[f"{asset}_lag2"] = tmp[asset].shift(2)
        # lag makro
        for m in exog_cols:
            tmp[f"{m}_lag1"] = tmp[m].shift(1)
            tmp[f"{m}_lag2"] = tmp[m].shift(2)

        sub = tmp.dropna().copy()
        Xcols = (
            exog_cols
            + [f"{m}_lag1" for m in exog_cols]
            + [f"{m}_lag2" for m in exog_cols]
            + [f"{asset}_lag1", f"{asset}_lag2"]
        )
        X = sm.add_constant(sub[Xcols])
        Y = sub[asset]
        ols = sm.OLS(Y, X).fit()
        resid = ols.resid

        # Split residu by regime
        mask_rec = sub["recession"].astype(int) == 1
        normal_resid[asset] = resid[~mask_rec]
        recess_resid[asset] = resid[ mask_rec]

        def _safe_var(a):
            a = np.asarray(a, dtype=float)
            a = a[np.isfinite(a)]
            return float(a.var(ddof=1)) if a.size > 1 else np.nan

        ivar  = _safe_var(resid)
        irvar = _safe_var(resid[mask_rec])
        tcorr = _ar1_rho(resid)
        rcorr = _ar1_rho(resid[mask_rec]) if mask_rec.sum() > 2 else np.nan

        row = {
            "Asset": asset,
            "Intercept": ols.params.get("const", np.nan),
            "Autocorr_Lag1": ols.params.get(f"{asset}_lag1", np.nan),
            "Autocorr_Lag2": ols.params.get(f"{asset}_lag2", np.nan),
            "ivar": ivar, "irvar": irvar, "tcorr": tcorr, "rcorr": rcorr,
            "std_pred_all": float(ols.fittedvalues.std(ddof=1)),
        }
        # koef makro
        for m in exog_cols:
            row[m] = ols.params.get(m, np.nan)
        for m in exog_cols:
            row[f"{m}_lag1"] = ols.params.get(f"{m}_lag1", np.nan)
        for m in exog_cols:
            row[f"{m}_lag2"] = ols.params.get(f"{m}_lag2", np.nan)

        # std_pred khusus resesi (opsional)
        if mask_rec.sum() > 2:
            Xr = sm.add_constant(sub.loc[mask_rec, Xcols])
            Yr = sub.loc[mask_rec, asset]
            mr = sm.OLS(Yr, Xr).fit()
            row["std_pred_rec"] = float(mr.fittedvalues.std(ddof=1))
        else:
            row["std_pred_rec"] = np.nan

        rows.append(row)

    map_df = pd.DataFrame(rows).set_index("Asset")
    coef_names = (
        ["Intercept","Autocorr_Lag1","Autocorr_Lag2"]
        + exog_cols
        + [f"{m}_lag1" for m in exog_cols]
        + [f"{m}_lag2" for m in exog_cols]
    )
    map_reg = map_df[coef_names].to_numpy(dtype=float)

    n_assets = len(asset_cols)
    s_norm   = np.sqrt(np.nan_to_num(map_df["ivar" ].to_numpy(), nan=0.0))
    s_rec    = np.sqrt(np.nan_to_num(map_df["irvar"].to_numpy(), nan=0.0))
    rho_norm = np.clip(np.nan_to_num(map_df["tcorr"].to_numpy(),  nan=0.0), -0.95, 0.95)
    rho_rec  = np.clip(np.nan_to_num(map_df["rcorr"].to_numpy(),  nan=0.0), -0.95, 0.95)

    sp_all = map_df.get("std_pred_all", pd.Series([np.nan]*n_assets)).to_numpy()
    sp_rec = map_df.get("std_pred_rec", pd.Series([np.nan]*n_assets)).to_numpy()
    sp_all = np.where((~np.isfinite(sp_all)) | (sp_all<=0), np.where(s_norm>0, s_norm, 1.0), sp_all)
    sp_rec = np.where((~np.isfinite(sp_rec)) | (sp_rec<=0), np.where(s_rec >0, s_rec,  1.0), sp_rec)

    regime = {
        "s_norm": s_norm, "s_rec": s_rec,
        "rho_norm": rho_norm, "rho_rec": rho_rec,
        "sp_all": sp_all, "sp_rec": sp_rec,
        "coef_names": coef_names,
    }
    return map_df, map_reg, regime, normal_resid, recess_resid

# ---------------- Logit helpers (konsisten dgn prob_resesi.py) ----------------
def _logit_prob(beta: np.ndarray, x12: np.ndarray) -> float:
    z = beta[0] + float(np.dot(beta[1:], x12))
    return 1.0 / (1.0 + np.exp(-z))

def _make_x12_from_factors(m_now: np.ndarray, m_lag1: np.ndarray, m_lag2: np.ndarray,
                           factors: List[str], exog_cols: List[str]) -> np.ndarray:
    idx = [exog_cols.index(f) for f in factors]
    return np.r_[m_now[idx], m_lag1[idx], m_lag2[idx]]

# ---------------- Step-10: Monte Carlo ESG ----------------
def _build_flat_H1(j: int, k: int,
                   mf_now: np.ndarray, mf_lag1: np.ndarray, mf_lag2: np.ndarray,
                   ar_lag1: np.ndarray, ar_lag2: np.ndarray) -> np.ndarray:
    flat = np.empty(3 + 3*k, dtype=float)
    flat[0] = 1.0
    flat[1] = ar_lag1[j]
    flat[2] = ar_lag2[j]
    flat[3:3+k]       = mf_now
    flat[3+k:3+2*k]   = mf_lag1
    flat[3+2*k:3+3*k] = mf_lag2
    return flat

def _simulate_esg(
    n_steps: int,
    Phi: np.ndarray, c: np.ndarray, L: np.ndarray,
    map_reg: np.ndarray,
    regime: Dict[str, np.ndarray],
    beta_rec: np.ndarray,  # logit beta
    exog_cols: List[str], asset_cols: List[str],
    hist_mf: np.ndarray, hist_ar: np.ndarray,
    seed: int = 0
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = len(exog_cols)
    n_assets = len(asset_cols)

    s_norm, s_rec = regime["s_norm"], regime["s_rec"]
    rho_norm, rho_rec = regime["rho_norm"], regime["rho_rec"]

    mf_lag2, mf_lag1 = hist_mf[0].copy(), hist_mf[1].copy()      # (k,)
    ar_lag2, ar_lag1 = hist_ar[0].copy(), hist_ar[1].copy()      # (N,)

    # status awal resesi dari (lag1, lag2)
    rec_flag = 1 if _logit_prob(beta_rec, _make_x12_from_factors(mf_lag1, mf_lag2, mf_lag2, factors=_infer_logit_factors(beta_rec, exog_cols), exog_cols=exog_cols)) > 0.5 else 0

    # residu AR(1) awal
    e_lag1 = np.zeros(n_assets, dtype=float)

    out = [np.r_[mf_lag1, rec_flag, ar_lag1]]

    for _ in range(n_steps):
        # 1) VAR(1) makro
        eps = rng.standard_normal(k)
        mf_now = c + Phi.dot(mf_lag1) + L.dot(eps)

        # 2) status resesi
        x12 = _make_x12_from_factors(mf_now, mf_lag1, mf_lag2, factors=_infer_logit_factors(beta_rec, exog_cols), exog_cols=exog_cols)
        rec_new = 1 if _logit_prob(beta_rec, x12) > 0.5 else 0

        # 3) y_pred per aset
        y_pred = np.empty(n_assets, dtype=float)
        for j in range(n_assets):
            flat = _build_flat_H1(j, k, mf_now, mf_lag1, mf_lag2, ar_lag1, ar_lag2)
            y_pred[j] = map_reg[j].dot(flat)

        # 4) residu AR(1) per aset sesuai rezim
        rho = rho_rec if rec_new == 1 else rho_norm
        sig = s_rec  if rec_new == 1 else s_norm

        z = rng.standard_normal(n_assets)
        e_now = rho * e_lag1 + sig * np.sqrt(np.maximum(1.0 - rho**2, 0.0)) * z
        ar_now = y_pred + e_now
        e_lag1 = e_now

        # update lags
        mf_lag2, mf_lag1 = mf_lag1, mf_now
        ar_lag2, ar_lag1 = ar_lag1, ar_now

        out.append(np.r_[mf_now, rec_new, ar_now])

    return np.vstack(out)   # (n_steps+1, k+1+N)

def _infer_logit_factors(beta_rec: np.ndarray, exog_cols: List[str]) -> List[str]:
    """
    Ambil faktor dari logit_params (panjang koef = 1 + 3*len(factors)).
    Karena file ini hanya menerima beta array, fungsi ini membuat
    fallback generik: gunakan preset umum jika tersedia; kalau tidak,
    pakai empat exog pertama.
    (Dalam run() sebenarnya kita kirim factors eksplisit dari logit_params.)
    """
    # Placeholder — akan di-override oleh argumen 'logit_factors' di run()
    return exog_cols[:4]

# ---------------- Summaries & Saving ----------------
def _summarize_sims(sims: np.ndarray) -> Dict[str, np.ndarray]:
    mean = sims.mean(axis=0)
    p10, p25 = np.percentile(sims, [10, 25], axis=0)
    p75, p90 = np.percentile(sims, [75, 90], axis=0)
    return {"mean": mean, "p10": p10, "p25": p25, "p75": p75, "p90": p90}

def _save_mapping(paths: Dict[str, Path], map_df: pd.DataFrame, regime: Dict[str, np.ndarray]) -> None:
    paths["out"].mkdir(parents=True, exist_ok=True)
    map_df.to_csv(paths["mapping_csv"])
    payload = {
        "coef_names": regime["coef_names"],
        "s_norm": regime["s_norm"].tolist(),
        "s_rec": regime["s_rec"].tolist(),
        "rho_norm": regime["rho_norm"].tolist(),
        "rho_rec": regime["rho_rec"].tolist(),
        "sp_all": regime["sp_all"].tolist(),
        "sp_rec": regime["sp_rec"].tolist(),
    }
    paths["mapping_json"].write_text(json.dumps(payload, indent=2), encoding="utf-8")

def _save_sims(paths: Dict[str, Path], sims_esg: np.ndarray) -> None:
    np.save(paths["sims_esg_npy"], sims_esg)

def _save_summary_csv(paths: Dict[str, Path],
                      summary: Dict[str, np.ndarray],
                      proj_dates: pd.DatetimeIndex,
                      exog_cols: List[str],
                      asset_cols: List[str]) -> pd.DataFrame:
    """
    Simpan CSV lebar MultiIndex kolom (series, stat) untuk seluruh makro, 'recession', dan aset.
    Urutan kolom pada simulasi: [macro(k)] + [recession(1)] + [assets(N)]
    """
    mean = summary["mean"]; p10 = summary["p10"]; p25 = summary["p25"]; p75 = summary["p75"]; p90 = summary["p90"]
    T, total = mean.shape
    k = len(exog_cols)
    labels = exog_cols + ["recession"] + asset_cols
    frames = []
    for j, col in enumerate(labels):
        dfj = pd.DataFrame({
            (col, "mean"): mean[:, j],
            (col, "p10"):  p10[:, j],
            (col, "p25"):  p25[:, j],
            (col, "p75"):  p75[:, j],
            (col, "p90"):  p90[:, j],
        }, index=proj_dates)
        frames.append(dfj)
    out_df = pd.concat(frames, axis=1)
    out_df.index.name = "Date"
    out_df.to_csv(paths["summary_esg_csv"])
    return out_df

# ---------------- Plotting ----------------
def _plot_esg_mc_grid(paths: Dict[str, Path],
                      dates_esg: pd.DatetimeIndex,
                      summary: Dict[str, np.ndarray],
                      exog_cols: List[str],
                      asset_cols: List[str]) -> None:
    labels  = [m for m in exog_cols if m != "aa10y"] + ["recession"] + asset_cols
    k = len(exog_cols)
    idxs = [exog_cols.index(m) for m in exog_cols if m != "aa10y"] + [k] + list(range(k+1, k+1+len(asset_cols)))

    ncols = 4
    n_panels = len(labels)
    total_axes = n_panels + 1  # slot legend
    nrows = int(np.ceil(total_axes / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 4.8*nrows), sharex=False)
    axes = np.atleast_1d(axes).ravel()

    mean = summary["mean"]; p10 = summary["p10"]; p25 = summary["p25"]; p75 = summary["p75"]; p90 = summary["p90"]

    legend_handles, legend_labels = None, None
    for i, (idx, lab) in enumerate(zip(idxs, labels)):
        ax = axes[i]
        ln1, = ax.plot(dates_esg, mean[:, idx],  label='Mean')
        ln2, = ax.plot(dates_esg, p10[:, idx], '--', label='10%')
        ln3, = ax.plot(dates_esg, p25[:, idx], '--', label='25%')
        ln4, = ax.plot(dates_esg, p75[:, idx], '--', label='75%')
        ln5, = ax.plot(dates_esg, p90[:, idx], '--', label='90%')
        ax.set_title(lab)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        if lab.lower() == "recession":
            ax.set_ylim(-0.25, 1.25)
            ax.set_yticks([0, 0.5, 1])

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    # Legend panel
    legend_ax = axes[n_panels]
    legend_ax.axis('off')
    if legend_handles:
        legend_ax.legend(
            legend_handles, legend_labels,
            loc='center', ncol=5, frameon=True, fancybox=True,
            fontsize=10, borderaxespad=0.0
        )
    # delete remaining axes
    for j in range(n_panels + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("ESG Monte Carlo: Makro + Recession + Assets", fontsize=14)
    plt.tight_layout()
    fig.savefig(paths["plot_mc_png"], dpi=160)
    plt.close(fig)

def _plot_combo_hist_proj(paths: Dict[str, Path],
                          df: pd.DataFrame,
                          dates_esg: pd.DatetimeIndex,
                          summary: Dict[str, np.ndarray],
                          exog_cols: List[str],
                          asset_cols: List[str]) -> None:
    labels  = [m for m in exog_cols if m != "aa10y"] + ["recession"] + asset_cols
    k = len(exog_cols)
    idxs = [exog_cols.index(m) for m in exog_cols if m != "aa10y"] + [k] + list(range(k+1, k+1+len(asset_cols)))

    if isinstance(df.index, pd.PeriodIndex):
        dates_hist = df.index.to_timestamp()
    else:
        dates_hist = pd.to_datetime(df.index)

    mean = summary["mean"]; p10 = summary["p10"]; p25 = summary["p25"]; p75 = summary["p75"]; p90 = summary["p90"]

    ncols = 4
    nrows = int(np.ceil(len(labels)/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 5.0*nrows), sharex=False)
    axes = np.atleast_1d(axes).ravel()

    for i, (lab, idx) in enumerate(zip(labels, idxs)):
        ax = axes[i]
        # historis
        if lab in df.columns:
            ax.plot(dates_hist, pd.to_numeric(df[lab], errors="coerce"), 'k', lw=1.8, label='Historis')
        # proyeksi
        ax.plot(dates_esg, mean[:, idx], 'C0', lw=1.5, label='Mean')
        ax.fill_between(dates_esg, p10[:, idx], p90[:, idx], color='C0', alpha=0.10)
        ax.fill_between(dates_esg, p25[:, idx], p75[:, idx], color='C0', alpha=0.20)
        ax.set_title(lab)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8, loc='upper left')
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    for j in range(len(labels), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Historis + Proyeksi ESG: Makro + Resesi + Aset", fontsize=14)
    plt.tight_layout()
    fig.savefig(paths["plot_combo_png"], dpi=160)
    plt.close(fig)

# ---------------- SoA Export Helpers ----------------
def _export_mapping_metrics(paths, map_df: pd.DataFrame, regime: dict) -> None:
    paths["soa_dir"].mkdir(parents=True, exist_ok=True)
    map_df.to_csv(paths["soa_mapping_metrics_csv"])
    s_norm = regime["s_norm"]; s_rec = regime["s_rec"]
    rho_norm = regime["rho_norm"]; rho_rec = regime["rho_rec"]
    payload = {
        "coef_names": regime["coef_names"],
        "n_assets": int(map_df.shape[0]),
        "k_macros": int((len(regime["coef_names"]) - 3) // 3),  # 3: const, AR1, AR2
        "stats": {
            "s_norm": {"min": float(np.nanmin(s_norm)), "max": float(np.nanmax(s_norm)), "mean": float(np.nanmean(s_norm))},
            "s_rec":  {"min": float(np.nanmin(s_rec)),  "max": float(np.nanmax(s_rec)),  "mean": float(np.nanmean(s_rec))},
            "rho_norm": {"min": float(np.nanmin(rho_norm)), "max": float(np.nanmax(rho_norm)), "mean": float(np.nanmean(rho_norm))},
            "rho_rec":  {"min": float(np.nanmin(rho_rec)),  "max": float(np.nanmax(rho_rec)),  "mean": float(np.nanmean(rho_rec))},
        },
        "note": "Mapping/metrics per-asset untuk audit SoA."
    }
    paths["soa_mapping_validation_json"].write_text(json.dumps(payload, indent=2), encoding="utf-8")

def _export_zero_shock_macro(paths, df: pd.DataFrame, exog_cols: list, Phi: np.ndarray, c: np.ndarray, steps: int = 6) -> pd.DataFrame:
    sub = df[exog_cols].dropna()
    if sub.shape[0] == 0:
        raise ValueError("Data makro kosong untuk zero-shock check.")
    x = sub.iloc[-1].to_numpy(dtype=float)
    out = [x.copy()]
    for _ in range(steps):
        x = c + Phi.dot(x)  # TANPA noise
        out.append(x.copy())
    zs = pd.DataFrame(out, columns=exog_cols)
    zs.index.name = "step"
    paths["soa_dir"].mkdir(parents=True, exist_ok=True)
    zs.to_csv(paths["soa_zero_shock_macro_csv"])
    return zs

def _export_one_step_resid_stats_AR1(paths, df: pd.DataFrame, exog_cols: list, asset_cols: list,
                                     Phi: np.ndarray, c: np.ndarray, L: np.ndarray,
                                     map_reg: np.ndarray, regime: dict, mcN: int = 1000, seed: int = 123) -> pd.DataFrame:
    # Ambil state dari histori
    data_exog = df[exog_cols].dropna()
    data_asset = df[asset_cols].dropna()
    if data_exog.shape[0] < 2 or data_asset.shape[0] < 2:
        raise ValueError("Minimal 2 baris historis untuk one-step stats.")
    mf_l2 = data_exog.iloc[-2].to_numpy(float)
    mf_l1 = data_exog.iloc[-1].to_numpy(float)
    ar_l2 = data_asset.iloc[-2].to_numpy(float)
    ar_l1 = data_asset.iloc[-1].to_numpy(float)
    k = len(exog_cols); N = len(asset_cols)
    rng = np.random.default_rng(seed)

    def _flat(j, mf_now, mf_l1, mf_l2, ar_l1, ar_l2):
        flat = np.empty(3 + 3*k)
        flat[0] = 1.0
        flat[1] = ar_l1[j]
        flat[2] = ar_l2[j]
        flat[3:3+k]       = mf_now
        flat[3+k:3+2*k]   = mf_l1
        flat[3+2*k:3+3*k] = mf_l2
        return flat

    def _simulate_once(sigma: np.ndarray, rho: np.ndarray) -> float:
        eps = rng.standard_normal(k)
        mf_now = c + Phi.dot(mf_l1) + L.dot(eps)
        # yhat diperlukan untuk konsistensi pipeline, meski std resid di sini murni sigma*sqrt(1-rho^2)
        _ = np.empty(N)
        for j in range(N):
            _[j] = map_reg[j].dot(_flat(j, mf_now, mf_l1, mf_l2, ar_l1, ar_l2))
        z = rng.standard_normal(N)
        resid = sigma * np.sqrt(np.maximum(1.0 - rho**2, 0.0)) * z
        return float(np.nanstd(resid))

    emp_norm = np.mean([_simulate_once(regime["s_norm"], regime["rho_norm"]) for _ in range(mcN)])
    tgt_norm = float(np.nanmean(regime["s_norm"] * np.sqrt(np.maximum(1.0 - regime["rho_norm"]**2, 0.0))))
    emp_rec  = np.mean([_simulate_once(regime["s_rec"], regime["rho_rec"]) for _ in range(mcN)])
    tgt_rec  = float(np.nanmean(regime["s_rec"]  * np.sqrt(np.maximum(1.0 - regime["rho_rec"] **2, 0.0))))

    df_out = pd.DataFrame([
        {"regime": "normal",    "empirical_std": emp_norm, "target_std": tgt_norm, "n_mc": mcN, "n_assets": N},
        {"regime": "recession", "empirical_std": emp_rec,  "target_std": tgt_rec,  "n_mc": mcN, "n_assets": N},
    ])
    paths["soa_dir"].mkdir(parents=True, exist_ok=True)
    df_out.to_csv(paths["soa_one_step_stats_csv"], index=False)
    return df_out

def _export_residuals_summary(paths, normal_resid: pd.DataFrame, recess_resid: pd.DataFrame) -> None:
    def _q(dfR):
        if dfR is None or dfR.empty:
            return {}
        q = dfR.quantile([0.10, 0.25, 0.50, 0.75, 0.90], interpolation="linear").T
        q.columns = ["P10","P25","P50","P75","P90"]
        return {col: {k: float(v) for k, v in q.loc[col].to_dict().items()} for col in q.index}
    payload = {
        "normal": _q(normal_resid),
        "recession": _q(recess_resid)
    }
    paths["soa_dir"].mkdir(parents=True, exist_ok=True)
    paths["soa_residuals_summary_json"].write_text(json.dumps(payload, indent=2), encoding="utf-8")

def _export_recession_mean_path(paths, df: pd.DataFrame, exog_cols: list,
                                Phi: np.ndarray, c: np.ndarray,
                                beta_rec: np.ndarray, factors: list,
                                horizon: int = 60) -> pd.DataFrame:
    sub = df[exog_cols].dropna()
    if sub.shape[0] < 2:
        raise ValueError("Butuh minimal 2 baris makro historis untuk mean-path.")
    k = len(exog_cols)
    m_l2 = sub.iloc[-2].to_numpy(float)
    m_l1 = sub.iloc[-1].to_numpy(float)
    out = []
    for _ in range(horizon+1):
        idxs = [exog_cols.index(f) for f in factors]
        x12 = np.r_[m_l1[idxs], m_l2[idxs], m_l2[idxs]]
        z = beta_rec[0] + beta_rec[1:].dot(x12)
        p = 1.0/(1.0+np.exp(-z))
        out.append(p)
        m_now = c + np.dot(Phi, m_l1)
        m_l2, m_l1 = m_l1, m_now

    if isinstance(df.index, pd.PeriodIndex):
        start_dt = df.index.to_timestamp()[-1]
    else:
        start_dt = pd.to_datetime(df.index)[-1]
    dates = pd.date_range(start=start_dt + MonthBegin(1), periods=horizon+1, freq="MS")
    s = pd.Series(out, index=dates, name="prob_rec_mean")

    paths["soa_dir"].mkdir(parents=True, exist_ok=True)
    s.to_frame().to_csv(paths["soa_rec_prob_csv"])

    fig, ax = plt.subplots(figsize=(12, 4))
    if isinstance(df.index, pd.PeriodIndex):
        ax.plot(df.index.to_timestamp(), df["recession"], drawstyle='steps-post', label='Recession (hist.)')
    else:
        ax.plot(pd.to_datetime(df.index), df["recession"], drawstyle='steps-post', label='Recession (hist.)')
    ax.plot(s.index, s.values, label='Prob(Recession) — mean macro path')
    ax.grid(True, alpha=0.3); ax.legend(loc='upper right')
    ax.set_title("Historis Resesi & Probabilitas Resesi (VAR mean path)")
    ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right'); plt.tight_layout()
    fig.savefig(paths["soa_rec_prob_png"], dpi=160); plt.close(fig)
    return s.to_frame()

# ---------------- Orchestrator ----------------
def run_model_esg(
    dataset_id: str,
    horizon: Optional[int] = None,   # jika None → pakai horizon VAR summary/params
    n_sim: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Pipeline penuh:
      1) Load data & selection
      2) Load VAR(1) params (Phi,c,L) dan logit params
      3) Estimasi mapping aset–makro–resesi (Step-8) + simpan
      4) Monte Carlo ESG (Step-10, residu AR(1) rezim) → sims_esg.npy & summary
      5) Plot Step-11 & Step-12 → PNG
      6) SoA exports (audit)
      7) Manifest
    """
    p = _paths(dataset_id)
    p["out"].mkdir(parents=True, exist_ok=True)

    # 1) data & selection
    df = _load_clean_csv(p["csv"])
    sel = _read_selection(p["sel"])
    exog_cols = [c for c in sel.get("exog", []) if c in df.columns]
    asset_cols = [c for c in sel.get("targets", []) if c in df.columns]
    if not exog_cols:
        raise ValueError("Tidak ada EXOG (makro) terpilih.")
    if not asset_cols:
        raise ValueError("Tidak ada TARGET (aset) terpilih.")
    _ensure_numeric(df, exog_cols + asset_cols + ["recession"])

    # 2) VAR(1) & Logit params
    Phi, c, L = _load_var_params(p["var_params_json"], exog_cols)
    logit_params = _load_logit_params(p["logit_params_json"])
    beta_rec = np.asarray(logit_params["beta"], dtype=float)
    logit_factors = list(logit_params.get("factors", []))
    if not logit_factors:
        preset_list = [c for c in ["unemploy","gdpinv","pconsump","gdpgr"] if c in exog_cols]
        logit_factors = preset_list if preset_list else exog_cols[:4]

    # 3) Mapping aset
    map_df, map_reg, regime, normal_resid, recess_resid = _fit_mapping(df, exog_cols, asset_cols)
    _save_mapping(p, map_df, regime)

    # --- SoA Exports (audit & sanity) ---
    _export_mapping_metrics(p, map_df, regime)
    _export_zero_shock_macro(p, df, exog_cols, Phi, c, steps=6)
    _export_one_step_resid_stats_AR1(p, df, exog_cols, asset_cols, Phi, c, L, map_reg, regime, mcN=1000, seed=123)
    _export_residuals_summary(p, normal_resid, recess_resid)
    _export_recession_mean_path(p, df, exog_cols, Phi, c, beta_rec, factors=logit_factors, horizon=(horizon if horizon is not None else 60))

    # 4) Monte Carlo ESG
    data_exog = df[exog_cols].dropna()
    data_asset = df[asset_cols].dropna()
    if data_exog.shape[0] < 2 or data_asset.shape[0] < 2:
        raise ValueError("Butuh minimal 2 baris historis untuk warm-start (exog & aset).")
    hist_mf = data_exog.tail(2).to_numpy(dtype=float)      # (2,k)
    hist_ar = data_asset.tail(2).to_numpy(dtype=float)     # (2,N)

    # horizon dan tanggal proyeksi
    if horizon is None:
        if p["var_summary_csv"].exists():
            df_sum = pd.read_csv(p["var_summary_csv"], header=[0,1], index_col=0, parse_dates=True)
            horizon = len(df_sum.index) - 1
            dates_esg = df_sum.index
        else:
            horizon = 60
            dates_esg = _build_proj_dates(df.index, horizon)
    else:
        dates_esg = _build_proj_dates(df.index, horizon)

    sims_esg = np.zeros((n_sim, horizon+1, len(exog_cols) + 1 + len(asset_cols)), dtype=float)

    def _is_rec(m_now, m_l1, m_l2) -> int:
        x12 = _make_x12_from_factors(m_now, m_l1, m_l2, factors=logit_factors, exog_cols=exog_cols)
        return 1 if _logit_prob(beta_rec, x12) > 0.5 else 0

    for i in range(n_sim):
        rng = np.random.default_rng(seed + i)
        k = len(exog_cols); N = len(asset_cols)
        mf_lag2, mf_lag1 = hist_mf[0].copy(), hist_mf[1].copy()
        ar_lag2, ar_lag1 = hist_ar[0].copy(), hist_ar[1].copy()
        e_lag1 = np.zeros(N, dtype=float)
        rec0 = _is_rec(mf_lag1, mf_lag2, mf_lag2)
        path = [np.r_[mf_lag1, rec0, ar_lag1]]

        for _ in range(horizon):
            eps = rng.standard_normal(k)
            mf_now = c + Phi.dot(mf_lag1) + L.dot(eps)
            rec_new = _is_rec(mf_now, mf_lag1, mf_lag2)

            y_pred = np.empty(N, dtype=float)
            for j in range(N):
                flat = _build_flat_H1(j, k, mf_now, mf_lag1, mf_lag2, ar_lag1, ar_lag2)
                y_pred[j] = map_reg[j].dot(flat)

            rho = regime["rho_rec"] if rec_new==1 else regime["rho_norm"]
            sig = regime["s_rec"]   if rec_new==1 else regime["s_norm"]

            z = rng.standard_normal(N)
            e_now = rho * e_lag1 + sig * np.sqrt(np.maximum(1.0 - rho**2, 0.0)) * z
            ar_now = y_pred + e_now
            e_lag1 = e_now

            mf_lag2, mf_lag1 = mf_lag1, mf_now
            ar_lag2, ar_lag1 = ar_lag1, ar_now

            path.append(np.r_[mf_now, rec_new, ar_now])

        sims_esg[i] = np.vstack(path)

    _save_sims(p, sims_esg)
    summary = _summarize_sims(sims_esg)

    # 5) Simpan summary & plot
    T = summary["mean"].shape[0]
    if len(dates_esg) != T:
        dates_esg = pd.date_range(start=dates_esg[0], periods=T, freq="MS")

    out_df = _save_summary_csv(p, summary, dates_esg, exog_cols, asset_cols)
    _plot_esg_mc_grid(p, dates_esg, summary, exog_cols, asset_cols)
    _plot_combo_hist_proj(p, df, dates_esg, summary, exog_cols, asset_cols)

    # 6) Manifest
    manifest = {
        "dataset_id": dataset_id,
        "exog_cols": exog_cols,
        "asset_cols": asset_cols,
        "logit_factors": logit_factors,
        "horizon": int(horizon),
        "n_sim": int(n_sim),
        "outputs": {
            "mapping_csv": str(p["mapping_csv"].relative_to(p["ddir"])),
            "mapping_json": str(p["mapping_json"].relative_to(p["ddir"])),
            "sims_esg_npy": str(p["sims_esg_npy"].relative_to(p["ddir"])),
            "summary_esg_csv": str(p["summary_esg_csv"].relative_to(p["ddir"])),
            "plot_mc_png": str(p["plot_mc_png"].relative_to(p["ddir"])),
            "plot_combo_png": str(p["plot_combo_png"].relative_to(p["ddir"])),
            "soa_outputs": {
                "mapping_metrics_per_asset": str(p["soa_mapping_metrics_csv"].relative_to(p["ddir"])),
                "mapping_validation":        str(p["soa_mapping_validation_json"].relative_to(p["ddir"])),
                "zero_shock_macro":          str(p["soa_zero_shock_macro_csv"].relative_to(p["ddir"])),
                "one_step_resid_stats":      str(p["soa_one_step_stats_csv"].relative_to(p["ddir"])),
                "recession_prob_mean_path_png": str(p["soa_rec_prob_png"].relative_to(p["ddir"])),
                "recession_prob_mean_path_csv": str(p["soa_rec_prob_csv"].relative_to(p["ddir"])),
                "residuals_summary":         str(p["soa_residuals_summary_json"].relative_to(p["ddir"]))
            }
        }
    }
    p["manifest_json"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"summary_df": out_df, "manifest": manifest, "paths": p}

# ---------------- CLI ----------------
def _parse_args():
    p = argparse.ArgumentParser(description="ESG Mapping + Monte Carlo ESG (Makro+Resesi+Aset)")
    p.add_argument("--dataset_id", required=True, type=str, help="ID dataset (folder di data/)")
    p.add_argument("--horizon", type=int, default=0, help="Horizon bulan; 0 = auto dari VAR / default 60")
    p.add_argument("--n_sim", type=int, default=1000, help="Jumlah simulasi (default 1000)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    return p.parse_args()

# ---------------- ASGI (opsional) ----------------
if _FASTAPI_AVAILABLE:
    app = FastAPI(title="PRISM ESG Model API", version="1.0")

    @app.post("/api/esg/{dataset_id}")
    def api_esg(
        dataset_id: str,
        horizon: int = Query(0, ge=0, le=120, description="0 = auto mengikuti VAR (atau 60 jika tidak ada summary)"),
        n_sim: int = Query(1000, ge=10, le=5000),
        seed: int = Query(42),
    ):
        try:
            hz = None if horizon == 0 else int(horizon)
            res = run_model_esg(dataset_id, horizon=hz, n_sim=n_sim, seed=seed)
            return JSONResponse(res["manifest"])
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        
    @app.get("/api/esg/{dataset_id}/plot_mc")
    def api_get_plot_mc(dataset_id: str):
        p = _paths(dataset_id)
        if not p["plot_mc_png"].exists():
            raise HTTPException(status_code=404, detail="Plot Monte Carlo belum tersedia. Jalankan /api/esg terlebih dahulu.")
        return FileResponse(p["plot_mc_png"], media_type="image/png", filename="esg_mc_grid.png")

    @app.get("/api/esg/{dataset_id}/plot_combo")
    def api_get_plot_combo(dataset_id: str):
        p = _paths(dataset_id)
        if not p["plot_combo_png"].exists():
            raise HTTPException(status_code=404, detail="Plot Historis+Proyeksi belum tersedia. Jalankan /api/esg terlebih dahulu.")
        return FileResponse(p["plot_combo_png"], media_type="image/png", filename="historis_plus_proyeksi_esg.png")

    @app.get("/api/esg/{dataset_id}/summary")
    def api_get_esg_summary(dataset_id: str):
        """Download ESG summary percentiles CSV"""
        p = _paths(dataset_id)
        if not p["summary_esg_csv"].exists():
            raise HTTPException(
             status_code=404, 
            detail="ESG summary CSV belum tersedia. Jalankan /api/esg terlebih dahulu."
             )
        return FileResponse(
                    p["summary_esg_csv"], 
                    media_type="text/csv", 
                    filename=f"summary_percentiles_esg_{dataset_id}.csv"
                )

# ---------------- Main ----------------
if __name__ == "__main__":
    args = _parse_args()
    hz = None if int(args.horizon) == 0 else int(args.horizon)
    out = run_model_esg(args.dataset_id, horizon=hz, n_sim=args.n_sim, seed=args.seed)
    print(json.dumps(out["manifest"], indent=2))
