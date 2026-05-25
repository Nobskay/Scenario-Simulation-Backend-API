
# path: backend/prob_resesi.py
# Probabilitas Resesi berbasis Logistic Regression + Mean Path dari VAR(1)
# -----------------------------------------------------------------------
# pip install fastapi uvicorn pandas numpy matplotlib scikit-learn

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Sequence, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.tseries.offsets import MonthBegin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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
    out = ddir / "resesi"
    return {
        "ddir": ddir,
        "csv": ddir / "timeseries_clean.csv",
        "sel": ddir / "selection.json",
        "var_summary_csv": var_dir / "summary_percentiles.csv",
        "var_params_json": var_dir / "var1_params.json",
        "out": out,
        "logit_params": out / "logit_params.json",
        "prob_csv": out / "prob_resesi.csv",
        "prob_png": out / "prob_resesi.png",
        "manifest": out / "manifest.json",
    }

# ---------------- Loaders ----------------
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
    if "exog" not in data and "macro_cols" in data:
        data["exog"] = data.get("macro_cols", [])
    if "targets" not in data and "asset_cols" in data:
        data["targets"] = data.get("asset_cols", [])
    data["exog"] = list(data.get("exog", []))
    data["targets"] = list(data.get("targets", []))
    return data

# ---------------- Flexible factor picker ----------------
def _print_macro_cols(cols: Sequence[str]) -> None:
    print("\n== Daftar Faktor Makro (exog) ==")
    print("(Ketik nomor atau nama kolom; pisahkan dengan koma/spasi; range pakai '-')\n")
    for i, c in enumerate(cols):
        print(f"{i:>3} : {c}")

def _parse_selection(raw: str, cols: Sequence[str]) -> List[str]:
    """
    Terima input seperti: "1 2 5" atau "gdpgr, infl" atau "2-4,7".
    Kembalikan list nama kolom valid (unik, urut sesuai input).
    """
    if not raw:
        return []
    out: List[str] = []
    tokens = [t.strip() for t in raw.replace(",", " ").split() if t.strip()]
    for t in tokens:
        if "-" in t and all(p.strip().isdigit() for p in t.split("-")):
            a, b = map(int, t.split("-"))
            for k in range(min(a, b), max(a, b) + 1):
                if 0 <= k < len(cols):
                    out.append(cols[k])
        elif t.isdigit():
            k = int(t)
            if 0 <= k < len(cols):
                out.append(cols[k])
        elif t in cols:
            out.append(t)
    # unik & urut
    seen, uniq = set(), []
    for c in out:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq

def pick_logit_factors_interactive(exog_cols: List[str]) -> List[str]:
    _print_macro_cols(exog_cols)
    raw = input("\nPilih faktor makro untuk model logit (contoh: 1 3 5-7): ").strip()
    factors = _parse_selection(raw, exog_cols)
    print("\n== Faktor Terpilih ==")
    print(factors if factors else "(tidak ada)")
    return factors

# ---------------- Helpers ----------------
def _ensure_numeric(df: pd.DataFrame, cols: Sequence[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def _build_proj_dates(last_index: pd.Index, horizon: int) -> pd.DatetimeIndex:
    last_date = last_index[-1]
    if isinstance(last_index, pd.PeriodIndex):
        last_date = last_date.to_timestamp()
    return pd.date_range(start=last_date + MonthBegin(1), periods=horizon + 1, freq="MS")

# ---------------- Mean macro path from VAR outputs ----------------
def _mean_path_from_summary(summary_csv: Path, exog_cols: List[str]) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    """
    Baca summary_percentiles.csv (MultiIndex kolom: (series, stat)).
    Ambil 'mean' untuk setiap exog_cols, urut sama seperti exog_cols.
    """
    df = pd.read_csv(summary_csv, header=[0, 1], index_col=0, parse_dates=True)
    # Validasi semua kolom tersedia
    missing = [c for c in exog_cols if (c, "mean") not in df.columns]
    if missing:
        raise KeyError(f"Kolom mean hilang di summary untuk: {missing}")
    # Bentuk array (T, k) sesuai urutan exog_cols
    mean_mat = np.column_stack([df[(c, "mean")].to_numpy(dtype=float) for c in exog_cols])
    return df.index, mean_mat

def _mean_path_from_params(params_json: Path, df_exog: pd.DataFrame, horizon: int, exog_cols: List[str]) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    """
    Fallback: gunakan var1_params.json (Phi, c) + nilai terakhir data_exog untuk simulasi mean path deterministik:
      x_{t+1} = c + Phi x_t   (tanpa noise)
    """
    params = json.loads(params_json.read_text(encoding="utf-8"))
    Phi = np.asarray(params["Phi"], dtype=float)
    c = np.asarray(params["c"], dtype=float)
    k = len(exog_cols)
    if Phi.shape != (k, k):
        raise ValueError(f"Bentuk Phi tidak cocok. Ditemukan {Phi.shape}, harap {k}x{k}.")
    if c.shape not in [(k,), (k, 1)]:
        raise ValueError(f"Bentuk c tidak cocok. Ditemukan {c.shape}, harap ({k},).")

    x = df_exog[exog_cols].dropna().iloc[-1].to_numpy(dtype=float)  # start
    T = horizon + 1
    path = np.zeros((T, k), dtype=float)
    path[0, :] = x
    for t in range(1, T):
        x = c + Phi.dot(x)
        path[t, :] = x
    dates = _build_proj_dates(df_exog.index, horizon)
    return dates, path

# ---------------- Logit model ----------------
def _fit_logit_with_lags(df: pd.DataFrame, factors: List[str]) -> Tuple[np.ndarray, List[str], Dict[str, float], pd.DataFrame]:
    """
    Build lag0, lag1, lag2 fitur sesuai 'factors', drop NA, fit LogisticRegression.
    Return:
      beta (intercept + coef), feat_names, metrics (train/test acc), work_df (after drop).
    """
    if "recession" not in df.columns:
        raise KeyError("Kolom 'recession' tidak ditemukan di timeseries_clean.csv")

    # Pastikan numerik
    _ensure_numeric(df, list(factors) + ["recession"])

    work = df.copy()
    # lag
    for col in factors:
        work[f"{col}_lag1"] = work[col].shift(1)
        work[f"{col}_lag2"] = work[col].shift(2)

    feat_names = factors + [f"{c}_lag1" for c in factors] + [f"{c}_lag2" for c in factors]
    drop_cols = ["recession"] + feat_names
    work = work.dropna(subset=drop_cols).copy()

    X = work[feat_names].to_numpy()
    y = work["recession"].astype(int).to_numpy()

    if X.shape[0] < 25:
        raise ValueError(f"Observasi terlalu sedikit untuk logit setelah dropna(): {X.shape[0]} baris (min ~25).")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y))>1 else None)
    logit = LogisticRegression(max_iter=10000)
    logit.fit(X_train, y_train)

    beta = np.concatenate(([float(logit.intercept_[0])], logit.coef_[0].astype(float)))
    metrics = {
        "train_acc": float(logit.score(X_train, y_train)),
        "test_acc": float(logit.score(X_test, y_test)),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_pos": int(y.sum()),
        "n_obs": int(len(y)),
    }
    return beta, feat_names, metrics, work

def _logit_prob(beta: np.ndarray, x_vec: np.ndarray) -> float:
    z = beta[0] + float(np.dot(beta[1:], x_vec))
    return 1.0 / (1.0 + np.exp(-z))

def _make_x_vec(m_now: np.ndarray, m_lag1: np.ndarray, m_lag2: np.ndarray, factors: List[str], exog_cols: List[str]) -> np.ndarray:
    # Map faktor → index di exog
    idx = [exog_cols.index(f) for f in factors]
    return np.r_[m_now[idx], m_lag1[idx], m_lag2[idx]]

# ---------------- Orchestrator ----------------
def run_prob_resesi(
    dataset_id: str,
    factors_spec: str = "",
    use_all: bool = False,
    preset: str = "",
    history: bool = True,
) -> Dict[str, Any]:
    """
    Hitung probabilitas resesi dengan Logit + mean macro path.
    - factors_spec: string pilihan faktor (mis. "1 3 5-7" atau "gdpgr, inflation")
    - use_all: gunakan semua exog sebagai faktor
    - preset: "recession" -> ['unemploy','gdpinv','pconsump','gdpgr'] (jika ada)
    - history: tampilkan historis 0/1 pada plot
    """
    p = _paths(dataset_id)
    p["out"].mkdir(parents=True, exist_ok=True)

    df = _load_clean_csv(p["csv"])
    sel = _read_selection(p["sel"])

    # Pastikan exog numerik
    _ensure_numeric(df, sel.get("exog", []))
    _ensure_numeric(df, ["recession"])

    exog_cols = [c for c in sel.get("exog", []) if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not exog_cols:
        raise ValueError("Tidak ada exog numerik di selection.json / CSV.")

    # Tentukan faktor
    if use_all:
        factors = exog_cols.copy()
    elif preset.lower() == "recession":
        preset_list = [c for c in ["unemploy", "gdpinv", "pconsump", "gdpgr"] if c in exog_cols]
        factors = preset_list if preset_list else exog_cols[:4]
    else:
        factors = _parse_selection(factors_spec, exog_cols) if factors_spec else []
        if not factors:
            # fallback aman: 4 pertama
            factors = exog_cols[:4]

    # Fit logit
    beta, feat_names, metrics, work_df = _fit_logit_with_lags(df, factors)

    # Ambil mean path dari ringkasan VAR bila tersedia; jika tidak, fallback ke params
    if p["var_summary_csv"].exists():
        dates_proj, mean_mat = _mean_path_from_summary(p["var_summary_csv"], exog_cols)
        horizon = len(dates_proj) - 1
    elif p["var_params_json"].exists():
        # fallback ke simulasi deterministik
        horizon = 60
        dates_proj, mean_mat = _mean_path_from_params(p["var_params_json"], df[exog_cols], horizon, exog_cols)
    else:
        raise FileNotFoundError("Tidak menemukan output VAR. Jalankan var_macro.py terlebih dahulu (summary atau params).")

    # Bangun prob resesi pada mean path
    prob = []
    # start lags dari dua observasi terakhir mean path? gunakan data historis terbaru dari df agar realistis
    data_var = df[exog_cols].dropna()
    if data_var.shape[0] < 2:
        raise ValueError("Data historis exog minimal 2 baris untuk membentuk lag.")
    mf_lag2 = data_var.iloc[-2].to_numpy(dtype=float)
    mf_lag1 = data_var.iloc[-1].to_numpy(dtype=float)

    for t in range(0, horizon + 1):
        mf_now = mean_mat[t, :]
        x_vec = _make_x_vec(mf_now, mf_lag1, mf_lag2, factors, exog_cols)
        prob.append(_logit_prob(beta, x_vec))
        mf_lag2, mf_lag1 = mf_lag1, mf_now

    prob = np.clip(np.array(prob, dtype=float), 0.0, 1.0)

    # ---------------- Save artefacts ----------------
    # Params
    params = {
        "dataset_id": dataset_id,
        "factors": factors,
        "feat_names": feat_names,
        "beta": [float(b) for b in beta],
        "metrics": metrics,
        "exog_cols": exog_cols,
        "horizon": int(horizon),
        "source": "summary_percentiles.csv" if p["var_summary_csv"].exists() else "var1_params.json",
    }
    p["logit_params"].write_text(json.dumps(params, indent=2), encoding="utf-8")

    # CSV probs
    prob_df = pd.DataFrame({"Date": dates_proj, "prob_recession": prob})
    prob_df.to_csv(p["prob_csv"], index=False)

    # Plot
    _plot_prob(
        out_png=p["prob_png"],
        df=df,
        dates_proj=dates_proj,
        prob=prob,
        history=history,
    )

    # Manifest
    manifest = {
        "dataset_id": dataset_id,
        "outputs": {
            "logit_params_json": str(p["logit_params"].relative_to(p["ddir"])),
            "prob_resesi_csv": str(p["prob_csv"].relative_to(p["ddir"])),
            "prob_resesi_png": str(p["prob_png"].relative_to(p["ddir"])),
        },
        "meta": {
            "train_acc": metrics["train_acc"],
            "test_acc": metrics["test_acc"],
            "factors": factors,
            "horizon": horizon,
        }
    }
    p["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {"manifest": manifest, "prob_df": prob_df, "params": params}

# ---------------- Plotting ----------------
def _plot_prob(
    out_png: Path,
    df: pd.DataFrame,
    dates_proj: pd.DatetimeIndex,
    prob: Sequence[float],
    history: bool = True,
) -> None:
    """Plot garis probabilitas resesi; opsional tampilkan historis 0/1 sebagai step."""
    prob = np.asarray(prob, dtype=float)
    fig, ax = plt.subplots(figsize=(12, 4))

    if history and "recession" in df.columns:
        y_hist = pd.to_numeric(df["recession"], errors="coerce").dropna()
        x_hist = y_hist.index if not isinstance(y_hist.index, pd.PeriodIndex) else y_hist.index.to_timestamp()
        ax.plot(x_hist, y_hist.values, drawstyle="steps-post", lw=2.0, color="k", label="Recession (hist.)")
        ax.fill_between(x_hist, 0, y_hist.values, step="post", color="grey", alpha=0.15)

    # Probabilitas (mean path)
    ax.plot(dates_proj, prob, lw=2.5, color="orange", label="Prob(Recession) — mean macro path")

    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Probability / Indicator")
    ax.grid(True, alpha=0.3)
    ax.set_title("Probabilitas Resesi 60 Bulan (Logit + Mean VAR Path)")
    ax.legend(loc="upper right")

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

# ---------------- CLI ----------------
def _parse_args():
    p = argparse.ArgumentParser(description="Probabilitas Resesi (Logit) + Mean Path dari VAR(1)")
    p.add_argument("--dataset_id", required=True, type=str, help="ID dataset (folder di data/)")
    p.add_argument("--factors", type=str, default="", help="Pilihan faktor, contoh: '1 3 5-7' atau 'gdpgr,inflation'")
    p.add_argument("--use_all", action="store_true", help="Gunakan semua exog sebagai faktor")
    p.add_argument("--preset", type=str, default="", help="Preset faktor: 'recession'")
    p.add_argument("--no_history", action="store_true", help="Sembunyikan historis resesi pada plot")
    return p.parse_args()

# ---------------- ASGI (opsional) ----------------
if _FASTAPI_AVAILABLE:
    app = FastAPI(title="PRISM Recession Probability API", version="1.0")

    @app.post("/api/recession/{dataset_id}")
    def api_prob_resesi(
        dataset_id: str,
        factors: str = Query("", description="Contoh: '1 3 5-7' atau 'gdpgr,inflation'"),
        use_all: bool = Query(False),
        preset: str = Query("", description="Gunakan 'recession' untuk preset"),
        history: bool = Query(True, description="Tampilkan historis 0/1 pada plot"),
    ):
        try:
            res = run_prob_resesi(dataset_id, factors_spec=factors, use_all=use_all, preset=preset, history=history)
            return JSONResponse(res["manifest"])
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/recession/{dataset_id}/plot")
    def api_get_plot(dataset_id: str):
        p = _paths(dataset_id)
        if not p["prob_png"].exists():
            raise HTTPException(status_code=404, detail="Plot belum tersedia. Jalankan proses dulu.")
        return FileResponse(p["prob_png"], media_type="image/png", filename="prob_resesi.png")

    @app.get("/api/recession/{dataset_id}/csv")
    def api_get_csv(dataset_id: str):
        p = _paths(dataset_id)
        if not p["prob_csv"].exists():
            raise HTTPException(status_code=404, detail="CSV belum tersedia. Jalankan proses dulu.")
        return FileResponse(p["prob_csv"], media_type="text/csv", filename="prob_resesi.csv")

# ---------------- Entry ----------------
if __name__ == "__main__":
    args = _parse_args()
    res = run_prob_resesi(
        dataset_id=args.dataset_id,
        factors_spec=args.factors,
        use_all=args.use_all,
        preset=args.preset,
        history=not args.no_history,
    )
    print("=== Prob Resesi Done ===")
    print(json.dumps(res["manifest"], indent=2))
