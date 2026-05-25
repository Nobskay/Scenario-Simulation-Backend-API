# load.py
# FastAPI: Upload -> Clean -> Display (audit + plot) — NO CLIPPING EXTREMES
# ------------------------------------------------------------------------
# Dependensi:
#   pip install fastapi uvicorn pandas numpy matplotlib python-multipart
#   pip install openpyxl  # (.xlsx)
#   pip install xlrd      # (.xls lama, opsional)
#
# Menjalankan:
#   uvicorn load:app --reload --port 8000

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Tuple
from uuid import uuid4
from io import BytesIO
import traceback
import os

app = FastAPI(title="PRISM Load/Clean API", version="1.3 (No-Clip Extremes + Audit Report)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # sesuaikan di production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Kandidat nama kolom tanggal yang umum
DATE_CANDIDATES = [
    "Date", "date", "DATE", "Tanggal", "tanggal", "TANGGAL",
    "period", "Period", "PERIOD", "Month", "month", "MONTH",
    "Unnamed: 0", "index",
]

# -------------------------------
# Util: Deteksi kolom tanggal
# -------------------------------
def _find_date_column(df: pd.DataFrame) -> str:
    cols_norm = [str(c).strip() for c in df.columns]
    df.columns = cols_norm
    for col in DATE_CANDIDATES:
        if col in df.columns:
            try:
                parsed = pd.to_datetime(df[col], errors="coerce", dayfirst=False)
                if parsed.notna().mean() > 0.8:
                    return col
            except Exception:
                pass
    # fallback: coba kolom pertama
    first = df.columns[0]
    try:
        parsed = pd.to_datetime(df[first], errors="coerce", dayfirst=False)
        if parsed.notna().mean() > 0.8:
            return first
    except Exception:
        pass
    raise HTTPException(status_code=400, detail=(
        "Kolom tanggal tidak ditemukan. Pastikan CSV/XLSX memiliki kolom tanggal "
        "mis. 'Date' atau letakkan tanggal di kolom pertama."
    ))

# -------------------------------
# Util: Plot multi-panel
# -------------------------------
def _multipanel_plot(df: pd.DataFrame) -> BytesIO:
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        raise HTTPException(status_code=400, detail="Tidak ada kolom numerik untuk diplot.")
    n_cols = 3
    n_rows = (len(num_cols) + n_cols - 1) // n_cols
    plt.figure(figsize=(18, 4 * n_rows))
    for i, col in enumerate(num_cols, start=1):
        ax = plt.subplot(n_rows, n_cols, i)
        ax.plot(df.index, df[col], linewidth=1)
        ax.set_title(str(col))
        ax.grid(True)
    plt.suptitle("Time Series (Extremes are NOT clipped)", y=1.02, fontsize=12)
    plt.tight_layout()
    bio = BytesIO()
    plt.savefig(bio, format="png", dpi=120, bbox_inches="tight")
    plt.close()
    bio.seek(0)
    return bio

# -------------------------------
# Util: Outlier report (report-only)
# -------------------------------
def _extreme_report(s: pd.Series, z_thresh: float = 3.0, low_q=0.01, high_q=0.99) -> dict:
    """
    Report-only untuk mendeteksi titik ekstrem tanpa mengubah data.
    - Kuantil: di luar [p01, p99]
    - Z-score: |z| > z_thresh
    """
    s_num = pd.to_numeric(s, errors="coerce")
    valid = s_num.dropna()
    if valid.empty:
        return {
            "filled_by_interpolation": 0,
            "q_outliers": 0, "q_outliers_pct": 0.0,
            "z_outliers": 0, "z_outliers_pct": 0.0,
            "p01": None, "p99": None, "mean": None, "std": None
        }
    mean = valid.mean()
    std = valid.std(ddof=1)
    # z-score flags
    if std and np.isfinite(std) and std != 0:
        z_flags = ((valid - mean).abs() > z_thresh * std)
        z_out = int(z_flags.sum())
        z_pct = float(100 * z_flags.mean())
    else:
        z_out, z_pct = 0, 0.0
    # quantile flags
    p01 = valid.quantile(low_q)
    p99 = valid.quantile(high_q)
    q_flags = (valid < p01) | (valid > p99)
    q_out = int(q_flags.sum())
    q_pct = float(100 * q_flags.mean())
    return {
        "q_outliers": q_out, "q_outliers_pct": q_pct,
        "z_outliers": z_out, "z_outliers_pct": z_pct,
        "p01": float(p01), "p99": float(p99),
        "mean": float(mean), "std": float(std if np.isfinite(std) else 0.0)
    }

# -------------------------------
# Core: Cleaning (tanpa clipping)
# -------------------------------
def _clean_timeseries(df: pd.DataFrame, date_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    - Parse tanggal, set index bulanan (MS)
    - Coerce numerik
    - Audit sebelum imputasi
    - Reindex ke kalender bulanan penuh
    - Interpolate only + ffill/bfill (TANPA clip/winsorize)
    - Outlier audit report (report-only)
    """
    df = df.copy()
    df.rename(columns={date_col: "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].isna().all():
        raise HTTPException(status_code=400, detail="Kolom tanggal gagal di-parse ke datetime.")
    df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
    df = df.asfreq("MS")

    # pastikan kolom numerik
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Audit sebelum imputasi
    audit_before = (
        df.agg(["count", "nunique", "min", "mean", "max", "std"]).T
          .assign(
              missing=lambda x: len(df) - x["count"],
              missing_pct=lambda x: 100 * (len(df) - x["count"]) / len(df)
          )
          .sort_index()
    )

    # Reindex kalender bulanan penuh
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="MS")
    df = df.reindex(full_idx)

    # Interpolasi time-based + ffill/bfill: hanya isi NaN, nilai non-NaN tidak diubah
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    filled_counts = {}
    if num_cols:
        before_nulls = df[num_cols].isna().sum()
        df[num_cols] = df[num_cols].interpolate(method="time", limit_direction="both")
        df[num_cols] = df[num_cols].ffill().bfill()
        after_nulls = df[num_cols].isna().sum()
        filled_counts = (before_nulls - after_nulls).to_dict()

    # Outlier report (report-only): kuantil + zscore
    report_rows = []
    for c in num_cols:
        stats = _extreme_report(df[c])
        report_rows.append({
            "column": c,
            "filled_by_interpolation": int(filled_counts.get(c, 0)),
            "q_outliers_count": stats["q_outliers"],
            "q_outliers_pct": stats["q_outliers_pct"],
            "z_outliers_count": stats["z_outliers"],
            "z_outliers_pct": stats["z_outliers_pct"],
            "p01": stats["p01"],
            "p99": stats["p99"],
            "mean": stats["mean"],
            "std": stats["std"]
        })
    audit_report = pd.DataFrame(report_rows).set_index("column") if report_rows else pd.DataFrame()

    return df, audit_before, audit_report

# -------------------------------
# Util: Simpan output
# -------------------------------
def _save_outputs(df_clean: pd.DataFrame, dataset_dir: Path) -> Path:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    out = df_clean.copy().sort_index().asfreq("MS")
    out = out.reset_index().rename(columns={"index": "Date"})
    out.to_csv(dataset_dir / "timeseries_clean.csv", index=False)
    return dataset_dir / "timeseries_clean.csv"

# -------------------------------
# Model response
# -------------------------------
class UploadResponse(BaseModel):
    dataset_id: str
    rows: int
    cols: int
    columns: List[str]
    date_start: Optional[str] = None
    date_end: Optional[str] = None
    source_type: str
    outputs: dict

# -------------------------------
# File sniffers (CSV/XLS/XLSX)
# -------------------------------
def _is_probably_xlsx(head: bytes) -> bool:
    # XLSX adalah ZIP; banyak file Office diawali 'PK'
    return len(head) >= 2 and head[:2] == b"PK"

def _is_probably_xls(head: bytes) -> bool:
    # XLS lama (BIFF) pakai OLE Compound File header
    return len(head) >= 8 and head[:8] == b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1"

def _read_excel(raw: bytes) -> pd.DataFrame:
    bio = BytesIO(raw)
    try:
        # pakai engine openpyxl untuk xlsx
        return pd.read_excel(bio, engine="openpyxl")
    except ImportError as ie:
        raise HTTPException(
            status_code=415,
            detail=f"Gagal membaca XLSX: {ie}. Install 'openpyxl'."
        )

def _read_xls(raw: bytes) -> pd.DataFrame:
    bio = BytesIO(raw)
    try:
        return pd.read_excel(bio, engine="xlrd")
    except ImportError as ie:
        raise HTTPException(
            status_code=415,
            detail=f"Gagal membaca XLS: {ie}. Install 'xlrd'."
        )

def _read_csv_with_fallbacks(raw: bytes) -> Optional[pd.DataFrame]:
    for sep in [",", ";", "\t", "|"]:
        try:
            df_try = pd.read_csv(BytesIO(raw), sep=sep)
            if df_try.shape[1] >= 1:
                return df_try
        except Exception:
            continue
    return None

def _read_uploaded_table(raw_bytes: bytes, filename: Optional[str], content_type: Optional[str]) -> Tuple[pd.DataFrame, str]:
    """
    Urutan deteksi:
    1) Signature bytes (paling andal): ZIP->XLSX; OLE->XLS
    2) Ekstensi (xlsx/xls/csv)
    3) MIME hint
    4) CSV fallback (multi delimiter)
    """
    head = raw_bytes[:8]
    name_lower = (filename or "").lower()
    ext = os.path.splitext(name_lower)[1]

    # 1) Signature bytes
    if _is_probably_xlsx(head):
        try:
            return _read_excel(raw_bytes), "excel_xlsx"
        except HTTPException:
            raise
        except Exception:
            # kalau gagal, coba CSV fallback (beberapa file salah ekstensi)
            df_csv = _read_csv_with_fallbacks(raw_bytes)
            if df_csv is not None:
                return df_csv, "csv"
            raise HTTPException(status_code=400, detail="File bertipe ZIP tetapi tidak valid sebagai XLSX/CSV.")
    if _is_probably_xls(head):
        try:
            return _read_xls(raw_bytes), "excel_xls"
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Gagal membaca XLS: {e}")

    # 2) Ekstensi
    if ext == ".xlsx":
        return _read_excel(raw_bytes), "excel_xlsx"
    if ext == ".xls":
        return _read_xls(raw_bytes), "excel_xls"
    if ext in [".csv", ".txt"]:
        df_csv = _read_csv_with_fallbacks(raw_bytes)
        if df_csv is not None:
            return df_csv, "csv"

    # 3) MIME hint
    ct = (content_type or "").lower()
    if "sheet" in ct:  # e.g., application/vnd.openxmlformats-officedocument.spreadsheetml.sheet
        return _read_excel(raw_bytes), "excel_xlsx"
    if "excel" in ct:
        try:
            return _read_excel(raw_bytes), "excel_xlsx"
        except HTTPException:
            return _read_xls(raw_bytes), "excel_xls"

    # 4) CSV fallback terakhir
    df_csv = _read_csv_with_fallbacks(raw_bytes)
    if df_csv is not None:
        return df_csv, "csv"

    raise HTTPException(
        status_code=400,
        detail="Gagal membaca file. Pastikan format CSV/XLSX/XLS valid (tidak terenkripsi/protected)."
    )

# -------------------------------
# Endpoints
# -------------------------------
@app.post("/upload", response_model=UploadResponse)
async def upload_table(file: UploadFile = File(...)):
    """
    Terima CSV/XLS/XLSX, langsung proses:
    - deteksi kolom tanggal
    - cleaning (monthly index, interpolate missing only; NO clipping extremes)
    - simpan output timeseries_clean.csv
    - generate audit awal (sebelum imputasi) + audit outlier report (report-only)
    - generate plot multi-panel
    """
    try:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="File kosong.")

        df, source_type = _read_uploaded_table(raw, file.filename, file.content_type)

        date_col = _find_date_column(df)
        df_clean, audit_before, audit_report = _clean_timeseries(df, date_col=date_col)

        dataset_id = uuid4().hex[:10]
        dataset_dir = DATA_DIR / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # simpan clean
        _save_outputs(df_clean, dataset_dir)

        # simpan audit
        (audit_before).to_csv(dataset_dir / "audit_before.csv")
        if not audit_report.empty:
            (audit_report).to_csv(dataset_dir / "audit_report.csv")

        # simpan head preview
        df_clean.reset_index().rename(columns={"index": "Date"}).head(10).to_csv(dataset_dir / "head.csv", index=False)

        # simpan plot
        plot_png = _multipanel_plot(df_clean)
        with open(dataset_dir / "plot.png", "wb") as f:
            f.write(plot_png.read())

        resp = UploadResponse(
            dataset_id=dataset_id,
            rows=df_clean.shape[0],
            cols=df_clean.shape[1],
            columns=[str(c) for c in df_clean.columns],
            date_start=(df_clean.index.min().strftime("%Y-%m-%d") if not df_clean.empty else None),
            date_end=(df_clean.index.max().strftime("%Y-%m-%d") if not df_clean.empty else None),
            source_type=source_type,
            outputs={
                "clean_csv": f"/dataset/{dataset_id}/download/clean",
                "audit_csv": f"/dataset/{dataset_id}/audit",
                "audit_report_csv": f"/dataset/{dataset_id}/audit_report",
                "head_csv": f"/dataset/{dataset_id}/head",
                "plot_png": f"/dataset/{dataset_id}/plot",
            }
        )
        return resp
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server error: {e}")

@app.get("/dataset/{dataset_id}/head")
def get_head(dataset_id: str):
    path = DATA_DIR / dataset_id / "head.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Dataset tidak ditemukan.")
    return FileResponse(path, media_type="text/csv", filename="head.csv")

@app.get("/dataset/{dataset_id}/audit")
def get_audit(dataset_id: str):
    path = DATA_DIR / dataset_id / "audit_before.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Dataset tidak ditemukan.")
    return FileResponse(path, media_type="text/csv", filename="audit_before.csv")

@app.get("/dataset/{dataset_id}/audit_report")
def get_audit_report(dataset_id: str):
    path = DATA_DIR / dataset_id / "audit_report.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audit report tidak ditemukan.")
    return FileResponse(path, media_type="text/csv", filename="audit_report.csv")

@app.get("/dataset/{dataset_id}/plot")
def get_plot(dataset_id: str):
    path = DATA_DIR / dataset_id / "plot.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Plot tidak ditemukan.")
    return FileResponse(path, media_type="image/png", filename="plot.png")

@app.get("/dataset/{dataset_id}/download/clean")
def download_clean(dataset_id: str):
    path = DATA_DIR / dataset_id / "timeseries_clean.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="File tidak ditemukan.")
    return FileResponse(path, media_type="text/csv", filename="timeseries_clean.csv")
