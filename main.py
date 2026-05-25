# path: backend/main.py
# PRISM Main Gateway — gabung: load, pilih_var, var_macro, prob_resesi, model_esg, file_history
# Jalankan:
#   uvicorn backend.main:app --reload --port 8000

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

# --- Sub-apps (masing-masing file sudah punya FastAPI app) ---
# Pastikan nama file dan variabel 'app' sesuai dengan file di backend/
from load import app as load_app                     # /upload, /dataset/... (sesuai load.py kamu)
from pilih_var import app as pilih_var_app           # /dataset/{id}/columns, /selection, /ui/select/{id}, ...
from var_macro import app as var_macro_app, run_var_macro
from prob_resesi import app as prob_resesi_app, run_prob_resesi
from model_esg import app as model_esg_app, run_model_esg
from file_history import app as file_history_app     # histori file/log (sesuai implementasi kamu)

# ---------- Main App ----------
app = FastAPI(title="PRISM Main API", version="1.0")

app.add_middleware(
    CORSMiddleware,
       allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "*"  # untuk development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Mount sub-apps di namespace rapi ----------
# Catatan: setiap sub-app tetap punya dokumentasi /docs masing-masing di path yang di-mount.
app.mount("/load", load_app)               # e.g., http://127.0.0.1:8000/load/docs
app.mount("/select", pilih_var_app)        # e.g., http://127.0.0.1:8000/select/docs
app.mount("/var", var_macro_app)           # e.g., http://127.0.0.1:8000/var/docs
app.mount("/resesi", prob_resesi_app)      # e.g., http://127.0.0.1:8000/resesi/docs
app.mount("/esg", model_esg_app)           # e.g., http://127.0.0.1:8000/esg/docs
app.mount("/history", file_history_app)    # e.g., http://127.0.0.1:8000/history/docs

# ---------- Halaman depan ringkas ----------
@app.get("/", response_class=HTMLResponse)
def root() -> str:
    html = """
    <html>
      <head><title>PRISM Main API</title></head>
      <body style="font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; padding: 18px;">
        <h1>PRISM Main API</h1>
        <p>Gateway untuk semua modul:</p>
        <ul>
          <li><a href="/load/docs">/load/docs</a> – Upload & persiapan data</li>
          <li><a href="/select/docs">/select/docs</a> – Pilih TARGET/EXOG + UI Checklist</li>
          <li><a href="/var/docs">/var/docs</a> – VAR(1) Monte Carlo makro</li>
          <li><a href="/resesi/docs">/resesi/docs</a> – Logistic (probabilitas resesi)</li>
          <li><a href="/esg/docs">/esg/docs</a> – Model ESG (mapping aset & simulasi)</li>
          <li><a href="/history/docs">/history/docs</a> – File history / audit</li>
        </ul>
        <h2>Pipeline Cepat</h2>
        <p>Endpoint ini menjalankan urutan penuh untuk sebuah <code>dataset_id</code>:</p>
        <pre>POST /api/pipeline/{dataset_id}?horizon=60&n_sim=1000&seed=42&history=true</pre>
        <p><a href="/docs">Dokumentasi utama (pipeline)</a></p>
      </body>
    </html>
    """
    return html

@app.get("/health")
def health():
    return {"ok": True}

# ---------- Orchestrator: jalankan berurutan VAR -> Resesi -> ESG ----------
@app.post("/api/pipeline/{dataset_id}")
def run_pipeline(
    dataset_id: str,
    horizon: int = Query(60, ge=6, le=240, description="Horizon bulan untuk VAR & ESG"),
    n_sim: int = Query(1000, ge=100, le=5000, description="Jumlah simulasi Monte Carlo"),
    seed: int = Query(42, description="Seed RNG"),
    history: bool = Query(True, description="Tampilkan historis pada plot resesi"),
    exclude: str = Query("aa10y", description="Makro yang dikecualikan pada plot (comma-separated)"),
) -> Dict[str, Any]:
    """
    Pipeline terurut:
      1) VAR(1) makro → summary & plot (var_macro.run_var_macro)
      2) Prob resesi (logit) di atas mean VAR path (prob_resesi.run_prob_resesi)
      3) Simulasi ESG (VAR + logit + mapping aset) → ringkasan & plot (model_esg.run_model_esg)
    Mengembalikan gabungan manifest dari tiap tahap.
    """
    try:
        # Step 1: VAR(1) makro
        exclude_cols = [s.strip() for s in exclude.split(",") if s.strip()]
        var_out = run_var_macro(
            dataset_id=dataset_id,
            horizon=horizon,
            n_sim=n_sim,
            exclude_cols=exclude_cols,
            seed=seed,
            show_history=True,   # overlay histori pada plot makro
        )

        # Step 2: Probabilitas resesi (gunakan hasil VAR summary)
        resesi_out = run_prob_resesi(
            dataset_id=dataset_id,
            factors_spec="",     # pakai preset otomatis bila ada
            use_all=False,
            preset="recession",
            history=history,
        )

        # Step 3: ESG Monte Carlo (mapping aset & simulasi)
        esg_out = run_model_esg(
            dataset_id=dataset_id,
            n_sim=n_sim,
            horizon=horizon,
            seed=seed,
            make_hist_plot=True,
            make_mc_plots=True,
        )

        return JSONResponse({
            "dataset_id": dataset_id,
            "pipeline": ["var_macro", "prob_resesi", "model_esg"],
            "var_macro": var_out["manifest"],
            "prob_resesi": resesi_out["manifest"],
            "model_esg": esg_out["manifest"],
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Helper redirect: bawa user ke docs utama ----------
@app.get("/docs")
def docs_redirect():
    # biar /docs di main tetap hidup untuk pipeline docs
    return RedirectResponse(url="/redoc")
