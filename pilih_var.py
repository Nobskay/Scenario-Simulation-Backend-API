# pilih_var.py
# FastAPI: Pilih variabel (TARGET / EXOG) via Checklist UI + API
# --------------------------------------------------------------
# Dependensi:
#   pip install fastapi uvicorn pandas python-multipart
#
# Jalankan:
#   uvicorn pilih_var:app --reload --port 8000
#
# Asumsi:
#   Struktur folder sama dengan load.py:
#     data/<dataset_id>/timeseries_clean.csv
#
# Fitur:
#   - GET  /dataset/{dataset_id}/columns     -> daftar kolom + tipe (JSON)
#   - GET  /dataset/{dataset_id}/selection   -> baca selection.json (jika sudah ada)
#   - POST /dataset/{dataset_id}/select      -> simpan pilihan target/exog (JSON body)
#   - GET  /dataset/{dataset_id}/matrix/y    -> download y.csv
#   - GET  /dataset/{dataset_id}/matrix/X    -> download X.csv
#   - GET  /ui/select/{dataset_id}           -> Checklist HTML (tanpa framework)
#
# Catatan:
#   - Menganggap kolom tanggal ada di kolom pertama (hasil dari load.py = "Date")
#   - Hanya izinkan kolom numerik untuk TARGET dan EXOG
#   - EXOG tidak boleh menumpuk dengan TARGET

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from pydantic import BaseModel, Field
from pathlib import Path
import pandas as pd
import json

app = FastAPI(title="PRISM Column Picker API", version="1.0")

app.add_middleware(
    CORSMiddleware,
      allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "*"  # untuk development
    ],  # ganti sesuai kebutuhan
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# ----------------- Utils -----------------
def _dataset_paths(dataset_id: str):
    ddir = DATA_DIR / dataset_id
    csv = ddir / "timeseries_clean.csv"
    sel = ddir / "selection.json"
    ycsv = ddir / "y.csv"
    Xcsv = ddir / "X.csv"
    return ddir, csv, sel, ycsv, Xcsv

def _load_clean_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="timeseries_clean.csv tidak ditemukan. Jalankan /upload dulu.")
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise HTTPException(status_code=400, detail="CSV tidak memiliki cukup kolom.")
    # pastikan kolom 0 adalah tanggal
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if df[date_col].isna().all():
        raise HTTPException(status_code=400, detail=f"Kolom tanggal '{date_col}' gagal diparse.")
    df = df.set_index(date_col).sort_index()
    return df

def _numeric_columns(df: pd.DataFrame):
    return df.select_dtypes(include="number").columns.tolist()

def _save_selection(sel_path: Path, payload: dict):
    sel_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

def _read_selection(sel_path: Path):
    if not sel_path.exists():
        raise HTTPException(status_code=404, detail="Belum ada selection. Buat dulu via POST /select.")
    return json.loads(sel_path.read_text(encoding="utf-8"))

# ----------------- Schemas -----------------
class SelectRequest(BaseModel):
    targets: list[str] = Field(default_factory=list)
    exog: list[str] = Field(default_factory=list)

class ColumnsResponse(BaseModel):
    dataset_id: str
    date_col: str
    numeric_cols: list[str]
    all_cols: list[str]

class SelectResponse(BaseModel):
    dataset_id: str
    date_col: str
    targets: list[str]
    exog: list[str]
    files: dict

# ----------------- Endpoints: JSON API -----------------
@app.get("/dataset/{dataset_id}/columns", response_model=ColumnsResponse)
def get_columns(dataset_id: str):
    ddir, csv_path, sel_path, ycsv, Xcsv = _dataset_paths(dataset_id)
    df = _load_clean_csv(csv_path)
    date_col = df.index.name or "Date"
    num_cols = _numeric_columns(df)
    all_cols = [date_col] + df.columns.tolist()
    return ColumnsResponse(
        dataset_id=dataset_id,
        date_col=date_col,
        numeric_cols=num_cols,
        all_cols=all_cols
    )

@app.get("/dataset/{dataset_id}/selection")
def get_selection(dataset_id: str):
    ddir, csv_path, sel_path, ycsv, Xcsv = _dataset_paths(dataset_id)
    data = _read_selection(sel_path)
    return JSONResponse(data)

@app.post("/dataset/{dataset_id}/select", response_model=SelectResponse)
def post_selection(dataset_id: str, req: SelectRequest = Body(...)):
    ddir, csv_path, sel_path, ycsv, Xcsv = _dataset_paths(dataset_id)
    df = _load_clean_csv(csv_path)
    date_col = df.index.name or "Date"

    # validasi: hanya numeric
    num_cols = set(_numeric_columns(df))

    # bersihkan duplikat & kolom tanggal
    tset = [c for c in dict.fromkeys(req.targets) if c in df.columns and c in num_cols]
    eset = [c for c in dict.fromkeys(req.exog) if c in df.columns and c in num_cols]

    # larang overlap & singkirkan tanggal (index)
    deny = set(tset)
    eset = [c for c in eset if c not in deny]

    if not tset:
        raise HTTPException(status_code=400, detail="Pilih minimal satu TARGET (kolom numerik).")

    # simpan y & X (boleh kosong untuk X)
    y_df = df[tset].copy()
    y_df.to_csv(ycsv)

    if eset:
        X_df = df[eset].copy()
        X_df.to_csv(Xcsv)
    else:
        # hapus X.csv jika ada sebelumnya agar konsisten
        if Xcsv.exists():
            Xcsv.unlink()

    payload = {
        "dataset_id": dataset_id,
        "date_col": date_col,
        "targets": tset,
        "exog": eset,
        "files": {
            "y_csv": f"/dataset/{dataset_id}/matrix/y",
            "X_csv": f"/dataset/{dataset_id}/matrix/X" if eset else None
        }
    }
    _save_selection(sel_path, payload)
    return SelectResponse(**payload)

@app.get("/dataset/{dataset_id}/matrix/y")
def download_y(dataset_id: str):
    ddir, csv_path, sel_path, ycsv, Xcsv = _dataset_paths(dataset_id)
    if not ycsv.exists():
        raise HTTPException(status_code=404, detail="y.csv belum tersedia. Buat selection dulu.")
    return FileResponse(ycsv, media_type="text/csv", filename="y.csv")

@app.get("/dataset/{dataset_id}/matrix/X")
def download_X(dataset_id: str):
    ddir, csv_path, sel_path, ycsv, Xcsv = _dataset_paths(dataset_id)
    if not Xcsv.exists():
        raise HTTPException(status_code=404, detail="X.csv belum tersedia (mungkin EXOG kosong).")
    return FileResponse(Xcsv, media_type="text/csv", filename="X.csv")

# ----------------- Endpoint: Checklist UI sederhana -----------------
@app.get("/ui/select/{dataset_id}", response_class=HTMLResponse)
def ui_select(dataset_id: str):
    ddir, csv_path, sel_path, ycsv, Xcsv = _dataset_paths(dataset_id)
    try:
        df = _load_clean_csv(csv_path)
    except HTTPException as e:
        return HTMLResponse(f"<h3>Error: {e.detail}</h3>", status_code=e.status_code)

    date_col = df.index.name or "Date"
    num_cols = df.select_dtypes(include="number").columns.tolist()
    all_cols = df.columns.tolist()

    rows = []
    for c in all_cols:
        is_num = c in num_cols
        badge = '<span class="badge badge-numeric">Numeric</span>' if is_num else '<span class="badge badge-non-numeric">Non-Numeric</span>'
        dis = "" if is_num else "disabled"
        rows.append(
            f'<tr data-col="{c}" class="table-row">'
            f'<td class="col-name"><span class="col-icon">📊</span><strong>{c}</strong></td>'
            f'<td class="col-type">{badge}</td>'
            f'<td class="col-checkbox"><input type="checkbox" name="target" value="{c}" {dis} class="checkbox-target"></td>'
            f'<td class="col-checkbox"><input type="checkbox" name="exog" value="{c}" {dis} class="checkbox-exog"></td>'
            f'</tr>'
        )
    rows_html = "\n".join(rows)

    html_template = """
<!doctype html>
<html lang="id">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Pilih Variabel - PRISM Analytics</title>
<style>
:root {
  --primary: #2563eb;
  --primary-dark: #1e40af;
  --success: #16a34a;
  --warning: #ea580c;
  --danger: #dc2626;
  --gray-50: #f9fafb;
  --gray-100: #f3f4f6;
  --gray-200: #e5e7eb;
  --gray-600: #4b5563;
  --gray-800: #1f2937;
  --shadow: 0 1px 3px 0 rgba(0,0,0,0.1), 0 1px 2px -1px rgba(0,0,0,0.1);
  --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.1), 0 4px 6px -4px rgba(0,0,0,0.1);
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  min-height: 100vh;
  padding: 24px;
  color: var(--gray-800);
}

.container {
  max-width: 1200px;
  margin: 0 auto;
  background: white;
  border-radius: 16px;
  box-shadow: var(--shadow-lg);
  overflow: hidden;
}

.header {
  background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
  color: white;
  padding: 32px;
}

.header h1 {
  font-size: 28px;
  font-weight: 700;
  margin-bottom: 8px;
  display: flex;
  align-items: center;
  gap: 12px;
}

.header-meta {
  font-size: 14px;
  opacity: 0.9;
  display: flex;
  gap: 24px;
  flex-wrap: wrap;
  margin-top: 12px;
}

.meta-item {
  display: flex;
  align-items: center;
  gap: 8px;
  background: rgba(255,255,255,0.15);
  padding: 6px 12px;
  border-radius: 8px;
  backdrop-filter: blur(10px);
}

.toolbar {
  background: var(--gray-50);
  padding: 20px 32px;
  border-bottom: 1px solid var(--gray-200);
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 12px;
}

.toolbar-left {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.toolbar-right {
  display: flex;
  gap: 8px;
}

.btn {
  padding: 10px 20px;
  border-radius: 8px;
  border: none;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  text-decoration: none;
}

.btn:hover {
  transform: translateY(-1px);
  box-shadow: var(--shadow);
}

.btn:active {
  transform: translateY(0);
}

.btn-primary {
  background: var(--primary);
  color: white;
}

.btn-primary:hover {
  background: var(--primary-dark);
}

.btn-outline {
  background: white;
  color: var(--gray-600);
  border: 1px solid var(--gray-200);
}

.btn-outline:hover {
  background: var(--gray-50);
  border-color: var(--gray-600);
}

.btn-success {
  background: var(--success);
  color: white;
}

.btn-secondary {
  background: var(--gray-600);
  color: white;
}

.table-container {
  padding: 32px;
  overflow-x: auto;
}

table {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  font-size: 14px;
}

thead th {
  background: var(--gray-100);
  padding: 12px 16px;
  text-align: left;
  font-weight: 600;
  color: var(--gray-800);
  border-bottom: 2px solid var(--gray-200);
  position: sticky;
  top: 0;
  z-index: 10;
}

thead th:first-child {
  border-radius: 8px 0 0 0;
}

thead th:last-child {
  border-radius: 0 8px 0 0;
}

.table-row {
  transition: background 0.15s;
  border-bottom: 1px solid var(--gray-100);
}

.table-row:hover {
  background: var(--gray-50);
}

.table-row td {
  padding: 14px 16px;
}

.col-name {
  font-weight: 500;
  display: flex;
  align-items: center;
  gap: 8px;
}

.col-icon {
  font-size: 18px;
}

.col-type {
  text-align: center;
}

.col-checkbox {
  text-align: center;
}

.badge {
  display: inline-block;
  padding: 4px 12px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.badge-numeric {
  background: #dbeafe;
  color: #1e40af;
}

.badge-non-numeric {
  background: #fce7f3;
  color: #9f1239;
}

input[type="checkbox"] {
  width: 20px;
  height: 20px;
  cursor: pointer;
  accent-color: var(--primary);
}

input[type="checkbox"]:disabled {
  opacity: 0.3;
  cursor: not-allowed;
}

.status {
  padding: 20px 32px;
  border-top: 1px solid var(--gray-200);
  background: var(--gray-50);
}

.status-message {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 16px;
  border-radius: 8px;
  font-size: 14px;
}

.status-success {
  background: #dcfce7;
  color: #166534;
  border: 1px solid #86efac;
}

.status-error {
  background: #fee2e2;
  color: #991b1b;
  border: 1px solid #fca5a5;
}

.status-icon {
  font-size: 20px;
  flex-shrink: 0;
}

.action-buttons {
  margin-top: 16px;
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.loading {
  display: inline-block;
  width: 16px;
  height: 16px;
  border: 2px solid rgba(255,255,255,0.3);
  border-radius: 50%;
  border-top-color: white;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.summary {
  padding: 16px 32px;
  background: var(--gray-50);
  border-top: 1px solid var(--gray-200);
  display: flex;
  gap: 32px;
  flex-wrap: wrap;
}

.summary-item {
  display: flex;
  align-items: center;
  gap: 12px;
}

.summary-icon {
  width: 40px;
  height: 40px;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 20px;
}

.summary-icon-target {
  background: #dbeafe;
  color: #1e40af;
}

.summary-icon-exog {
  background: #ddd6fe;
  color: #6b21a8;
}

.summary-text {
  display: flex;
  flex-direction: column;
}

.summary-label {
  font-size: 12px;
  color: var(--gray-600);
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.summary-count {
  font-size: 24px;
  font-weight: 700;
  color: var(--gray-800);
}

@media (max-width: 768px) {
  body { padding: 12px; }
  .header { padding: 20px; }
  .toolbar { padding: 16px 20px; }
  .table-container { padding: 20px; }
  .header h1 { font-size: 22px; }
}
</style>
</head>
<body>
<div class="container">
  <!-- Header -->
  <div class="header">
    <h1>
      <span>📊</span> Pilih Variabel Dataset
    </h1>
    <div class="header-meta">
      <div class="meta-item">
        <span>🆔</span>
        <span><strong>Dataset:</strong> DATASET_PLACEHOLDER</span>
      </div>
      <div class="meta-item">
        <span>📅</span>
        <span><strong>Index:</strong> DATECOL_PLACEHOLDER</span>
      </div>
      <div class="meta-item">
        <span>📈</span>
        <span><strong>Total Kolom:</strong> <span id="totalCols">0</span></span>
      </div>
    </div>
  </div>

  <!-- Toolbar -->
  <div class="toolbar">
    <div class="toolbar-left">
      <button class="btn btn-outline" id="selectAllExog">
        <span>✅</span> Pilih Semua Exog
      </button>
      <button class="btn btn-outline" id="clearAll">
        <span>🗑️</span> Clear All
      </button>
    </div>
    <div class="toolbar-right">
      <button class="btn btn-success" id="save">
        <span>💾</span> Simpan Pilihan
      </button>
    </div>
  </div>

  <!-- Summary -->
  <div class="summary" id="summary">
    <div class="summary-item">
      <div class="summary-icon summary-icon-target">🎯</div>
      <div class="summary-text">
        <div class="summary-label">Target</div>
        <div class="summary-count" id="targetCount">0</div>
      </div>
    </div>
    <div class="summary-item">
      <div class="summary-icon summary-icon-exog">🔧</div>
      <div class="summary-text">
        <div class="summary-label">Exogenous</div>
        <div class="summary-count" id="exogCount">0</div>
      </div>
    </div>
  </div>

  <!-- Table -->
  <div class="table-container">
    <table id="tbl">
      <thead>
        <tr>
          <th>Kolom</th>
          <th>Tipe</th>
          <th style="text-align:center;">Target (Y)</th>
          <th style="text-align:center;">Exogenous (X)</th>
        </tr>
      </thead>
      <tbody>
ROWS_PLACEHOLDER
      </tbody>
    </table>
  </div>

  <!-- Status -->
  <div class="status" id="status"></div>
</div>

<script>
// Configuration
const BASE_URL = window.location.origin;
const PREFIX = "/select";
const DATASET_ID = "DATASET_PLACEHOLDER";

const API = {
  columns: BASE_URL + PREFIX + "/dataset/" + DATASET_ID + "/columns",
  selection: BASE_URL + PREFIX + "/dataset/" + DATASET_ID + "/selection",
  select: BASE_URL + PREFIX + "/dataset/" + DATASET_ID + "/select",
  ycsv: BASE_URL + PREFIX + "/dataset/" + DATASET_ID + "/matrix/y",
  Xcsv: BASE_URL + PREFIX + "/dataset/" + DATASET_ID + "/matrix/X"
};

// Update summary counts
function updateSummary() {
  const targetCount = document.querySelectorAll('input[name="target"]:checked').length;
  const exogCount = document.querySelectorAll('input[name="exog"]:checked').length;
  document.getElementById('targetCount').textContent = targetCount;
  document.getElementById('exogCount').textContent = exogCount;
}

// Mutual exclusion: target vs exog
const tbl = document.getElementById('tbl');
tbl.addEventListener('change', function(e) {
  const row = e.target.closest('tr');
  if (!row) return;
  
  if (e.target.name === 'target' && e.target.checked) {
    const ex = row.querySelector('input[name="exog"]');
    if (ex) ex.checked = false;
  }
  
  if (e.target.name === 'exog' && e.target.checked) {
    const tg = row.querySelector('input[name="target"]');
    if (tg) tg.checked = false;
  }
  
  updateSummary();
});

// Apply selection
function applySelection(targets, exog) {
  const setChecked = function(name, list) {
    const want = new Set(list || []);
    document.querySelectorAll('input[name="' + name + '"]').forEach(function(cb) {
      cb.checked = want.has(cb.value);
    });
  };
  setChecked('target', targets);
  setChecked('exog', exog);
  updateSummary();
}

// Select all exog button
document.getElementById('selectAllExog').onclick = function() {
  document.querySelectorAll('tbody tr').forEach(function(r) {
    const ex = r.querySelector('input[name="exog"]');
    const tg = r.querySelector('input[name="target"]');
    if (ex && !ex.disabled) {
      ex.checked = true;
      if (tg) tg.checked = false;
    }
  });
  updateSummary();
};

// Clear all button
document.getElementById('clearAll').onclick = function() {
  document.querySelectorAll('input[type="checkbox"]').forEach(function(cb) {
    cb.checked = false;
  });
  updateSummary();
};

// Show status message
function showStatus(type, message, actions) {
  const box = document.getElementById('status');
  const icon = type === 'success' ? '✅' : '❌';
  const className = type === 'success' ? 'status-success' : 'status-error';
  
  let html = '<div class="status-message ' + className + '">' +
    '<span class="status-icon">' + icon + '</span>' +
    '<div style="flex:1;">' +
      '<div>' + message + '</div>';
  
  if (actions) {
    html += '<div class="action-buttons">' + actions + '</div>';
  }
  
  html += '</div></div>';
  box.innerHTML = html;
}

// Save button
document.getElementById('save').onclick = async function() {
  const targets = Array.from(document.querySelectorAll('input[name="target"]:checked')).map(function(e) { return e.value; });
  const exog = Array.from(document.querySelectorAll('input[name="exog"]:checked')).map(function(e) { return e.value; });
  
  if (targets.length === 0) {
    showStatus('error', 'Pilih minimal 1 variabel Target terlebih dahulu!');
    return;
  }
  
  const btn = document.getElementById('save');
  const originalHTML = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = '<span class="loading"></span> Menyimpan...';
  
  try {
    const res = await fetch(API.select, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ targets: targets, exog: exog })
    });
    
    if (!res.ok) {
      const err = await res.json().catch(function() { return {detail:'Gagal menyimpan'}; });
      showStatus('error', err.detail || 'Gagal menyimpan pilihan variabel.');
      btn.disabled = false;
      btn.innerHTML = originalHTML;
      return;
    }
    
    const data = await res.json();
    
    // Success message with actions
    const actions = 
      '<a href="' + API.ycsv + '" target="_blank" class="btn btn-outline" style="display:inline-flex;">📥 Unduh y.csv</a>' +
      (exog.length ? '<a href="' + API.Xcsv + '" target="_blank" class="btn btn-outline" style="display:inline-flex;">📥 Unduh X.csv</a>' : '') +
      '<button onclick="nextStep()" class="btn btn-success">➡️ Lanjut ke VAR Macro</button>' +
      '<button onclick="backToDashboard()" class="btn btn-secondary">🏠 Dashboard</button>';
    
    showStatus('success', 
      '✨ Pilihan berhasil disimpan! Target: <strong>' + targets.length + '</strong> variabel, Exogenous: <strong>' + exog.length + '</strong> variabel.',
      actions
    );
    
    btn.disabled = false;
    btn.innerHTML = originalHTML;
    
  } catch (e) {
    console.error("Error:", e);
    showStatus('error', 'Network error: ' + e.message);
    btn.disabled = false;
    btn.innerHTML = originalHTML;
  }
};

// Navigation functions
function nextStep() {
  window.location.href = 'http://localhost:8000/var/ui/' + DATASET_ID;
}

function backToDashboard() {
  if (window.opener) {
    window.close();
  } else {
    window.location.href = 'http://localhost:8080/dashboard';
  }
}

// Initialize
(async function init() {
  // Count total columns
  const totalCols = document.querySelectorAll('tbody tr').length;
  document.getElementById('totalCols').textContent = totalCols;
  
  // Load existing selection if any
  try {
    const res = await fetch(API.selection);
    if (res.ok) {
      const data = await res.json();
      console.log("Loaded existing selection:", data);
      applySelection(data.targets, data.exog);
      showStatus('success', '📂 Pilihan sebelumnya berhasil dimuat.');
    }
  } catch (e) {
    console.log("No existing selection found");
  }
})();
</script>
</body>
</html>
"""
    
    # Replace placeholders
    html = (html_template
            .replace("DATASET_PLACEHOLDER", dataset_id)
            .replace("DATECOL_PLACEHOLDER", str(date_col))
            .replace("ROWS_PLACEHOLDER", rows_html))
    
    return HTMLResponse(html)