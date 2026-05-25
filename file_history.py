# file_history.py
# FastAPI — File history & PDF summary for PRISM datasets

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
import humanize
import mimetypes
import math
import os

# PDF (reportlab)
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

app = FastAPI(title="PRISM — File History", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
OUT_DIR  = BASE_DIR / "generated"
DATA_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

# -------- Helpers --------

def _assert_dataset(dataset_id: str) -> Path:
    root = DATA_DIR / dataset_id
    if not root.exists():
        raise HTTPException(404, f"Dataset '{dataset_id}' tidak ditemukan di {DATA_DIR}")
    return root

def _infer_category(p: Path, dataset_id: str) -> str:
    """Kategori sederhana berdasarkan path / nama file."""
    low = str(p).lower()
    if f"{os.sep}{dataset_id}{os.sep}risk{os.sep}" in low or low.endswith("_base_forecast.csv"):
        return "risk"
    if f"{os.sep}{dataset_id}{os.sep}shock{os.sep}" in low:
        return "shock"
    if f"{os.sep}{dataset_id}{os.sep}esg{os.sep}" in low:
        return "esg"
    if f"{os.sep}{dataset_id}{os.sep}plots{os.sep}" in low or low.endswith(".png"):
        return "plots"
    if low.endswith("timeseries_clean.csv"):
        return "cleaned"
    if "X_future_" in p.name:
        return "macro"
    if p.suffix.lower() in [".json"]:
        return "meta"
    return "other"

def _scan_files(dataset_id: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    root = _assert_dataset(dataset_id)
    items: List[Dict[str, Any]] = []
    total_size = 0
    latest_mtime = 0

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        try:
            stat = p.stat()
        except Exception:
            continue
        rel = str(p.relative_to(DATA_DIR))
        size = stat.st_size
        mtime = stat.st_mtime
        latest_mtime = max(latest_mtime, mtime)
        total_size += size
        items.append({
            "name": p.name,
            "rel": rel,                        # e.g. "<dataset>/esg/X_future_mean.csv"
            "path": f"/files/raw/{dataset_id}?name={rel}",
            "size_bytes": size,
            "size_human": humanize.naturalsize(size, binary=False),
            "modified": datetime.fromtimestamp(mtime).isoformat(timespec="seconds"),
            "category": _infer_category(p, dataset_id),
            "ext": p.suffix.lower(),
        })

    # ringkas per kategori
    by_cat: Dict[str, Dict[str, Any]] = {}
    for it in items:
        c = it["category"]
        by_cat.setdefault(c, {"count": 0, "bytes": 0})
        by_cat[c]["count"] += 1
        by_cat[c]["bytes"] += it["size_bytes"]

    summary = {
        "dataset_id": dataset_id,
        "root": str(root.relative_to(DATA_DIR)),
        "file_count": len(items),
        "total_bytes": total_size,
        "total_size": humanize.naturalsize(total_size, binary=False),
        "last_modified": datetime.fromtimestamp(latest_mtime).isoformat(timespec="seconds") if latest_mtime else None,
        "by_category": {
            k: {"count": v["count"], "size": humanize.naturalsize(v["bytes"], binary=False)}
            for k, v in sorted(by_cat.items(), key=lambda x: x[0])
        }
    }
    # sort tampilan default: kategori → waktu terbaru
    items.sort(key=lambda x: (x["category"], x["modified"]), reverse=True)
    return items, summary

def _safe_file_path(dataset_id: str, rel: str) -> Path:
    """Validasi agar rel path tetap di bawah DATA_DIR/dataset_id."""
    base = DATA_DIR / dataset_id
    path = DATA_DIR / rel
    try:
        path.resolve().relative_to(base.resolve())
    except Exception:
        raise HTTPException(400, "Path di luar dataset.")
    if not path.exists():
        raise HTTPException(404, "File tidak ditemukan.")
    return path

# -------- API: JSON --------

@app.get("/files/{dataset_id}")
def list_files(dataset_id: str):
    items, summary = _scan_files(dataset_id)
    return JSONResponse({"summary": summary, "files": items})

# -------- API: Raw file download --------

@app.get("/files/raw/{dataset_id}")
def get_raw(dataset_id: str, name: str):
    path = _safe_file_path(dataset_id, name)
    media, _ = mimetypes.guess_type(path.name)
    media = media or "application/octet-stream"
    return FileResponse(path, filename=path.name, media_type=media)

# -------- API: PDF summary --------

def _build_pdf(dataset_id: str, summary: Dict[str, Any], files: List[Dict[str, Any]]) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"file_history_{dataset_id}_{ts}.pdf"

    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=1.5*cm, rightMargin=1.5*cm, topMargin=1.5*cm, bottomMargin=1.5*cm
    )
    styles = getSampleStyleSheet()
    title = styles["Title"]; title.textColor = colors.HexColor("#1e293b")
    normal = styles["Normal"]
    h2 = styles["Heading2"]; h2.textColor = colors.HexColor("#111827")
    h3 = styles["Heading3"]; h3.textColor = colors.HexColor("#111827")

    flow: List[Any] = []
    flow.append(Paragraph(f"PRISM · File History — {dataset_id}", title))
    flow.append(Spacer(1, 0.25*cm))

    # Summary block
    flow.append(Paragraph("Ringkasan Dataset", h2))
    kv_rows = [
        ["Dataset ID", summary["dataset_id"]],
        ["Root", summary["root"]],
        ["Jumlah File", f"{summary['file_count']}"],
        ["Total Size", summary["total_size"]],
        ["Last Modified", summary["last_modified"] or "-"],
    ]
    cat_text = ", ".join([f"{k}: {v['count']} file ({v['size']})" for k, v in summary["by_category"].items()]) or "-"
    kv_rows.append(["Kategori", cat_text])

    t = Table(kv_rows, hAlign="LEFT", colWidths=[4.5*cm, None])
    t.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("TEXTCOLOR", (0,0), (0,-1), colors.HexColor("#64748b")),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    flow.append(t)
    flow.append(Spacer(1, 0.3*cm))

    # Files table (paged)
    flow.append(Paragraph("Daftar File", h2))
    headers = ["Category", "File", "Size", "Modified", "Type"]
    data_rows = [headers]
    for f in files:
        data_rows.append([
            f["category"],
            f["rel"],
            f["size_human"],
            f["modified"].replace("T", " "),
            f["ext"] or "-",
        ])

    # Break jadi beberapa tabel jika sangat panjang
    page_chunk = 45  # baris per tabel (kira-kira)
    for i in range(0, len(data_rows), page_chunk):
        chunk = data_rows[i:i+page_chunk]
        tbl = Table(chunk, repeatRows=1, colWidths=[2.5*cm, 8.0*cm, 2.0*cm, 3.5*cm, 1.5*cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e5e7eb")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.HexColor("#111827")),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,0), 9),
            ("ALIGN", (2,1), (2,-1), "RIGHT"),
            ("ALIGN", (4,1), (4,-1), "CENTER"),
            ("FONTNAME", (0,1), (-1,-1), "Helvetica"),
            ("FONTSIZE", (0,1), (-1,-1), 8),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
            ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#cbd5e1")),
        ]))
        flow.append(tbl)
        flow.append(Spacer(1, 0.2*cm))

    # Footer
    flow.append(Spacer(1, 0.5*cm))
    gen = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    flow.append(Paragraph(f"Generated at {gen}", normal))

    doc.build(flow)
    return out_path

@app.get("/files/pdf/{dataset_id}")
def files_pdf(dataset_id: str):
    files, summary = _scan_files(dataset_id)
    if summary["file_count"] == 0:
        raise HTTPException(404, "Tidak ada file untuk dataset ini.")
    pdf_path = _build_pdf(dataset_id, summary, files)
    return FileResponse(pdf_path, filename=pdf_path.name, media_type="application/pdf")

# -------- UI --------
@app.get("/files/ui/{dataset_id}", response_class=HTMLResponse)
def files_ui(dataset_id: str):
    html = """
<!doctype html>
<html lang="id">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>PRISM · File History</title>
<style>
  :root {
    --bg:#0b1220; --card:#101827; --ink:#e5e7eb; --muted:#9ca3af; --bd:#1f2937; --pri:#60a5fa;
  }
  * { box-sizing: border-box }
  body {
    margin:0; background:linear-gradient(180deg,#0b1220,#0b1220 60%,#0e1628);
    color:var(--ink); font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,Arial;
  }
  .wrap { max-width:1120px; margin:18px auto; padding:0 12px }
  .hdr {
    display:flex; align-items:center; justify-content:space-between; gap:12px;
    padding:12px 14px; background:rgba(255,255,255,0.03); border:1px solid var(--bd);
    border-radius:14px;
  }
  .hdr h1 { font-size:16px; margin:0 }
  .meta { color:var(--muted); font-size:12px }
  .row { display:flex; gap:8px; flex-wrap:wrap }
  .tag { display:inline-flex;align-items:center;gap:6px;padding:6px 10px;border:1px solid var(--bd);border-radius:999px;background:#0f192e;font-size:12px;color:var(--ink);text-decoration:none }
  table { width:100%; border-collapse:collapse; font-size:13px }
  th,td { padding:6px 8px; border-bottom:1px solid var(--bd) }
  th { color:var(--muted); text-align:left; font-weight:600 }
  .right { text-align:right }
  .card { background:rgba(255,255,255,0.03); border:1px solid var(--bd); border-radius:14px; padding:12px; margin-top:12px }
  .pill { padding:2px 8px; border:1px solid var(--bd); border-radius:999px; background:#0c1427; color:#cbd5e1; margin-right:6px }
</style>
</head>
<body>
<div class="wrap">
  <div class="hdr">
    <div>
      <h1>PRISM · File History</h1>
      <div class="meta" id="meta"></div>
    </div>
    <div class="row">
      <a id="pdf-link" class="tag" href="#" target="_blank">Download PDF</a>
      <a id="json-link" class="tag" href="#" target="_blank">JSON</a>
    </div>
  </div>

  <div class="card" id="summary"></div>
  <div class="card">
    <div style="display:flex;align-items:center;justify-content:space-between;">
      <div style="font-weight:600;">Files</div>
      <div>
        <label class="meta">Filter category:</label>
        <select id="cat">
          <option value="">All</option>
        </select>
      </div>
    </div>
    <table id="tbl">
      <thead>
        <tr><th>Category</th><th>File</th><th class="right">Size</th><th>Modified</th><th>Action</th></tr>
      </thead>
      <tbody></tbody>
    </table>
  </div>
</div>

<script>
const BASE = window.location.origin;
const ds = window.location.pathname.split('/').pop();
document.getElementById('json-link').href = `${BASE}/files/${ds}`;
document.getElementById('pdf-link').href  = `${BASE}/files/pdf/${ds}`;

async function load() {
  const res = await fetch(`${BASE}/files/${ds}`);
  if(!res.ok){ document.getElementById('meta').textContent = 'Gagal memuat data.'; return; }
  const data = await res.json();
  const S = data.summary;
  const rows = data.files || [];

  document.getElementById('meta').textContent = `Dataset: ${S.dataset_id}`;
  const sm = document.getElementById('summary');
  sm.innerHTML = `
    <div style="display:grid;grid-template-columns:160px 1fr;gap:6px 12px;">
      <div class="meta">Dataset ID</div><div>${S.dataset_id}</div>
      <div class="meta">Root</div><div>${S.root}</div>
      <div class="meta">Last Modified</div><div>${S.last_modified || '-'}</div>
      <div class="meta">Files</div><div>${S.file_count} file · ${S.total_size}</div>
      <div class="meta">By Category</div>
      <div>${
        Object.entries(S.by_category || {}).map(([k,v]) =>
          `<span class="pill">${k} · ${v.count} · ${v.size}</span>`
        ).join(' ')
      }</div>
    </div>`;

  const cats = Array.from(new Set(rows.map(r => r.category))).sort();
  const sel = document.getElementById('cat');
  cats.forEach(c => {
    const o = document.createElement('option');
    o.value = c; o.textContent = c; sel.appendChild(o);
  });
  sel.addEventListener('change', () => render(rows, sel.value));

  render(rows, sel.value);
}

function render(rows, cat) {
  const tb = document.querySelector('#tbl tbody');
  tb.innerHTML = '';
  rows.filter(r => !cat || r.category === cat).forEach(r => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${r.category}</td>
      <td>${r.rel}</td>
      <td class="right">${r.size_human}</td>
      <td>${r.modified.replace('T',' ')}</td>
      <td><a href="${r.path}" target="_blank">Open</a></td>`;
    tb.appendChild(tr);
  });
}

load();
</script>
</body>
</html>
"""
    return HTMLResponse(html)


@app.get("/")
def root():
    return JSONResponse({"message": "PRISM File History ready", "status": "ok"})
