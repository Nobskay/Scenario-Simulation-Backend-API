# Forecasting Backend API

This Backend API is a modular FastAPI-based backend for macroeconomic forecasting, recession probability modeling, ESG scenario generation, and dataset management.

The system is designed to support end-to-end economic scenario analysis workflows, including:

* Dataset upload & preprocessing
* Variable selection
* VAR(1) macroeconomic simulation
* Recession probability estimation
* ESG scenario generation
* File history & audit tracking

---

# Features

## 1. Data Upload & Cleaning

Module: `load.py`

Features:

* Upload CSV/XLSX datasets
* Automatic date column detection
* Data cleaning & preprocessing
* Time-series visualization
* Audit reporting
* Multi-panel plotting for numeric variables

---

## 2. Variable Selection

Module: `pilih_var.py`

Features:

* Select target variables
* Select exogenous (macro) variables
* Export `y.csv` and `X.csv`
* HTML checklist UI
* Save configuration into `selection.json`

---

## 3. VAR(1) Macroeconomic Simulation

Module: `var_macro.py`

Features:

* VAR(1) Monte Carlo simulation
* Fan chart generation
* Forecast percentile summaries
* Historical + forecast visualization
* Automatic fallback to AutoReg for single-variable models

---

## 4. Recession Probability Modeling

Module: `prob_resesi.py`

Features:

* Logistic regression recession model
* Recession probability forecasting
* Visualization of recession probability paths
* Mean path estimation from VAR output

---

## 5. ESG Scenario Generator

Module: `model_esg.py`

Features:

* Asset-to-macro mapping
* ESG Monte Carlo simulation
* Stress testing outputs
* Validation metrics
* Historical + projected ESG visualization
* SoA export generation

---

## 6. File History & Audit

Module: `file_history.py`

Features:

* Dataset file tracking
* Audit metadata
* Categorized file scanning
* Dataset manifest generation

---

# Project Structure

```bash
backend/
│
├── main.py                # Main API gateway
├── load.py                # Upload & cleaning module
├── pilih_var.py           # Variable selection module
├── var_macro.py           # VAR(1) simulation module
├── prob_resesi.py         # Recession probability module
├── model_esg.py           # ESG scenario module
├── file_history.py        # File audit/history module
├── requirements.txt       # Python dependencies
│
└── data/
    └── <dataset_id>/
        ├── timeseries_clean.csv
        ├── selection.json
        ├── var1_macro/
        ├── resesi/
        ├── esg/
        └── ...
```

---

# Installation

## 1. Clone Repository

```bash
git clone <your-repository-url>
cd backend
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the API

Run the main gateway:

```bash
uvicorn main:app --reload --port 8000
```

API will be available at:

```bash
http://127.0.0.1:8000
```

---

# API Documentation

Each module has its own FastAPI documentation.

| Module             | Docs URL        |
| ------------------ | --------------- |
| Load/Clean         | `/load/docs`    |
| Variable Selection | `/select/docs`  |
| VAR Macro          | `/var/docs`     |
| Recession Model    | `/resesi/docs`  |
| ESG Model          | `/esg/docs`     |
| File History       | `/history/docs` |

Example:

```bash
http://127.0.0.1:8000/load/docs
```

---

# Workflow Pipeline

## Step 1 — Upload Dataset

Upload CSV/XLSX dataset through:

```http
POST /load/upload
```

Output:

* Cleaned dataset
* Time-series plots
* Dataset ID

---

## Step 2 — Select Variables

Choose:

* Target variables
* Exogenous variables

Endpoints:

```http
GET  /select/dataset/{dataset_id}/columns
POST /select/dataset/{dataset_id}/select
```

---

## Step 3 — Run VAR(1) Simulation

Generate macroeconomic forecasts:

```http
POST /var/run
```

Outputs:

* Monte Carlo simulations
* Fan charts
* Percentile summaries

---

## Step 4 — Run Recession Probability Model

```http
POST /resesi/run
```

Outputs:

* Recession probability forecast
* Probability plots

---

## Step 5 — Generate ESG Scenarios

```http
POST /esg/run
```

Outputs:

* ESG scenario forecasts
* Asset stress projections
* Validation metrics

---

# Generated Outputs

The backend automatically generates outputs inside:

```bash
data/<dataset_id>/
```

Possible outputs include:

```bash
selection.json
summary_percentiles.csv
prob_resesi.csv
fan_chart_grid.png
historis_plus_fan.png
esg_mc_grid.png
manifest.json
```

---

# Technologies Used

* Python
* FastAPI
* Pandas
* NumPy
* Matplotlib
* Statsmodels
* Scikit-learn
* Uvicorn

---

# Dependencies

Main dependencies used in this project:

```txt
fastapi
uvicorn
pandas
numpy
matplotlib
statsmodels
scikit-learn
python-multipart
openpyxl
xlrd
```

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

# Example Development Setup

```bash
# Create virtual environment
python -m venv venv

# Activate environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn main:app --reload --port 8000
```

---

# Notes

* The backend is modular and each module can run independently.
* Generated datasets are stored locally under the `data/` directory.
* CORS is enabled for development purposes.
* The system is designed for economic forecasting and ESG scenario analysis workflows.

---

# Future Improvements

Potential future enhancements:

* Authentication & authorization
* PostgreSQL integration
* Docker deployment
* Cloud storage integration
* Real-time dashboard integration
* Automated model retraining
* Advanced stress-testing scenarios

---

# License

This project is intended for educational, research, and internal development purposes.

---

# Author

Developed by Noorharsy Imanullah.
