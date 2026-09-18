# 🚦 Indian Urban Mobility Intelligence

An interactive data analytics and machine learning dashboard for analyzing road accident patterns in India using official Government of India road-accident datasets.

The project combines data cleaning, exploratory data analysis (EDA), visualization, state-level comparison, road and junction analysis, and a machine learning model for estimating State/UT-level reported road accidents for 2019 from historical accident counts.

> **Important scope:** The current project is primarily a **road accident and road-safety analytics system**. It is **not a real-time traffic congestion, speed, density, or vehicle-flow prediction system** because the datasets currently used do not contain live traffic-flow measurements.

---

## 1. Project Objectives

The project aims to:

- Analyze long-term road accident trends in India.
- Compare reported road accidents across States and Union Territories.
- Study reported accident categories associated with contributing factors.
- Analyze road characteristics associated with reported accidents.
- Analyze junction types and traffic-control categories.
- Explore reported black-spot and high-accident-location information.
- Build an interactive Streamlit dashboard.
- Develop a State/UT-level machine learning model using historical accident data.
- Present the complete workflow in a reproducible and understandable format.

---

## 2. Main Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Streamlit
- Joblib
- Jupyter Notebook
- Git / GitHub

---

## 3. Data Sources

### Primary Source

The primary data source is the **Government of India Open Government Data (OGD) Platform**, specifically the **Road Accidents in India 2019** catalogue published by the Ministry of Road Transport & Highways (MoRTH).

Official catalogue:

https://www.data.gov.in/catalog/road-accidents-india-2019

The catalogue contains multiple annual road-accident resources covering accident totals, State/UT comparisons, causes/contributing factors, road features, junctions, traffic control, black spots and other accident-related classifications.

### Official MoRTH Report

The source report is:

**Road Accidents in India – 2019**  
Government of India  
Ministry of Road Transport & Highways  
Transport Research Wing

Official report:

https://morth.gov.in/sites/default/files/RA_Uploading.pdf

The report states that the Transport Research Wing compiles annual accident information based on information supplied by State and UT police departments using standardized formats.

---

## 4. Datasets Used in This Project

The downloaded CSV files are stored in `data/` and cleaned versions are stored in `data/processed/`.

### A1 — India Accident Trend

File:

`RA2019_A1_cleaned.csv`

Contains:

- Year
- Total road accidents
- Persons killed
- Persons injured
- Population
- Registered motor vehicles
- Road length
- Accident and fatality rates

Used for:

- Long-term national accident trend
- Accident, killed and injured KPI values

---

### A2 — State/UT Accident Statistics

File:

`RA2019_A2_cleaned.csv`

Contains State/UT-level accident information from 2016 to 2019 and related rates/shares.

Used for:

- State comparison
- Historical accident counts
- Machine learning dataset creation

Official OGD catalogue:

https://www.data.gov.in/catalog/road-accidents-india-2019

---

### A47 — Accident Causes / Contributing Factors

File:

`RA2019_A47_cleaned.csv`

Contains reported accident categories associated with factors such as:

- Over-speeding
- Drunken driving / consumption of alcohol and drugs
- Wrong-side related categories
- Jumping red light
- Mobile phone use
- Other categories

Used for:

- Accident cause/contributing-factor analysis
- Location-level comparison

---

### A48 — Road Features

File:

`RA2019_A48_cleaned.csv`

Contains accident classifications by road characteristics such as:

- Straight road
- Curved road
- Bridge
- Culvert
- Pot holes
- Steep grade
- Ongoing road works / under construction
- Others

Used for:

- Road-feature analysis

---

### A49 — Junction Types

File:

`RA2019_A49_cleaned.csv`

Contains reported accident classifications by junction type such as:

- T-junction
- Y-junction
- Four-arm junction
- Staggered junction
- Roundabout
- Others

Used for:

- Junction analysis

---

### A50 — Traffic Control

File:

`RA2019_A50_cleaned.csv`

Contains reported accident classifications according to traffic-control categories such as:

- Traffic light signal
- Police controlled
- Stop sign
- Flashing signal / blinker
- Uncontrolled
- Others

Used for:

- Traffic-control analysis

---

### A51 — Black Spots

File:

`RA2019_A51_cleaned.csv`

Contains State/UT information on black spots on National Highways reported by the States.

Used for:

- Black-spot information display

---

### A52 — High Accident Locations

File:

`RA2019_A52_cleaned.csv`

Contains location-level information including:

- State
- District / traffic unit
- Jurisdictional police station
- Highway / road number
- Location
- Kilometre information
- Accident counts for selected years
- Fatality counts for selected years

Used for:

- High-accident-location exploration

---

## 5. Data Collection and Preprocessing

The original CSV files were downloaded from the official OGD catalogue and placed in the project's `data/` directory.

The preprocessing script:

`src/process_traffic_data.py`

performs:

1. CSV loading with encoding fallback.
2. Column-name cleaning.
3. Text cleaning.
4. Numeric conversion where appropriate.
5. Removal of completely empty rows/columns.
6. Duplicate-row checking.
7. Saving cleaned datasets.
8. Generating a dataset summary.

Cleaned files are stored in:

`data/processed/`

Missing values were inspected rather than blindly replacing all missing values, because an unavailable/not-reported value should not automatically be treated as zero.

---

## 6. Exploratory Data Analysis

The notebook:

`notebooks/01_data_exploration.ipynb`

contains the main EDA work.

The analysis includes:

- India-wide accident trend.
- State/UT comparison.
- Reported accident categories.
- Road-feature analysis.
- Junction analysis.
- Traffic-control analysis.
- Black-spot information.
- High-accident-location exploration.

Important terminology:

The dashboard uses terms such as **reported accidents**, **reported accident categories**, and **reported classifications**. These categories should not automatically be interpreted as mutually exclusive causal explanations.

---

## 7. Machine Learning

### Prediction Task

The machine learning component estimates:

**State/UT-level reported road accidents for 2019**

using historical accident information from:

- 2016
- 2017
- 2018

### Features

The model uses six input features:

1. `Accidents_2016`
2. `Accidents_2017`
3. `Accidents_2018`
4. `Average_Accidents_2016_2018`
5. `Accident_Change_2018_vs_2016`
6. `Accident_Growth_2018_vs_2016`

Target:

`Accidents_2019`

### Feature Engineering

The feature-engineering script:

`src/feature_engineering.py`

creates:

- Three historical accident-count features.
- Three derived features:
  - Historical average.
  - 2018 minus 2016 change.
  - Percentage growth from 2016 to 2018.

The resulting dataset is:

`data/processed/ml_accident_dataset.csv`

The verified dataset contains 36 State/UT observations and 8 columns.

### Model

The existing training script:

`src/model.py`

compares:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

The model selected by the existing script was:

**Linear Regression**

based on the lowest RMSE in its current train/test split.

The trained model is saved as:

`models/accident_prediction_model.pkl`

---

## 8. Current Model Evaluation

The current recorded evaluation from the training script is:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 1074.75 | 1692.60 | 0.9925 |
| Random Forest | 2261.40 | 3909.94 | 0.9599 |
| Gradient Boosting | 1427.20 | 2652.36 | 0.9815 |

The current training/test split used:

- Training samples: 28
- Testing samples: 8

The Linear Regression model was selected based on the lowest RMSE.

### Important Interpretation

These metrics should be treated as **dataset-specific project results**, not evidence that the model is production-ready.

The ML dataset is small, with only 36 State/UT observations, and the target is one historical year. The current evaluation is therefore useful for demonstrating the ML workflow but should not be presented as a general-purpose national forecasting model.

---

## 9. Dashboard

The dashboard is implemented in:

`app.py`

Framework:

**Streamlit**

Current navigation:

1. ML Prediction
2. Accident Trends
3. State Analysis
4. Accident Causes
5. Road Features
6. Traffic Control
7. Junction Analysis
8. Black Spots
9. High Accident Locations
10. Data & Methodology
11. Overview

The ML page is intentionally placed first so the machine-learning component is clearly visible in the project demonstration.

---

## 10. Dashboard Features

### ML Prediction

- State/UT selection
- Automatic loading of actual 2016–2018 values
- Calculated features
- 2019 prediction
- Actual 2019 value
- Absolute error
- Actual vs predicted chart
- State-wise predictions
- Model evaluation table

### Accident Trends

- India-wide historical accident trend
- Persons killed trend

### State Analysis

- State/UT ranking
- Interactive top-N selection
- Tabular data

### Accident Causes

- Reported accident-factor categories
- Over-speeding location analysis

### Road Features

- Road-characteristic analysis

### Traffic Control

- Traffic-control category analysis

### Junction Analysis

- Junction-type analysis

### Black Spots

- State/UT black-spot information

### High Accident Locations

- Searchable location-level data

### Data & Methodology

- Dataset overview
- ML features
- Workflow
- Scope and limitations

### Overview

- National road-safety snapshot
- Project modules
- India-wide accident trend
- Project description

---

## 11. Project Structure

```text
Indian urban mobility traffic/
│
├── app.py
├── README.md
├── PROJECT_REPORT.md
│
├── data/
│   ├── RA2019_A1.csv
│   ├── RA2019_A2.csv
│   ├── RA2019_A47.csv
│   ├── RA2019_A48.csv
│   ├── RA2019_A49.csv
│   ├── RA2019_A50.csv
│   ├── RA2019_A51.csv
│   ├── RA2019_A52.csv
│   │
│   └── processed/
│       ├── RA2019_A1_cleaned.csv
│       ├── RA2019_A2_cleaned.csv
│       ├── RA2019_A47_cleaned.csv
│       ├── RA2019_A48_cleaned.csv
│       ├── RA2019_A49_cleaned.csv
│       ├── RA2019_A50_cleaned.csv
│       ├── RA2019_A51_cleaned.csv
│       ├── RA2019_A52_cleaned.csv
│       ├── dataset_summary.csv
│       └── ml_accident_dataset.csv
│
├── models/
│   ├── accident_prediction_model.pkl
│   ├── feature_names.txt
│   └── model_results.csv
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
└── src/
    ├── process_traffic_data.py
    ├── feature_engineering.py
    └── model.py
```

---

## 12. How to Run

### Create and activate virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install pandas numpy scikit-learn plotly streamlit joblib openpyxl matplotlib seaborn jupyter
```

### Process data

```powershell
python src\process_traffic_data.py
```

### Create ML dataset

```powershell
python src\feature_engineering.py
```

### Train model

```powershell
python src\model.py
```

### Start dashboard

```powershell
streamlit run app.py
```

---

## 13. Reproducibility

A fresh environment can reproduce the project by following:

1. Install Python.
2. Create the virtual environment.
3. Install dependencies.
4. Place the official CSV files in `data/`.
5. Run preprocessing.
6. Run feature engineering.
7. Train the model.
8. Launch Streamlit.

The trained model should be regenerated after changing the feature-engineering or model-training code.

---

## 14. Limitations

1. The primary datasets are road-accident and road-safety datasets.
2. They do not provide live traffic speed, traffic density or real-time congestion.
3. The ML dataset contains only 36 State/UT observations.
4. The current model predicts a historical 2019 target from historical State/UT accident counts.
5. The current evaluation uses a small train/test split.
6. High R² on this small dataset should not be interpreted as production-level forecasting performance.
7. Some source categories may represent classifications rather than mutually exclusive causal factors.
8. Missing source values are not automatically equivalent to zero.
9. The dashboard does not independently redefine the Government of India black-spot classification.

---

## 15. Future Scope

Future versions can add:

- Multi-year accident forecasting.
- Cross-validation and stronger model evaluation.
- Confidence / prediction intervals.
- Indian urban traffic-flow data.
- Vehicle counts.
- Speed and density measurements.
- Congestion prediction.
- Time-of-day analysis.
- Weather integration.
- GIS maps.
- Geospatial hotspot analysis.
- Computer vision using Indian traffic datasets.
- Live traffic APIs where legally and technically appropriate.
- More recent accident datasets.

---

## 16. References

1. Government of India Open Government Data Platform — Road Accidents in India 2019  
   https://www.data.gov.in/catalog/road-accidents-india-2019

2. Ministry of Road Transport & Highways — Road Accidents in India 2019  
   https://morth.gov.in/sites/default/files/RA_Uploading.pdf

3. Government of India OGD — State/UT-wise total number of road accidents in India from 2016 to 2019  
   https://www.data.gov.in/catalog/road-accidents-india-2019

4. Government of India OGD — Road Accidents in India classified according to various parameters  
   https://tn.data.gov.in/catalog/road-accidents-india-classified-according-various-parameters

5. IIT Madras RBCDSAI — Synchronized Multi Scale and Multi Sensor Traffic Data From Indian Urban Roads  
   https://rbcdsai.iitm.ac.in/projects/synchronized-multi-scale-and-multi-sensor-traffic-data-from-indian-urban-roads/

6. IEEE — ITD: Indian Traffic Dataset for Intelligent Transportation Systems  
   https://ieeexplore.ieee.org/document/10427394/

The IIT Madras and IEEE datasets are included as related research/future-scope references; they are **not claimed as source data for the current dashboard unless separately added to the project**.

---

## 17. License / Data Attribution

The project code is intended as an academic portfolio project.

The underlying government data remains subject to the Government Open Data License - India and the terms stated by the original data publisher.

Always retain attribution to the Government of India / Ministry of Road Transport & Highways when redistributing the source data.

---

## 18. Author

**Tisha Sutariya**

B.Tech Artificial Intelligence & Data Science

Academic / Portfolio Project
