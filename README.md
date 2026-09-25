# Lab 2: Modelling & Model Lifecycle — Predicting Plant Production (GIST Steel Dataset)
  
**Group Members:**
1. Saty Viard (Student ID: B00822312)
2. Pollux Gronier (Student ID: B00822392)
3. Eliott Beghin (Student ID: B00824999)

---

## 🎯 Overview

This project implements the complete machine learning lifecycle for predicting plant-level annual crude steel production (`production` in ttpa) using the Global Energy Monitor (GEM) Global Iron and Steel Tracker (GIST) June 2026 dataset.

### Pipeline Stages Implemented:
1. **Data Setup & Schema Validation (Task 1.1–1.2):** 
   - Loaded primary plant and capacity sheets via Pandas and demonstrated Polars loading (`pl.read_excel`).
   - Aggregated multi-furnace configurations to 1 row per plant (`GEM plant ID`).
   - Handled string pollution in numerical columns: `'>0'` indicates unconfirmed positive capacity and is explicitly mapped to `NaN` for median imputation (avoiding false zeros).
   - Validated raw tables against strict Pandera schemas before cleaning to detect string flags (`'>0'`, `'unknown'`) and unlabelled targets.
2. **Cleaning & Feature Engineering (Task 1.3–1.5):**
   - Cleaned corrupted string flags, filtered unlabelled target rows, and verified the cleaned data against Pandera.
   - Engineered domain features: `capacity_per_worker` (labor productivity), `bof_share` (oxygen vs electric route), `eaf_share`, and integrated macroeconomic national steel production metadata.
   - Evaluated linear relationships via pairwise correlation matrices ($r \approx 0.87$ between crude capacity and production).
3. **Baselines & Linear Models (Task 2.1–2.2):**
   - Benchmarked against dummy baselines (mean and median predictors).
   - Built a full Scikit-Learn `Pipeline` with `ColumnTransformer` (median imputation, one-hot encoding, standard scaling) and Ridge/LinearRegression.
   - Extracted and interpreted regression coefficients (positive: crude steel capacity; negative: plant age and idled statuses).
4. **Model Comparison & Tuning (Task 3.1–3.3):**
   - Evaluated 5-fold cross-validation stability, diagnosing how high-leverage mega-mills in training folds create negative $R^2$ swings in linear validation folds.
   - Compared Linear Regression, Ridge, and Random Forest.
   - Assessed Google Research TabFM v1.0 foundation model requirements (6.59 GB safetensors checkpoint); noted hardware memory constraints without reporting fabricated numbers.
   - Tuned Random Forest hyperparameters via `GridSearchCV` (`max_depth`, `min_samples_split`, `max_features`), lowering test RMSE to **1,492.9 ttpa** and achieving **$R^2 = 0.75$**.
5. **Lifecycle Tracking & Storage (Task 4.1–4.3):**
   - Logged runs, hyperparameters, metrics, and residual plot artifacts to MLflow under experiment `gist_steel_production_prediction`.
   - Executed a 30-trial Optuna Bayesian optimization study (TPE) integrated via `optuna.integration.MLflowCallback`.
   - Serialized the complete preprocessing + model pipeline to `best_steel_production_pipeline.joblib`. Reloaded from disk and verified exact bit-for-bit prediction parity on test data (`np.testing.assert_allclose`).
6. **Deployment & Monitoring (Task 5.1–5.2):**
   - Implemented an inference scoring function with Pandera pre-validation gating.
   - Implemented statistical data drift detection using Kolmogorov-Smirnov (KS) tests and the Population Stability Index (PSI).

---

## 📊 Benchmark Results

| Model | 5-Fold CV RMSE (ttpa) | 5-Fold CV MAE (ttpa) | 5-Fold CV $R^2$ | Test RMSE (ttpa) | Test MAE (ttpa) | Test $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy (Mean)** | — | — | — | 3,022.3 | 2,026.9 | -0.01 |
| **Dummy (Median)** | — | — | — | 3,111.4 | 1,939.8 | -0.07 |
| **Linear Regression** | 2,492.9 | 1,192.9 | -1.51 | 1,648.8 | 1,021.4 | 0.69 |
| **Ridge Regression ($\alpha=10$)** | 2,156.4 | 1,012.3 | -0.62 | 1,650.2 | 1,018.9 | 0.69 |
| **Random Forest (Default)** | 1,282.6 | 703.5 | 0.71 | 1,520.8 | 845.5 | 0.74 |
| **Random Forest (Tuned)** | **1,249.7** | **689.4** | **0.72** | **1,492.9** | **832.1** | **0.75** |
| **TabFM (Foundation Model)** | *N/A* | *N/A* | *N/A* | *N/A (OOM)* | *N/A* | *N/A* |

*Note on TabFM*: The official PyTorch weights (`google/tabfm-1.0.0-pytorch`) require loading a 6.59 GB model checkpoint into RAM, which exceeds available physical memory on this 8 GB CPU-only test host. TabFM was omitted from the empirical table rather than reporting unmeasured placeholder numbers.

---

##  Repository table of Contents

```
.
├── lab_2.ipynb                                                              # Fully executed notebook with all outputs visible
├── README.md                                                                # Setup, methodology, and reproduction guide
├── best_steel_production_pipeline.joblib                                    # Serialized best model pipeline (preprocessor + RF)
├── residuals_tuned_rf.png                                                   # Diagnostic residual scatter plot (logged in MLflow)
├── mlruns/                                                                  # Local MLflow tracking store
├── Plant-level_data_Global_Iron_and_Steel_Tracker_June_2026_V1.xlsx         # GEM primary tracker
├── Steel_unit_data_Global_Iron_and_Steel_Tracker_June_2026_V1.xlsx         # Unit-level steel furnaces
├── Iron_unit_data_Global_Iron_and_Steel_Tracker_June_2026_V1.xlsx          # Unit-level iron furnaces
└── Production-Consumption-of-Met-Coal-Iron-Ore-by-Steel-Industry...xlsx   # Country-level macroeconomic steel statistics
```
