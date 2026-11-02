# XGBoost Training Results

Results from the XGBoost experiments.

## System Info

- **OS:** macOS (Apple Silicon)
- **Python:** venv, Python 3.11.7
- **RAM:** 16 GB total (approx. 5.56 GB available at the start of the 500k-row run)

## Runs

Each run of `train_large_model.py` is documented below in chronological order.

### 1. XGBoost `libomp.dylib` import error

- **Result:** `import xgboost` failed because the OpenMP runtime library (`libomp.dylib`) was not installed. XGBoost requires OpenMP to run.
- **Status:** Fixed by installing the runtime so XGBoost could load.

### 2. f-string format error

- **Result:** A syntax error — a stray `)` in the format specifier broke the `memory_usage` f-string print.
- **Status:** Fixed.

### 3. First run — `n_samples=500` (no profiling)

- **Result:** Completed successfully. Saved model to `xgboost_model.pkl`.

### 4. Second run — `n_samples=500` (no profiling)

- **Result:** Completed successfully.

### 5. Third run — `n_samples=500` (with cProfile/pstats)

- **Result:** Completed successfully, with cProfile/pstats profiling enabled around `grid.fit(...)`.
- **Duration:** 9.92 s

### 6. Final run — `n_samples=500000`

- **Result:** Completed successfully.
- **Duration:** 105.66 s
- **Best parameters:** `{'learning_rate': 0.1, 'max_depth': 9, 'n_estimators': 200}`
- **Best score:** 0.9879 (3-fold CV accuracy on synthetic data)
- **Model saved to:** `xgboost_model.pkl`
