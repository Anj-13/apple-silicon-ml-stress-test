"""
Workload #1: Train XGBoost on 500k rows with GridSearch
This uses 16GB RAM + all CPU cores
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score
import time
import psutil
import os

print(f"RAM available: {psutil.virtual_memory().available / 1e9:.2f} GB")

print("Generating 500k samples...")
X, y = make_classification(
    n_samples=500000,
    n_features=50,
    n_informative=30,
    n_redundant=10,
    random_state=42,
)

# Convert to DataFrame to use more RAM
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
df['target'] = y

print(f"DataFrame memory: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
print(f"RAM used: {psutil.virtual_memory().used / 1e9:.2f} GB")

X_train, X_test, y_train, y_test = train_test_split(
    df.drop('target', axis=1),
    df['target'],
    test_size=0.2,
    random_state=42,
)

# XGBoost with GridSearch (uses 16GB)
print("\n🤔Training XGBoost with GridSearch...")
start = time.time()

param_grid = {
    'max_depth': [3, 6, 9],
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
}

xgb_model = xgb.XGBClassifier(
    n_jobs=-1, # use all core
    tree_method='hist',
)

grid = GridSearchCV(
    xgb_model, param_grid, cv = 3,
    scoring='accuracy', n_jobs=1,
    verbose=1
)

import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

grid.fit(X_train, y_train)

profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumtime')
stats.print_stats()

print(f"Training completed in {time.time() - start:.2f} seconds")
print(f"Best params: {grid.best_params_}")
print(f"Best score: {grid.best_score_:.4f}")
print(f"Test accuracy: {accuracy_score(y_test, grid.predict(X_test)):.4f}")

# Save model
import joblib
joblib.dump(grid.best_estimator_, 'xgboost_model.pkl')
print("Model saved to xgboosst_model.pkl 🙈")