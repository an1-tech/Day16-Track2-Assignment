import json
from time import perf_counter

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score,
)

start = perf_counter()
df = pd.read_csv("creditcard.csv")
load_seconds = perf_counter() - start

X = df.drop(columns="Class")
y = df["Class"]

# Split into 60% training, 20% validation, 20% test.
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev, test_size=0.25,
    stratify=y_dev, random_state=42,
)

model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    random_state=42,
    n_jobs=2,
    verbosity=-1,
)

start = perf_counter()
model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(50, verbose=False)],
)
training_seconds = perf_counter() - start

probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= 0.5).astype(int)

one_row = X_test.iloc[:1]
batch = X_test.iloc[:1000]

# Warm up inference before timing.
for _ in range(10):
    model.predict_proba(one_row)
model.predict_proba(batch)

latencies = []
for _ in range(100):
    start = perf_counter()
    model.predict_proba(one_row)
    latencies.append(perf_counter() - start)

start = perf_counter()
for _ in range(20):
    model.predict_proba(batch)
batch_seconds = (perf_counter() - start) / 20

result = {
    "dataset_rows": int(len(df)),
    "train_rows": int(len(X_train)),
    "validation_rows": int(len(X_val)),
    "test_rows": int(len(X_test)),
    "load_data_seconds": load_seconds,
    "training_seconds": training_seconds,
    "best_iteration": int(model.best_iteration_),
    "auc_roc": float(roc_auc_score(y_test, probabilities)),
    "accuracy": float(accuracy_score(y_test, predictions)),
    "f1_score": float(f1_score(
        y_test, predictions, zero_division=0
    )),
    "precision": float(precision_score(
        y_test, predictions, zero_division=0
    )),
    "recall": float(recall_score(
        y_test, predictions, zero_division=0
    )),
    "inference_latency_1_row_ms": float(
        np.mean(latencies) * 1000
    ),
    "inference_batch_rows": int(len(batch)),
    "inference_batch_seconds": batch_seconds,
    "inference_throughput_rows_per_second": (
        len(batch) / batch_seconds
    ),
    "lightgbm_version": lgb.__version__,
}

with open("benchmark_result.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))

