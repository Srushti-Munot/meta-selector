import sys
import time
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from src.config import RESULTS_DIR, SEED, CV_FOLDS
from src.datasets import get_cc18_ids, get_extra_ids, load_dataset

warnings.filterwarnings("ignore")

SCORES_CSV = RESULTS_DIR / "benchmark_scores.csv"


def make_models():
    """The 6 candidate algorithms, all with default-ish settings."""
    return {
        "logreg": LogisticRegression(max_iter=1000),
        "knn": KNeighborsClassifier(),
        "rf": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED),
        "extratrees": ExtraTreesClassifier(n_estimators=100, n_jobs=-1, random_state=SEED),
        "lgbm": LGBMClassifier(n_estimators=100, random_state=SEED, verbose=-1),
        "svm": SVC(),
    }


def prepare(X):
    """Split columns into numeric / categorical and build the preprocessor.
    The preprocessor is fitted INSIDE each CV fold (via Pipeline), so no leakage."""
    X = X.copy()
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    if cat_cols:
        X[cat_cols] = X[cat_cols].astype(object)

    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=20)),
    ])
    prep = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])
    return X, prep


def append_row(row):
    pd.DataFrame([row]).to_csv(SCORES_CSV, mode="a", header=not SCORES_CSV.exists(), index=False)


def benchmark_dataset(d, done):
    X, prep = prepare(d["X"])
    y = LabelEncoder().fit_transform(d["y"])

    n_splits = min(CV_FOLDS, int(np.bincount(y).min()))
    if n_splits < 2:
        print(f"   skipped (a class has <2 samples)")
        return
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    for name, model in make_models().items():
        if (d["id"], name) in done:
            continue  # already benchmarked -> resumable
        pipe = Pipeline([("prep", prep), ("model", model)])
        t0 = time.time()
        row = {"dataset_id": d["id"], "dataset_name": d["name"], "algorithm": name}
        try:
            scores = cross_val_score(pipe, X, y, cv=cv, scoring="balanced_accuracy", error_score="raise")
            row.update(score=scores.mean(), score_std=scores.std(), status="ok")
        except Exception as e:
            row.update(score=np.nan, score_std=np.nan, status=f"error: {str(e)[:80]}")
        row["time_sec"] = round(time.time() - t0, 2)
        append_row(row)
        print(f"   {name:<11} {row['score']:.4f}  ({row['time_sec']}s)  {row['status']}")


def main(ids):
    done = set()
    if SCORES_CSV.exists():
        old = pd.read_csv(SCORES_CSV)
        done = set(zip(old.dataset_id, old.algorithm))

    count = 0
    for did in ids:
        try:
            d = load_dataset(did)
        except Exception as e:
            print(f"skip {did}: load failed ({str(e)[:60]})")
            continue
        if d is None:
            continue
        count += 1
        print(f"[{count}] {d['id']} {d['name']}  X={d['X'].shape}")
        try:
            benchmark_dataset(d, done)
        except Exception as e:
            print(f"   dataset failed: {str(e)[:80]}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "cc18"
    if which == "cc18":
        main(get_cc18_ids())
    elif which == "extra":
        main(get_extra_ids())
    else:
        print("usage: python -m src.benchmark [cc18|extra]")