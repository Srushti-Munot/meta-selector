import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RESULTS_DIR, SEED

ALGOS = ["logreg", "knn", "rf", "extratrees", "lgbm", "svm"]
META_DATASET_CSV = RESULTS_DIR / "meta_dataset.csv"


def load_meta_dataset(path=META_DATASET_CSV):
    """Returns: full table, meta-feature matrix X, score matrix S (one row per dataset)."""
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in ["dataset_id", "dataset_name"] + ALGOS]
    X = df[feature_cols].fillna(0)
    S = df[ALGOS]
    return df, X, S


def make_targets(S, target):
    if target == "absolute":
        return S
    if target == "relative":  # score minus the dataset's own average score
        return S.sub(S.mean(axis=1), axis=0)
    raise ValueError(target)


def make_meta_model(kind):
    if kind == "rf":
        return RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=SEED, n_jobs=-1)
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    raise ValueError(kind)