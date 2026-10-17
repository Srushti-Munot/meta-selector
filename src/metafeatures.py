import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from src.benchmark import prepare, SCORES_CSV
from src.config import RESULTS_DIR, SEED
from src.datasets import load_dataset

warnings.filterwarnings("ignore")

META_CSV = RESULTS_DIR / "meta_features.csv"
META_DATASET_CSV = RESULTS_DIR / "meta_dataset.csv"
ALGOS = ["logreg", "knn", "rf", "extratrees", "lgbm", "svm"]


def safe(v):
    return 0.0 if v is None or np.isnan(v) else float(v)


def compute_meta_features(X_raw, y_raw):
    """Describe one dataset with cheap numbers. Same function will be used in the app."""
    X, prep = prepare(X_raw)
    y = LabelEncoder().fit_transform(y_raw)
    n, p = X.shape
    k = len(np.unique(y))

    num = X.select_dtypes(include="number")
    cat = X.drop(columns=num.columns)
    f = {}

    # --- size and shape ---
    f["n_rows"] = n
    f["n_features"] = p
    f["n_classes"] = k
    f["log_rows"] = np.log10(n)
    f["dim_ratio"] = p / n
    f["frac_categorical"] = cat.shape[1] / p
    f["missing_frac"] = float(X.isna().to_numpy().mean())

    # --- class balance ---
    probs = np.bincount(y) / n
    probs = probs[probs > 0]
    f["class_entropy"] = float(-(probs * np.log2(probs)).sum() / np.log2(k))
    f["log_imbalance"] = float(np.log10(probs.max() / probs.min()))

    # --- numeric feature statistics ---
    if num.shape[1] > 0:
        skew = num.skew().abs()
        f["mean_abs_skew"] = safe(np.nanmean(skew))
        f["frac_skewed"] = float((skew > 1).mean())
        f["mean_kurtosis"] = safe(np.nanmean(num.kurt()))
    else:
        f["mean_abs_skew"] = f["frac_skewed"] = f["mean_kurtosis"] = 0.0
    if num.shape[1] > 1:
        c = num.corr().abs().to_numpy()
        f["mean_abs_corr"] = safe(np.nanmean(c[np.triu_indices_from(c, k=1)]))
    else:
        f["mean_abs_corr"] = 0.0
    f["mean_cat_cardinality"] = float(cat.nunique().mean()) if cat.shape[1] else 0.0

    # --- how compressible is the data? (PCA on the encoded features) ---
    Z = prep.fit_transform(X)
    f["n_features_encoded"] = Z.shape[1]
    pca = PCA(n_components=0.95, svd_solver="full").fit(Z[:2000])
    f["pca_95_frac"] = pca.n_components_ / Z.shape[1]

    # --- landmarkers: very cheap models whose scores describe the dataset ---
    n_splits = min(3, int(np.bincount(y).min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    landmarkers = {
        "lm_stump": DecisionTreeClassifier(max_depth=1, random_state=SEED),
        "lm_tree3": DecisionTreeClassifier(max_depth=3, random_state=SEED),
        "lm_1nn": KNeighborsClassifier(n_neighbors=1),
        "lm_nb": GaussianNB(),
    }
    for name, model in landmarkers.items():
        pipe = Pipeline([("prep", prep), ("model", model)])
        f[name] = float(cross_val_score(pipe, X, y, cv=cv, scoring="balanced_accuracy").mean())

    return f


def main():
    ids = pd.read_csv(SCORES_CSV).dataset_id.unique()
    rows = []
    for i, did in enumerate(ids, 1):
        d = load_dataset(int(did))
        if d is None:
            continue
        try:
            f = compute_meta_features(d["X"], d["y"])
        except Exception as e:
            print(f"[{i}/{len(ids)}] {d['name']}: FAILED ({str(e)[:70]})")
            continue
        rows.append({"dataset_id": d["id"], "dataset_name": d["name"], **f})
        print(f"[{i}/{len(ids)}] {d['name']}")

    meta = pd.DataFrame(rows)
    meta.to_csv(META_CSV, index=False)

    # merge with benchmark scores -> one row per dataset: meta-features + 6 scores
    scores = pd.read_csv(SCORES_CSV).pivot_table(index="dataset_id", columns="algorithm", values="score")
    full = meta.merge(scores.reset_index(), on="dataset_id")
    full.to_csv(META_DATASET_CSV, index=False)
    print(f"\nSaved meta_dataset.csv: {full.shape[0]} datasets, {full.shape[1]} columns")



if __name__ == "__main__":
    main()