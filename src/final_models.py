import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier

from src.config import ROOT, RESULTS_DIR, SEED
from src.meta_model import ALGOS, load_meta_dataset, make_meta_model, make_targets

warnings.filterwarnings("ignore")

DISCOVERY_CSV = RESULTS_DIR / "meta_dataset_discovery.csv"
OUT_PATH = ROOT / "app" / "final_models.joblib"
THRESH = 0.01
SIMPLE = ["logreg", "svm", "knn"]
BAND_EDGES = [0.25, 0.50]  # fixed in advance (not tuned): Low < 0.25 <= Medium < 0.50 <= High
BAND_NAMES = ["Low", "Medium", "High"]


def new_clf():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1)


def loses_label(S):
    """1 if LightGBM is more than 0.01 behind the best of the 6 algorithms."""
    return ((S.max(axis=1) - S["lgbm"]) > THRESH).astype(int).to_numpy()


def wilson(k, n, z=1.96):
    """95% interval for a proportion k/n (works for small n)."""
    if n == 0:
        return np.nan, np.nan
    ph = k / n
    denom = 1 + z * z / n
    centre = (ph + z * z / (2 * n)) / denom
    half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def main():
    disc_df, Xd, Sd = load_meta_dataset(DISCOVERY_CSV)
    full_df, Xf, Sf = load_meta_dataset()
    complete = Sf.notna().all(axis=1)
    is_conf = ~full_df.dataset_id.isin(disc_df.dataset_id) & complete
    Xc = Xf[is_conf][Xd.columns].reset_index(drop=True)
    Sc = Sf[is_conf].reset_index(drop=True)

    # ---- 1. risk bands: classifier trained on DISCOVERY only, rates observed on UNSEEN datasets ----
    clf = new_clf().fit(Xd, loses_label(Sd))
    p = clf.predict_proba(Xc)[:, 1]
    y = loses_label(Sc)
    bands = np.digitize(p, BAND_EDGES)  # 0 = Low, 1 = Medium, 2 = High
    rows = []
    for b, name in enumerate(BAND_NAMES):
        m = bands == b
        n_b, k = int(m.sum()), int(y[m].sum())
        lo, hi = wilson(k, n_b)
        rows.append({"band": name, "n_datasets": n_b, "lgbm_lost": k,
                     "observed_rate": k / n_b if n_b else np.nan, "ci_low": lo, "ci_high": hi})
    band_table = pd.DataFrame(rows)
    print(f"LightGBM risk bands, checked on {len(y)} unseen datasets (overall LightGBM-loses rate: {y.mean():.0%})")
    print(band_table.round(3).to_string(index=False))
    print("(Bands use cutoffs 0.25 and 0.50, fixed in advance. Small n_datasets = wide interval = treat with care.)\n")

    # ---- 2. the size pattern, for the app's explanation text ----
    small = (Xc["n_rows"] <= 1500).to_numpy()
    simple_best = Sc.idxmax(axis=1).isin(SIMPLE).to_numpy()
    size_stats = {
        "small_n": int(small.sum()), "small_share": float(simple_best[small].mean()) if small.any() else np.nan,
        "large_n": int((~small).sum()), "large_share": float(simple_best[~small].mean()) if (~small).any() else np.nan,
    }
    print(f"Simple model best: {size_stats['small_share']:.0%} of datasets <=1500 rows (n={size_stats['small_n']}) "
          f"vs {size_stats['large_share']:.0%} above (n={size_stats['large_n']})\n")

    # ---- 3. final models: trained on ALL complete datasets (discovery + confirmation) ----
    Xall, Sall = Xf[complete][Xd.columns], Sf[complete]
    ranker = make_meta_model("rf").fit(Xall, make_targets(Sall, "relative"))
    risk_clf = new_clf().fit(Xall, loses_label(Sall))

    bundle = {
        "ranker": ranker, "risk_clf": risk_clf, "feature_cols": list(Xd.columns), "algos": ALGOS,
        "band_edges": BAND_EDGES, "band_names": BAND_NAMES, "band_table": band_table.to_dict("records"),
        "base_rate": float(y.mean()), "size_stats": size_stats, "n_train": int(complete.sum()),
        "sklearn_version": sklearn.__version__,
    }
    OUT_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, OUT_PATH, compress=3)
    print(f"Final models trained on {int(complete.sum())} datasets -> {OUT_PATH} ({OUT_PATH.stat().st_size/1e6:.1f} MB)")
    print(f"scikit-learn version used: {sklearn.__version__} (the deployed app must use the same one)")


if __name__ == "__main__":
    main()