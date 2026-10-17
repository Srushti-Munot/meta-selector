import warnings

import numpy as np
import pandas as pd
from scipy.stats import binomtest, fisher_exact, mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from src.config import RESULTS_DIR, SEED
from src.meta_model import load_meta_dataset

warnings.filterwarnings("ignore")

DISCOVERY_CSV = RESULTS_DIR / "meta_dataset_discovery.csv"
THRESHOLDS = [0.005, 0.01, 0.02]
SIMPLE = ["logreg", "svm", "knn"]
SMALL = 1500


def loses(S, t):
    """True where LightGBM is more than t behind the best of the 6 algorithms."""
    return ((S.max(axis=1) - S["lgbm"]) > t).to_numpy()


def pattern_row(X, S, t, label):
    y = loses(S, t)
    n = len(y)
    n_rows = X["n_rows"].to_numpy()
    small = n_rows <= SMALL
    row = {"threshold": t, "set": label, "n": n, "lgbm_loses": int(y.sum()), "loss_rate": y.mean()}

    # (a) are the datasets where LightGBM loses smaller?  (AUC < 0.5 = yes)
    if 0 < y.sum() < n:
        u, p = mannwhitneyu(n_rows[y], n_rows[~y], alternative="less")
        row["n_rows_AUC"], row["AUC_p"] = u / (y.sum() * (~y).sum()), p
    else:
        row["n_rows_AUC"], row["AUC_p"] = np.nan, np.nan

    # (b) LightGBM-loss rate on small vs large datasets
    ks, ns, kl, nl = int(y[small].sum()), int(small.sum()), int(y[~small].sum()), int((~small).sum())
    row["loss_rate_<=1500"] = ks / ns if ns else np.nan
    row["loss_rate_>1500"] = kl / nl if nl else np.nan
    row["fisher_p"] = fisher_exact([[ks, ns - ks], [kl, nl - kl]], alternative="greater")[1] if ns and nl else np.nan

    # (c) when LightGBM loses, is a simple model the best?
    simple_best = S.idxmax(axis=1).isin(SIMPLE).to_numpy()
    row["simple_best_in_losses"] = simple_best[y].mean() if y.sum() else np.nan
    return row


def h4_row(Xd, Sd, Xc, Sc, t):
    yd, yc = loses(Sd, t).astype(int), loses(Sc, t).astype(int)
    out = {"threshold": t, "train_losses": int(yd.sum()), "test_losses": int(yc.sum())}
    if yd.sum() < 2 or yc.sum() == 0 or yc.sum() == len(yc):
        return {**out, "AUC": np.nan, "CI_low": np.nan, "CI_high": np.nan}
    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1).fit(Xd, yd)
    proba = clf.predict_proba(Xc)[:, 1]
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, len(yc), len(yc))
        if 0 < yc[idx].sum() < len(idx):
            boots.append(roc_auc_score(yc[idx], proba[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {**out, "AUC": roc_auc_score(yc, proba), "CI_low": lo, "CI_high": hi}


def main():
    disc_df, Xd, Sd = load_meta_dataset(DISCOVERY_CSV)
    full_df, Xf, Sf = load_meta_dataset()
    is_conf = ~full_df.dataset_id.isin(disc_df.dataset_id) & Sf.notna().all(axis=1)
    Xc = Xf[is_conf][Xd.columns].reset_index(drop=True)
    Sc = Sf[is_conf].reset_index(drop=True)

    # H1 does not depend on the threshold
    print("H1 (threshold-free): share of datasets where a simple model (logreg/svm/knn) is best")
    for label, X, S in [("discovery", Xd, Sd), ("confirmation", Xc, Sc)]:
        sb = S.idxmax(axis=1).isin(SIMPLE).to_numpy()
        sm = (X["n_rows"] <= SMALL).to_numpy()
        p = fisher_exact([[int(sb[sm].sum()), int((~sb[sm]).sum())],
                          [int(sb[~sm].sum()), int((~sb[~sm]).sum())]], alternative="greater")[1]
        print(f"   {label:<13} <=1500 rows: {sb[sm].mean():.0%} (n={sm.sum()})  >1500 rows: {sb[~sm].mean():.0%} (n={(~sm).sum()})  Fisher p={p:.4f}")

    rows = []
    for t in THRESHOLDS:
        rows.append(pattern_row(Xd, Sd, t, "discovery"))
        rows.append(pattern_row(Xc, Sc, t, "confirmation"))
    pat = pd.DataFrame(rows)
    print("\nSmall-data pattern at each threshold (n_rows_AUC < 0.5 and loss_rate_<=1500 > loss_rate_>1500 = pattern holds)")
    print(pat.round(4).to_string(index=False))

    h4 = pd.DataFrame([h4_row(Xd, Sd, Xc, Sc, t) for t in THRESHOLDS])
    print("\nH4 at each threshold: classifier trained on discovery, AUC on the unseen confirmation datasets")
    print(h4.round(3).to_string(index=False))

    pat.to_csv(RESULTS_DIR / "sensitivity_pattern.csv", index=False)
    h4.to_csv(RESULTS_DIR / "sensitivity_h4.csv", index=False)

    # pre-set consistency criteria (defined before running)
    c1 = bool((pat["n_rows_AUC"] < 0.5).all())
    c2 = bool((pat["loss_rate_<=1500"] > pat["loss_rate_>1500"]).all())
    c3 = bool((h4["CI_low"] > 0.5).all())
    print("\nPre-set criteria:")
    print(f"  C1 n_rows AUC < 0.5 at every threshold, in both sets:          {'YES' if c1 else 'NO'}")
    print(f"  C2 loss rate higher on <=1500 rows at every threshold, both sets: {'YES' if c2 else 'NO'}")
    print(f"  C3 H4 AUC interval entirely above 0.5 at every threshold:      {'YES' if c3 else 'NO'}")
    print("  => " + ("Findings are robust to the threshold." if (c1 and c2 and c3)
                     else "NOT all criteria hold; report which fail (see tables)."))
    few = pat[pat.lgbm_loses < 20]
    if len(few):
        print("  Note: fewer than 20 'LightGBM loses' datasets in: "
              + ", ".join(f"T={r.threshold} ({r.set})" for r in few.itertuples()))


if __name__ == "__main__":
    main()