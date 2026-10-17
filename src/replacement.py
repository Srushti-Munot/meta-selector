import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from src.config import RESULTS_DIR, SEED
from src.evaluate import bootstrap_diff
from src.meta_model import ALGOS, load_meta_dataset

warnings.filterwarnings("ignore")

DISCOVERY_CSV = RESULTS_DIR / "meta_dataset_discovery.csv"
LGBM = ALGOS.index("lgbm")
LOGREG = ALGOS.index("logreg")
ALTS = [i for i in range(len(ALGOS)) if i != LGBM]   # the 5 alternatives
SWITCH_P = 0.5   # fixed in advance
SMALL = 1500


def beat_labels(S):
    """n_datasets x 5: 1 if that alternative scored higher than LightGBM."""
    Sv = S.to_numpy()
    return (Sv[:, ALTS] > Sv[:, [LGBM]]).astype(int)


def fit_predict(Xtr, Ytr, Xte, n_trees=300):
    """For each alternative: P(it beats LightGBM) on the test datasets."""
    P = np.zeros((len(Xte), len(ALTS)))
    for j in range(len(ALTS)):
        y = Ytr[:, j]
        if y.min() == y.max():
            P[:, j] = float(y[0])
        else:
            clf = RandomForestClassifier(n_estimators=n_trees, min_samples_leaf=2, random_state=SEED, n_jobs=-1)
            P[:, j] = clf.fit(Xtr, y).predict_proba(Xte)[:, 1]
    return P


def pick(P, switch_at):
    alt = np.array(ALTS)[P.argmax(axis=1)]
    if switch_at is None:
        return alt
    return np.where(P.max(axis=1) >= switch_at, alt, LGBM)


def report(title, X, S, P):
    Sv, n = S.to_numpy(), len(S)
    best = Sv.max(axis=1)
    idx = np.arange(n)
    policies = {
        "always LightGBM": np.full(n, LGBM),
        "size rule (logreg if <=1500 rows)": np.where(X["n_rows"].to_numpy() <= SMALL, LOGREG, LGBM),
        f"A: best alternative if P>={SWITCH_P}, else LightGBM": pick(P, SWITCH_P),
        "B: always the most likely alternative": pick(P, None),
    }
    base_regret = best - Sv[:, LGBM]
    rows, regrets = [], {}
    for name, picks in policies.items():
        r = best - Sv[idx, picks]
        sw = picks != LGBM
        rows.append({
            "policy": name, "mean_regret": r.mean(), "within_1pct": (r <= 0.01).mean(),
            "switched": sw.mean(),
            "switch_beat_lgbm": (Sv[idx, picks] > Sv[:, LGBM])[sw].mean() if sw.any() else np.nan,
            "better_than_lgbm": int((r < base_regret).sum()), "worse_than_lgbm": int((r > base_regret).sum()),
        })
        regrets[name] = r
    print(f"\n=== {title}  ({n} datasets) ===")
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    print("\nPolicy minus always-LightGBM regret (negative = better), 95% bootstrap CI:")
    for name in list(policies)[1:]:
        d, lo, hi = bootstrap_diff(regrets[name], regrets["always LightGBM"])
        verdict = "SUPPORTED" if hi < 0 else "not supported"
        print(f"  {name:<52} {d:+.4f}  [{lo:+.4f}, {hi:+.4f}]  -> {verdict}")

    Y = beat_labels(S)
    print("\nCan we predict whether each alternative beats LightGBM? (AUC; 0.5 = chance)")
    for j, a in enumerate(ALTS):
        pos = int(Y[:, j].sum())
        auc = roc_auc_score(Y[:, j], P[:, j]) if 0 < pos < n else np.nan
        print(f"  {ALGOS[a]:<11} beats LightGBM on {pos}/{n} datasets ({pos/n:.0%}) | AUC = {auc:.3f}")


def lodo(X, S, n_trees=150):
    Y, n = beat_labels(S), len(S)
    P = np.zeros((n, len(ALTS)))
    for i in range(n):
        tr = np.flatnonzero(np.arange(n) != i)
        P[i] = fit_predict(X.iloc[tr], Y[tr], X.iloc[[i]], n_trees)[0]
        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{n}", flush=True)
    return P


def main():
    disc_df, Xd, Sd = load_meta_dataset(DISCOVERY_CSV)
    full_df, Xf, Sf = load_meta_dataset()
    complete = Sf.notna().all(axis=1)
    is_conf = ~full_df.dataset_id.isin(disc_df.dataset_id) & complete
    Xc = Xf[is_conf][Xd.columns].reset_index(drop=True)
    Sc = Sf[is_conf].reset_index(drop=True)

    print("EXPLORATORY (the confirmation set was already used for H1-H4).")
    P = fit_predict(Xd, beat_labels(Sd), Xc)
    report("PRIMARY: trained on discovery, tested on unseen confirmation datasets", Xc, Sc, P)

    if len(sys.argv) > 1 and sys.argv[1] == "lodo":
        Xa, Sa = Xf[complete][Xd.columns].reset_index(drop=True), Sf[complete].reset_index(drop=True)
        print("\nPooled leave-one-dataset-out over all datasets (a few minutes)...")
        report("SECONDARY: pooled leave-one-dataset-out", Xa, Sa, lodo(Xa, Sa))


if __name__ == "__main__":
    main()