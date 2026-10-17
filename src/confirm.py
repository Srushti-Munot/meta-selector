import warnings

import numpy as np
import pandas as pd
from scipy.stats import binomtest, fisher_exact, mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from src.config import RESULTS_DIR, SEED
from src.evaluate import bootstrap_diff, evaluate
from src.meta_model import ALGOS, load_meta_dataset, make_meta_model, make_targets

warnings.filterwarnings("ignore")

DISCOVERY_CSV = RESULTS_DIR / "meta_dataset_discovery.csv"
THRESH = 0.01
SIMPLE = ["logreg", "svm", "knn"]


def verdict(ok):
    return "SUPPORTED" if ok else "NOT SUPPORTED"


def main():
    disc_df, Xd, Sd = load_meta_dataset(DISCOVERY_CSV)
    full_df, Xf, Sf = load_meta_dataset()

    is_conf = ~full_df.dataset_id.isin(disc_df.dataset_id) & Sf.notna().all(axis=1)
    conf_df = full_df[is_conf].reset_index(drop=True)
    Xc = Xf[is_conf][Xd.columns].reset_index(drop=True)
    Sc = Sf[is_conf].reset_index(drop=True)
    Sv = Sc.to_numpy()
    n = len(Sc)
    print(f"Discovery: {len(disc_df)} datasets | Confirmation: {n} datasets (never used to build anything)\n")

    best = Sc.max(axis=1)
    best_algo = Sc.idxmax(axis=1)
    simple_best = best_algo.isin(SIMPLE)
    loses = (best - Sc["lgbm"]) > THRESH
    print(f"LightGBM loses by >{THRESH} on {loses.sum()} of {n} ({loses.mean():.0%}); discovery was 29%.\n")

    # ---------- H1: simple models are best more often on small datasets ----------
    small = Xc["n_rows"] <= 1500
    table = [[int(simple_best[small].sum()), int((~simple_best[small]).sum())],
             [int(simple_best[~small].sum()), int((~simple_best[~small]).sum())]]
    _, p1 = fisher_exact(table, alternative="greater")
    print("H1: share of datasets where a simple model (logreg/svm/knn) is best")
    print(f"    <=1500 rows: {simple_best[small].mean():.0%} (n={small.sum()}) | >1500 rows: {simple_best[~small].mean():.0%} (n={(~small).sum()})  [discovery: 49% vs 25%]")
    print(f"    one-sided Fisher p = {p1:.4f}  -> {verdict(p1 < 0.05)}\n")

    # ---------- H1b: datasets where LightGBM loses are smaller ----------
    if loses.sum() > 0 and (~loses).sum() > 0:
        a, b = Xc.loc[loses, "n_rows"], Xc.loc[~loses, "n_rows"]
        u, p1b = mannwhitneyu(a, b, alternative="less")
        auc = u / (len(a) * len(b))
        print(f"H1b: n_rows AUC (loses vs ok) = {auc:.3f}  [discovery 0.305; <0.5 = losers are smaller]")
        print(f"    one-sided Mann-Whitney p = {p1b:.4f}  -> {verdict(p1b < 0.05)}\n")

    # ---------- H2: when LightGBM loses, a simple model is usually best ----------
    k, m = int(simple_best[loses].sum()), int(loses.sum())
    if m > 0:
        p2 = binomtest(k, m, 0.5, alternative="greater").pvalue
        print(f"H2: simple model best in {k}/{m} = {k/m:.0%} of LightGBM's losses  [discovery 71%]")
        print(f"    criterion (>50%) -> {verdict(k / m > 0.5)}  (binomial p = {p2:.3f}, for context)\n")

    # ---------- H3: the chooser, trained on discovery only ----------
    rows, regrets = [], {}

    def add(name, P):
        row, reg = evaluate(name, P, Sv)
        rows.append(row)
        regrets[name] = reg

    for a_name in ["lgbm", "rf"]:
        P = np.zeros((n, len(ALGOS)))
        P[:, ALGOS.index(a_name)] = 1
        add(f"always {a_name}", P)
    add("best-average (from discovery)", np.tile(Sd.mean().to_numpy(), (n, 1)))

    R = best.to_numpy()[:, None] - Sv
    rows.append({"method": "random pick", "mean_regret": R.mean(), "median_regret": np.median(R.mean(axis=1)),
                 "within_1pct": (R <= 0.01).mean(), "top1_hit": 1 / 6, "top2_hit": 2 / 6, "spearman": np.nan})

    for target, label in [("relative", "CHOOSER rf/relative (primary)"), ("absolute", "chooser rf/absolute (exploratory)")]:
        model = make_meta_model("rf")
        model.fit(Xd, make_targets(Sd, target))
        add(label, model.predict(Xc))

    res = pd.DataFrame(rows).round(4)
    print("H3: results on the confirmation datasets")
    print(res.to_string(index=False))
    res.to_csv(RESULTS_DIR / "confirmation_results.csv", index=False)

    d, lo, hi = bootstrap_diff(regrets["CHOOSER rf/relative (primary)"], regrets["always lgbm"])
    better = int((regrets["CHOOSER rf/relative (primary)"] < regrets["always lgbm"]).sum())
    worse = int((regrets["CHOOSER rf/relative (primary)"] > regrets["always lgbm"]).sum())
    print(f"\n    chooser minus always-lgbm: {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  (better on {better}, worse on {worse})")
    print(f"    criterion (CI excludes 0 on the good side) -> {verdict(hi < 0)}\n")

    # ---------- H4: can we predict 'LightGBM loses' on unseen datasets? ----------
    loses_d = ((Sd.max(axis=1) - Sd["lgbm"]) > THRESH).astype(int)
    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1).fit(Xd, loses_d)
    proba = clf.predict_proba(Xc)[:, 1]
    y = loses.to_numpy().astype(int)
    if 0 < y.sum() < len(y):
        auc4 = roc_auc_score(y, proba)
        rng = np.random.default_rng(SEED)
        boots = []
        for _ in range(2000):
            idx = rng.integers(0, len(y), len(y))
            if 0 < y[idx].sum() < len(idx):
                boots.append(roc_auc_score(y[idx], proba[idx]))
        lo4, hi4 = np.percentile(boots, [2.5, 97.5])
        print(f"H4: AUC for predicting 'LightGBM loses' = {auc4:.3f}  95% CI [{lo4:.3f}, {hi4:.3f}]  [discovery CV: 0.645]")
        print(f"    criterion (AUC>0.5) -> {verdict(auc4 > 0.5)}  (CI excludes 0.5: {lo4 > 0.5})")


if __name__ == "__main__":
    main()