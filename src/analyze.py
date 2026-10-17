import warnings

import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text

from src.config import RESULTS_DIR, SEED
from src.meta_model import load_meta_dataset

warnings.filterwarnings("ignore")

DISCOVERY_CSV = RESULTS_DIR / "meta_dataset_discovery.csv"
THRESH = 0.01  # "LightGBM loses" = more than 1 point behind the best algorithm

from scipy.stats import spearmanr


def size_check(X, S):
    """Is the 'small data -> LightGBM loses' effect real, or just noisier scores?"""
    best = S.max(axis=1)
    best_algo = S.idxmax(axis=1)
    adv = S["lgbm"] - S.drop(columns="lgbm").mean(axis=1)  # LightGBM minus average of the other 5
    bins = pd.cut(X["n_rows"], [0, 750, 1500, 3000, 4999, 5000])
    g = pd.DataFrame({
        "bin": bins,
        "lgbm_is_best": best_algo == "lgbm",
        "lgbm_regret": best - S["lgbm"],
        "lgbm_vs_others": adv,
        "simple_model_best": best_algo.isin(["logreg", "svm", "knn"]),
    }).groupby("bin", observed=True).agg(
        n=("lgbm_is_best", "size"),
        lgbm_is_best=("lgbm_is_best", "mean"),
        lgbm_regret=("lgbm_regret", "mean"),
        adv_mean=("lgbm_vs_others", "mean"),
        adv_std=("lgbm_vs_others", "std"),
        simple_best=("simple_model_best", "mean"),
    )
    print("\nBy dataset size (last bin = datasets capped at 5000 rows):")
    print(g.round(3).to_string())
    rho, p = spearmanr(X["n_rows"], adv)
    print(f"Spearman(n_rows, lgbm advantage over others) = {rho:.3f}, p = {p:.4f}")

def main():
    df, X, S = load_meta_dataset(DISCOVERY_CSV)
    best = S.max(axis=1)
    loses = ((best - S["lgbm"]) > THRESH).astype(int)
    print(f"{len(df)} datasets. LightGBM loses by >{THRESH} on {loses.sum()} ({loses.mean():.0%}).\n")

    # 1. who beats LightGBM when it loses?
    print("Best algorithm on the datasets where LightGBM loses:")
    print(S[loses == 1].idxmax(axis=1).value_counts().to_string())

    # 2. which meta-features differ between 'loses' and 'fine'?
    rows = []
    for c in X.columns:
        a, b = X.loc[loses == 1, c], X.loc[loses == 0, c]
        if X[c].nunique() < 2:
            continue
        u, p = mannwhitneyu(a, b)
        rows.append({"feature": c, "median_when_loses": a.median(), "median_when_ok": b.median(),
                     "auc": u / (len(a) * len(b)), "p_value": p})
    t = pd.DataFrame(rows)
    t = t.reindex((t.auc - 0.5).abs().sort_values(ascending=False).index)
    print("\nMeta-features vs 'LightGBM loses'  (auc: 0.5 = no difference; >0.5 = higher when LightGBM loses)")
    print("Careful: ~20 features tested, so about 1 will look 'significant' (p<0.05) by luck alone.")
    print(t.round(3).to_string(index=False))
    t.round(4).to_csv(RESULTS_DIR / "lgbm_loss_features.csv", index=False)

    # 3. is 'LightGBM loses' predictable at all from meta-features?
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1)
    auc = cross_val_score(rf, X, loses, cv=cv, scoring="roc_auc")
    print(f"\nCan meta-features predict 'LightGBM loses'?  CV AUC = {auc.mean():.3f} (+/- {auc.std():.3f})   [0.5 = no signal]")
    rf.fit(X, loses)
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("Top features by importance:\n" + imp.head(8).round(3).to_string())

    # 4. a tiny readable rule (descriptive only, fit on the discovery set)
    tree = DecisionTreeClassifier(max_depth=2, min_samples_leaf=8, random_state=SEED).fit(X, loses)
    print("\nSimple rule (class 1 = LightGBM loses):")
    print(export_text(tree, feature_names=list(X.columns)))

    size_check(X, S)
    
if __name__ == "__main__":
    main()