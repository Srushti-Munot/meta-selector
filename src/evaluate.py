import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import LeaveOneOut

from src.config import RESULTS_DIR, SEED
from src.meta_model import ALGOS, load_meta_dataset, make_meta_model, make_targets

warnings.filterwarnings("ignore")


def lodo_predictions(X, T, kind):
    """For each dataset: train the chooser on all OTHER datasets, then predict this one."""
    preds = np.zeros(T.shape)
    for train_idx, test_idx in LeaveOneOut().split(X):
        model = make_meta_model(kind)
        model.fit(X.iloc[train_idx], T.iloc[train_idx])
        preds[test_idx] = model.predict(X.iloc[test_idx])
    return preds


def evaluate(name, P, Sv):
    """P = how much the method likes each algorithm (n_datasets x 6). Highest wins."""
    best = Sv.max(axis=1)
    picks = P.argmax(axis=1)
    regret = best - Sv[np.arange(len(Sv)), picks]
    top2 = np.argsort(-P, axis=1)[:, :2]
    true_best = Sv.argmax(axis=1)
    rho = [spearmanr(P[i], Sv[i]).correlation for i in range(len(Sv))]
    row = {
        "method": name,
        "mean_regret": regret.mean(),
        "median_regret": np.median(regret),
        "within_1pct": (regret <= 0.01).mean(),
        "top1_hit": (picks == true_best).mean(),
        "top2_hit": np.mean([true_best[i] in top2[i] for i in range(len(Sv))]),
        "spearman": np.nanmean(rho),
    }
    return row, regret


def bootstrap_diff(regret_a, regret_b, n_boot=2000):
    """Mean regret of A minus B, with 95% CI (negative = A is better)."""
    rng = np.random.default_rng(SEED)
    d = regret_a - regret_b
    boots = [rng.choice(d, size=len(d), replace=True).mean() for _ in range(n_boot)]
    return d.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def main():
    df, X, S = load_meta_dataset()
    Sv = S.to_numpy()
    n = len(Sv)
    best = Sv.max(axis=1)
    rows, regrets = [], {}

    def add(name, P):
        row, reg = evaluate(name, P, Sv)
        rows.append(row)
        regrets[name] = reg

    # --- baselines ---
    for a in ["lgbm", "rf"]:
        P = np.zeros((n, len(ALGOS)))
        P[:, ALGOS.index(a)] = 1
        add(f"always {a}", P)
    add("best-average (fair, LODO)", (Sv.sum(axis=0) - Sv) / (n - 1))

    R = best[:, None] - Sv  # expected result of picking at random
    rows.append({"method": "random pick", "mean_regret": R.mean(),
                 "median_regret": np.median(R.mean(axis=1)), "within_1pct": (R <= 0.01).mean(),
                 "top1_hit": 1 / 6, "top2_hit": 2 / 6, "spearman": np.nan})

    # --- learned choosers ---
    for kind in ["rf", "ridge"]:
        for target in ["absolute", "relative"]:
            T = make_targets(S, target)
            P = lodo_predictions(X, T, kind)
            add(f"chooser: {kind} / {target}", P)
            print("done", kind, target)

    res = pd.DataFrame(rows).round(4)
    print("\n=== Leave-one-dataset-out results (", n, "datasets ) ===")
    print(res.to_string(index=False))
    res.to_csv(RESULTS_DIR / "eval_results.csv", index=False)

    print("\n=== Chooser vs always-lgbm (mean regret difference, 95% bootstrap CI; negative = chooser better) ===")
    for name in [r["method"] for r in rows if r["method"].startswith("chooser")]:
        m, lo, hi = bootstrap_diff(regrets[name], regrets["always lgbm"])
        better = (regrets[name] < regrets["always lgbm"]).sum()
        worse = (regrets[name] > regrets["always lgbm"]).sum()
        print(f"{name:<28} diff={m:+.4f}  CI=[{lo:+.4f}, {hi:+.4f}]  better on {better}, worse on {worse}")

    out = df[["dataset_id", "dataset_name"]].copy()
    out["true_best"] = [ALGOS[i] for i in Sv.argmax(axis=1)]
    for name, reg in regrets.items():
        out["regret " + name] = reg
    out.to_csv(RESULTS_DIR / "per_dataset_regret.csv", index=False)


if __name__ == "__main__":
    main()