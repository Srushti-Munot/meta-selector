import itertools
import warnings

import numpy as np
import pandas as pd

from src.config import RESULTS_DIR
from src.evaluate import bootstrap_diff
from src.meta_model import ALGOS, load_meta_dataset, make_meta_model, make_targets

warnings.filterwarnings("ignore")

DISCOVERY_CSV = RESULTS_DIR / "meta_dataset_discovery.csv"
SCORES_CSV = RESULTS_DIR / "benchmark_scores.csv"
PAIRS = list(itertools.combinations(range(len(ALGOS)), 2))


def stats(sel, Sv, Tv):
    """sel = array (n_datasets x k) of chosen algorithm indices.
    Returns per-dataset: regret, compute saved, and whether the true best is in the shortlist."""
    rows = np.arange(len(Sv))[:, None]
    regret = Sv.max(axis=1) - Sv[rows, sel].max(axis=1)
    saved = 1 - Tv[rows, sel].sum(axis=1) / Tv.sum(axis=1)
    hit = (sel == Sv.argmax(axis=1)[:, None]).any(axis=1).astype(float)
    return regret, saved, hit


def summarize(name, regret, saved, hit):
    return {"method": name, "mean_regret": regret.mean(), "close_within_1pct": (regret <= 0.01).mean(),
            "true_best_in_shortlist": hit.mean(), "mean_compute_saved": saved.mean(),
            "median_compute_saved": np.median(saved)}


def main():
    disc_df, Xd, Sd = load_meta_dataset(DISCOVERY_CSV)
    full_df, Xf, Sf = load_meta_dataset()
    is_conf = ~full_df.dataset_id.isin(disc_df.dataset_id) & Sf.notna().all(axis=1)
    Xc = Xf[is_conf][Xd.columns].reset_index(drop=True)
    Sc = Sf[is_conf].reset_index(drop=True)
    conf_ids = full_df.dataset_id[is_conf].tolist()

    times = pd.read_csv(SCORES_CSV).pivot_table(index="dataset_id", columns="algorithm", values="time_sec")[ALGOS]
    Td, Tc = times.loc[disc_df.dataset_id].to_numpy(), times.loc[conf_ids].to_numpy()
    Svd, Svc = Sd.to_numpy(), Sc.to_numpy()
    n = len(Svc)

    # ---- choose the fixed baselines using ONLY discovery data ----
    pair_regret = {p: stats(np.tile(p, (len(Svd), 1)), Svd, Td)[0].mean() for p in PAIRS}
    best_pair = min(pair_regret, key=pair_regret.get)
    mean_time = Td.mean(axis=0)
    cheap_pair = tuple(np.argsort(mean_time)[:2])
    name = lambda p: " + ".join(ALGOS[i] for i in p)
    print(f"Best fixed pair (picked on discovery): {name(best_pair)}")
    print(f"Cheapest pair (picked on discovery):   {name(cheap_pair)}")
    print("Mean seconds per algorithm (discovery): " + ", ".join(f"{a}={t:.1f}" for a, t in zip(ALGOS, mean_time)) + "\n")

    # ---- chooser trained on discovery, shortlist on confirmation ----
    model = make_meta_model("rf").fit(Xd, make_targets(Sd, "relative"))
    P = model.predict(Xc)
    chooser_sel = np.argsort(-P, axis=1)[:, :2]
    lgbm_share = (chooser_sel == ALGOS.index("lgbm")).any(axis=1).mean()

    rows, reg = [], {}

    def add(label, sel):
        r, s, h = stats(sel, Svc, Tc)
        rows.append(summarize(label, r, s, h))
        reg[label] = r

    rows.append({"method": "train all 6", "mean_regret": 0.0, "close_within_1pct": 1.0, "true_best_in_shortlist": 1.0,
                 "mean_compute_saved": 0.0, "median_compute_saved": 0.0})
    add("lgbm only (k=1)", np.full((n, 1), ALGOS.index("lgbm")))
    add("CHOOSER top-2", chooser_sel)
    add(f"best fixed pair ({name(best_pair)})", np.tile(best_pair, (n, 1)))
    add(f"cheapest pair ({name(cheap_pair)})", np.tile(cheap_pair, (n, 1)))

    # random pair = average over all 15 possible pairs
    allp = [stats(np.tile(p, (n, 1)), Svc, Tc) for p in PAIRS]
    rows.append(summarize("random pair (avg of 15)", *[np.mean([a[i] for a in allp], axis=0) for i in range(3)]))

    res = pd.DataFrame(rows).round(4)
    print(f"Results on {n} unseen (confirmation) datasets  [chooser trained on discovery only]")
    print(res.to_string(index=False))
    res.to_csv(RESULTS_DIR / "shortlist_results.csv", index=False)
    print(f"\nChooser's shortlist contains LightGBM on {lgbm_share:.0%} of datasets.")

    bp = f"best fixed pair ({name(best_pair)})"
    d, lo, hi = bootstrap_diff(reg["CHOOSER top-2"], reg[bp])
    print(f"\nE1: chooser top-2 regret minus best-fixed-pair regret = {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print("    -> " + ("SUPPORTED (chooser better)" if hi < 0 else "NOT SUPPORTED (no evidence the chooser beats a fixed pair)"))


if __name__ == "__main__":
    main()