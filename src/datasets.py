import joblib
import openml
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RAW_DIR, RESULTS_DIR, SEED, MAX_ROWS, MAX_FEATURES


def get_cc18_ids():
    """IDs of the 72 datasets in the OpenML-CC18 benchmark suite."""
    return list(openml.study.get_suite("OpenML-CC18").data)


def load_dataset(did):
    """Download (or load from cache) one dataset. Returns dict, or None if filtered out."""
    cache = RAW_DIR / f"{did}.joblib"
    if cache.exists():
        return joblib.load(cache)

    ds = openml.datasets.get_dataset(did)
    target = ds.default_target_attribute
    if not target or "," in target:   # no target, or multi-target -> skip
        return None
    X, y, cat_indicator, names = ds.get_data(target=target)

    # filters
    if y is None or y.nunique() < 2 or X.shape[1] > MAX_FEATURES:
        return None

    n_original = len(X)
    if n_original > MAX_ROWS:
        try:
            X, _, y, _ = train_test_split(
                X, y, train_size=MAX_ROWS, stratify=y, random_state=SEED
            )
        except ValueError:  # a class too small to stratify
            X, _, y, _ = train_test_split(X, y, train_size=MAX_ROWS, random_state=SEED)

    out = {
        "id": did,
        "name": ds.name,
        "X": X.reset_index(drop=True),
        "y": y.reset_index(drop=True),
        "n_rows_original": n_original,
    }
    joblib.dump(out, cache)
    return out

def get_extra_ids(n=250):
    """Random sample of extra OpenML classification datasets (saved so reruns are identical)."""
    path = RESULTS_DIR / "extra_ids.txt"
    if path.exists():
        return [int(x) for x in path.read_text().split()]

    df = openml.datasets.list_datasets(status="active")
    if not isinstance(df, pd.DataFrame):
        df = openml.datasets.list_datasets(status="active", output_format="dataframe")

    cc18 = get_cc18_ids()
    cc18_names = set(df[df.did.isin(cc18)].name)
    df = df[~df.name.isin(cc18_names)]          # drop CC18 datasets and their other versions
    df = df[
        (df.NumberOfClasses >= 2) & (df.NumberOfClasses <= 30)
        & (df.NumberOfInstances >= 300) & (df.NumberOfInstances <= 100_000)
        & (df.NumberOfFeatures >= 3) & (df.NumberOfFeatures <= MAX_FEATURES)
        & (df.MinorityClassSize >= 20)
    ]
    df = df.drop_duplicates(subset="name")      # OpenML has many versions of the same data
    ids = df.sample(min(n, len(df)), random_state=SEED).did.astype(int).tolist()
    path.write_text("\n".join(map(str, ids)))
    return ids

if __name__ == "__main__":
    # quick test: grab the first 5 datasets that pass the filters
    got = []
    for did in get_cc18_ids():
        d = load_dataset(did)
        if d is None:
            print(f"skip {did}")
            continue
        print(f"{d['id']:>6} {d['name']:<25} X={d['X'].shape} classes={d['y'].nunique()}")
        got.append(d)
        if len(got) == 5:
            break