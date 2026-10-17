import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.benchmark import make_models, prepare
from src.config import CV_FOLDS, MAX_FEATURES, MAX_ROWS, SEED
from src.metafeatures import compute_meta_features

warnings.filterwarnings("ignore")

BUNDLE_PATH = Path(__file__).resolve().parent / "final_models.joblib"
REQUIRED_KEYS = ["feature_cols", "risk_clf", "band_edges", "ranker", "algos", "band_table",
                 "band_names", "base_rate", "size_stats", "n_train", "sklearn_version"]

NAMES = {"logreg": "Logistic Regression", "knn": "kNN", "rf": "Random Forest",
         "extratrees": "Extra Trees", "lgbm": "LightGBM", "svm": "SVM (RBF)"}

# Numbers from the held-out experiments (159 datasets not used to build anything).
# Source: results/shortlist_output.txt and results/confirm_output.txt
PAIR_CLOSE, PAIR_SAVED, LGBM_CLOSE = 0.887, 0.748, 0.767
CHOOSER_REGRET, LGBM_REGRET = 0.0082, 0.0084
FIXED_PAIR_REGRET, SEL_PAIR_REGRET = 0.0030, 0.0040
SEL_CLOSE, SEL_SAVED = 0.874, 0.585
N_UNSEEN = 159


@st.cache_resource
def load_bundle():
    return joblib.load(BUNDLE_PATH)


def read_csv(file):
    """Read an uploaded CSV; show a friendly error and return None if it can't be read."""
    try:
        try:
            return pd.read_csv(file)
        except UnicodeDecodeError:
            file.seek(0)
            return pd.read_csv(file, encoding="latin-1")
    except pd.errors.EmptyDataError:
        st.error("This file is empty (0 bytes or no columns). Check that the CSV was saved with data in it, then upload it again.")
    except pd.errors.ParserError as e:
        st.error(f"Could not parse this file as a CSV: {e}")
    return None


def load_example(name):
    loader = {"Breast cancer (569 rows)": load_breast_cancer,
              "Wine (178 rows, small)": load_wine,
              "Digits (1,797 rows, 10 classes)": load_digits}[name]
    return loader(as_frame=True).frame, "target"


def analyze(df, target, drop_cols, bundle):
    """Check the data, compute meta-features, and score LightGBM risk."""
    notes = []
    df = df.drop(columns=drop_cols).dropna(subset=[target]).reset_index(drop=True)
    y, X = df[target], df.drop(columns=[target])
    n_orig = len(df)

    if X.shape[1] < 1:
        raise ValueError("No feature columns left after removing the target and dropped columns.")
    if X.shape[1] > 1000:
        raise ValueError("More than 1,000 feature columns; this tool is not built for data that wide.")
    n_classes = y.nunique()
    if n_classes < 2 or n_classes > 30:
        raise ValueError(f"The target has {n_classes} distinct values. This tool handles classification "
                         "with 2 to 30 classes (a numeric target with many values looks like regression).")
    counts = y.value_counts()
    if counts.min() < 5:
        raise ValueError(f"Class '{counts.idxmin()}' has only {counts.min()} rows. Each class needs at least 5.")
    if n_orig < 100:
        raise ValueError(f"Only {n_orig} rows. Need at least 100.")

    if n_orig < 300:
        notes.append(f"Only {n_orig} rows: below the 300-row minimum of the datasets used to build the models.")
    if X.shape[1] > MAX_FEATURES:
        notes.append(f"{X.shape[1]} features: more than the {MAX_FEATURES} seen when building the models, "
                     "so the risk estimate is an extrapolation.")
    id_like = [c for c in X.columns if X[c].nunique() == len(X) and not pd.api.types.is_float_dtype(X[c])]
    if id_like:
        notes.append("These columns look like IDs (every value unique) and may hurt every model: "
                     + ", ".join(map(str, id_like[:5])) + ". Consider dropping them above.")

    if n_orig > MAX_ROWS:
        X, _, y, _ = train_test_split(X, y, train_size=MAX_ROWS, stratify=y, random_state=SEED)
        X, y = X.reset_index(drop=True), y.reset_index(drop=True)
        notes.append(f"{n_orig:,} rows: a stratified sample of {MAX_ROWS:,} is used, "
                     "the same cap as in the study.")

    feats = compute_meta_features(X, y)
    row = pd.DataFrame([feats])[bundle["feature_cols"]].fillna(0)
    p = float(bundle["risk_clf"].predict_proba(row)[0, 1])
    band = int(np.digitize(p, bundle["band_edges"]))
    order = np.argsort(-bundle["ranker"].predict(row)[0])
    ranking = [bundle["algos"][i] for i in order]
    majority = float(y.value_counts(normalize=True).iloc[0])

    return {"X": X, "y": y, "feats": feats, "band": band, "ranking": ranking, "row": row,
            "notes": notes, "n_orig": n_orig, "majority": majority}


def run_verify(X, y, which, fixed, selector):
    Xp, prep = prepare(X)
    yy = LabelEncoder().fit_transform(y)
    cv = StratifiedKFold(n_splits=min(CV_FOLDS, int(np.bincount(yy).min())), shuffle=True, random_state=SEED)
    models = make_models()
    rows, bar = [], st.progress(0.0)
    for i, name in enumerate(which):
        t0 = time.time()
        sc = cross_val_score(Pipeline([("prep", prep), ("model", models[name])]), Xp, yy,
                             cv=cv, scoring="balanced_accuracy")
        src = " + ".join(x for x, ok in [("fixed pair", name in fixed), ("selector top-2", name in selector)] if ok)
        rows.append({"Algorithm": NAMES[name], "Balanced accuracy": round(sc.mean(), 4),
                     "Std across folds": round(sc.std(), 4), "Time (s)": round(time.time() - t0, 1),
                     "Suggested by": src, "_key": name})
        bar.progress((i + 1) / len(which))
    bar.empty()
    return pd.DataFrame(rows).sort_values("Balanced accuracy", ascending=False).reset_index(drop=True)


def verdict(v):
    """Plain-language, noise-aware comparison of the top two results."""
    if len(v) < 2:
        return None
    a, b = v.iloc[0], v.iloc[1]
    gap = float(a["Balanced accuracy"] - b["Balanced accuracy"])
    spread = float(max(a["Std across folds"], b["Std across folds"]))
    if gap <= spread:
        return (f"{a['Algorithm']} scored {gap * 100:.1f} points higher than {b['Algorithm']}; the fold-to-fold "
                f"spread is about {spread * 100:.1f} points, so treat this as a near tie.")
    return (f"{a['Algorithm']} scored {gap * 100:.1f} points higher than {b['Algorithm']}, which is larger than the "
            f"fold-to-fold spread (about {spread * 100:.1f} points), so the lead looks real on this data.")


# ----------------------------------------------------------------------------- page
st.set_page_config(page_title="Which model should I try first?", layout="centered")
bundle = load_bundle()
missing = [k for k in REQUIRED_KEYS if k not in bundle]
if missing:
    st.error(f"final_models.joblib is missing expected entries: {missing}. "
             "Re-run `python -m src.final_models` or tell the developer.")
    st.stop()
if sklearn.__version__ != bundle["sklearn_version"]:
    st.warning(f"Models were saved with scikit-learn {bundle['sklearn_version']} but this app runs "
               f"{sklearn.__version__}. Results may be unreliable.")

st.title("Which model should I try first?")
st.caption(
    f"Built from a study of {bundle['n_train']} OpenML tabular classification datasets. This tool does not claim to "
    "pick the single best model: in held-out tests, no learned selector beat simple fixed defaults. It profiles your "
    "data, flags when LightGBM is more likely to be beaten, and lets you check a cheap two-model shortlist on your own data."
)

source = st.radio("Data", ["Upload a CSV", "Use an example dataset"], horizontal=True)
df, default_target, label = None, None, None
if source == "Upload a CSV":
    up = st.file_uploader("CSV file (classification: one column is the label)", type="csv")
    if up is not None:
        df, label = read_csv(up), up.name
else:
    ex = st.selectbox("Example", ["Breast cancer (569 rows)", "Wine (178 rows, small)",
                                  "Digits (1,797 rows, 10 classes)"])
    df, default_target = load_example(ex)
    label = ex

if df is not None:
    cols = list(df.columns)
    target = st.selectbox("Target column (what you want to predict)", cols,
                          index=cols.index(default_target) if default_target in cols else len(cols) - 1)
    drop_cols = st.multiselect("Columns to ignore (IDs, free text, dates)", [c for c in cols if c != target])
    st.caption(f"{len(df):,} rows, {len(cols)} columns")

    if st.button("Analyze dataset", type="primary"):
        key = (label, target, tuple(drop_cols))
        try:
            with st.spinner("Profiling the dataset (a few seconds)..."):
                st.session_state["analysis"] = analyze(df, target, drop_cols, bundle)
                st.session_state["key"] = key
                st.session_state["verify"] = None
        except ValueError as e:
            st.session_state["analysis"] = None
            st.error(str(e))

a = st.session_state.get("analysis")
if a is not None:
    f = a["feats"]
    for n in a["notes"]:
        st.warning(n)

    st.subheader("1. Dataset profile")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows used", f"{int(f['n_rows']):,}")
    c2.metric("Features", int(f["n_features"]))
    c3.metric("Classes", int(f["n_classes"]))
    c4.metric("Largest class", f"{a['majority']:.0%}")
    c5, c6, _, _ = st.columns(4)
    c5.metric("Categorical features", f"{f['frac_categorical']:.0%}")
    c6.metric("Missing values", f"{f['missing_frac']:.1%}")

    st.subheader("2. Is LightGBM likely to be beaten?")
    rec = bundle["band_table"][a["band"]]
    name = bundle["band_names"][a["band"]]
    text = (f"**Risk: {name}.** Among unseen datasets that our risk model put in this band, LightGBM was beaten by more "
            f"than 1 point of balanced accuracy on **{rec['observed_rate']:.0%}** (95% interval "
            f"{rec['ci_low']:.0%} to {rec['ci_high']:.0%}, {rec['n_datasets']} datasets). "
            f"Across all {N_UNSEEN} unseen datasets that happened {bundle['base_rate']:.0%} of the time.")
    {"Low": st.success, "Medium": st.warning, "High": st.error}[name](text)
    if rec["n_datasets"] < 20:
        st.caption(f"Only {rec['n_datasets']} unseen datasets fell in this band, so the rate above is rough.")

    ss = bundle["size_stats"]
    if f["n_rows"] <= 1500:
        st.info(f"Size: {int(f['n_rows']):,} rows. On unseen datasets of 1,500 rows or fewer, a simple model "
                f"(logistic regression, SVM or kNN) was the best of the six on {ss['small_share']:.0%} of them, "
                f"against {ss['large_share']:.0%} for larger datasets. This is about how often a simple model wins, "
                "not by how much, and which simple model wins varies (logistic regression, SVM or kNN), so this "
                "is not advice to switch to logistic regression.")
    else:
        st.info(f"Size: {int(f['n_rows']):,} rows. On unseen datasets above 1,500 rows, a simple model was the best "
                f"of the six on {ss['large_share']:.0%} of them, against {ss['small_share']:.0%} for smaller datasets.")

    imp = pd.Series(bundle["risk_clf"].feature_importances_, index=bundle["feature_cols"]).sort_values(ascending=False)
    with st.expander("What the risk model looks at"):
        top = imp.head(5)
        st.dataframe(pd.DataFrame({"Meta-feature": top.index, "Importance (overall)": top.round(3).values,
                                   "Your dataset": [round(float(f[k]), 3) for k in top.index]}),
                     hide_index=True)
        st.caption("Importances describe the model overall, not why it scored your particular dataset.")

    st.subheader("3. Two algorithms to try")
    fixed = ["logreg", "lgbm"]
    selector = a["ranking"][:2]
    st.markdown(
        f"**Start with Logistic Regression + LightGBM.** Across {N_UNSEEN} unseen datasets, this pair came within 1 "
        f"point of the best of all six models on **{PAIR_CLOSE:.0%}** of them (LightGBM alone: {LGBM_CLOSE:.0%}), and "
        f"saved about **{PAIR_SAVED:.0%}** of the training time compared with training all six (average of measured "
        "times). This pair was picked on the discovery datasets; the comparison on unseen data is exploratory.")
    st.markdown(
        f"**Dataset-specific experimental suggestion: {NAMES[selector[0]]} + {NAMES[selector[1]]}.** In held-out tests "
        f"this per-dataset prediction did slightly worse than the fixed pair: mean regret {SEL_PAIR_REGRET} vs "
        f"{FIXED_PAIR_REGRET}, within 1 point of the best on {SEL_CLOSE:.0%} vs {PAIR_CLOSE:.0%} of datasets, and "
        f"about {SEL_SAVED:.0%} vs {PAIR_SAVED:.0%} training time saved. The regret difference is not statistically "
        "significant, which is why the fixed pair comes first. Verify below to see which works better on your data.")

    also = st.checkbox("Also train the remaining models (slower)")
    if st.button("Verify on my data"):
        which = list(dict.fromkeys(fixed + selector))
        if also:
            which += [m for m in bundle["algos"] if m not in which]
        with st.spinner("Training with 5-fold cross-validation..."):
            st.session_state["verify"] = run_verify(a["X"], a["y"], which, fixed, selector)
    v = st.session_state.get("verify")
    if v is not None:
        st.dataframe(v.drop(columns=["_key"]), hide_index=True)
        msg = verdict(v)
        if msg:
            st.info(msg)
        st.caption("Cross-validated balanced accuracy on the data above (a stratified sample of at most 5,000 rows "
                   "if yours is larger). Default hyperparameters, as in the study.")

    with st.expander("Experimental: six-way ranking from the selector (did not beat fixed defaults)"):
        st.write("Predicted order, best first: " + " > ".join(NAMES[k] for k in a["ranking"]))
        st.caption(f"On unseen datasets this selector's mean regret was {CHOOSER_REGRET} against {LGBM_REGRET} for "
                   "always using LightGBM, a difference that is not statistically significant, and a fixed "
                   "Logistic Regression + LightGBM pair did better than its top-2. Treat it as a curiosity, not advice.")

with st.expander("About the study and its limits"):
    st.markdown(
        f"- {bundle['n_train']} OpenML classification datasets, six algorithms (logistic regression, kNN, random forest, "
        "extra trees, LightGBM, RBF-SVM), default hyperparameters, 5-fold balanced accuracy.\n"
        f"- Reported results come from models trained on 147 datasets and tested on {N_UNSEEN} unseen ones. The models "
        f"in this app were retrained on all {bundle['n_train']} and have no separate test.\n"
        "- Datasets are capped at 5,000 rows and 200 features; classification only; OpenML contains near-duplicate "
        "dataset families, which may make the held-out results look better than they are.\n"
        "- Training times are from one machine, so the compute savings are relative estimates.")