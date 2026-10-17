import html
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

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --paper: #FAF6EE; --card: #FFFDF8; --ink: #2B2420; --muted: #7A6E63;
  --line: #E4DACA; --accent: #B4532A; --accent-soft: #F3E3D6;
  --low: #5B7B4A; --low-bg: #EEF2E6; --med: #B07A1E; --med-bg: #F8EFD8;
  --high: #A33A2B; --high-bg: #F6E1DB;
}
html, body, [data-testid="stAppViewContainer"], .stApp { background: var(--paper); color: var(--ink); }
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { max-width: 760px; padding-top: 3.2rem; padding-bottom: 5rem; }
html, body, p, li, label, .stMarkdown, [data-testid="stCaptionContainer"] {
  font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;
}
h1, h2, h3, .serif { font-family: 'Source Serif 4', Georgia, 'Times New Roman', serif !important; color: var(--ink); letter-spacing: -0.01em; }
h1 { font-size: 2.5rem !important; font-weight: 600 !important; line-height: 1.12 !important; margin-bottom: .2rem; }
h2 { font-size: 1.55rem !important; font-weight: 600 !important; margin-top: 2.4rem !important; }
h3 { font-size: 1.2rem !important; font-weight: 600 !important; margin-top: 1.6rem !important; }
.kicker { font-family: 'JetBrains Mono', monospace; font-size: .72rem; letter-spacing: .14em;
  text-transform: uppercase; color: var(--accent); margin-bottom: .6rem; }
.lede { font-family: 'Source Serif 4', Georgia, serif; font-size: 1.12rem; line-height: 1.6; color: #4a3f37; margin: .6rem 0 1.4rem; }
.section-label { font-family: 'JetBrains Mono', monospace; font-size: .72rem; letter-spacing: .14em;
  text-transform: uppercase; color: var(--muted); border-top: 1px solid var(--line);
  padding-top: .8rem; margin: 2.4rem 0 .8rem; }
.callout { border: 1px solid var(--line); border-left: 4px solid var(--accent); background: var(--card);
  border-radius: 4px; padding: .95rem 1.1rem; margin: .6rem 0 1rem; line-height: 1.55; font-size: .95rem; }
.callout .t { font-family: 'JetBrains Mono', monospace; font-size: .72rem; letter-spacing: .12em;
  text-transform: uppercase; margin-bottom: .35rem; }
.callout.low { border-left-color: var(--low); background: var(--low-bg); } .callout.low .t { color: var(--low); }
.callout.medium { border-left-color: var(--med); background: var(--med-bg); } .callout.medium .t { color: var(--med); }
.callout.high { border-left-color: var(--high); background: var(--high-bg); } .callout.high .t { color: var(--high); }
.callout.note .t { color: var(--accent); }
.callout.pick { border-left-color: var(--accent); background: var(--accent-soft); }
.callout.pick .t { color: var(--accent); }
.callout.exp { border-left-color: var(--line); } .callout.exp .t { color: var(--muted); }
[data-testid="stMetric"] { background: var(--card); border: 1px solid var(--line); border-radius: 4px; padding: .7rem .9rem; }
[data-testid="stMetricLabel"] p { font-family: 'JetBrains Mono', monospace; font-size: .68rem !important;
  letter-spacing: .1em; text-transform: uppercase; color: var(--muted); }
[data-testid="stMetricValue"] { font-family: 'Source Serif 4', Georgia, serif; font-size: 1.7rem; }
.stButton > button { border-radius: 3px; border: 1px solid var(--ink); background: var(--card); color: var(--ink);
  font-family: 'Inter', sans-serif; font-weight: 500; }
.stButton > button:hover { border-color: var(--accent); color: var(--accent); }
.stButton > button[kind="primary"] { background: var(--ink); color: var(--paper); }
.stButton > button[kind="primary"]:hover { background: var(--accent); color: #fff; border-color: var(--accent); }
button[data-baseweb="tab"] { font-family: 'JetBrains Mono', monospace; font-size: .8rem; letter-spacing: .08em; text-transform: uppercase; }
[data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 4px; background: var(--card); }
table { font-size: .9rem; } th { font-family: 'JetBrains Mono', monospace; font-size: .72rem !important;
  letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
.step { display: flex; gap: 1rem; margin: 1.1rem 0; }
.step .n { font-family: 'Source Serif 4', Georgia, serif; font-size: 1.6rem; color: var(--accent); min-width: 2rem; line-height: 1.2; }
.step .b { line-height: 1.6; }
.step .b b { font-family: 'Source Serif 4', Georgia, serif; font-size: 1.05rem; }
</style>
"""


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


def callout(kind, title, body_html):
    """Styled box. body_html may contain our own <b> tags; never pass user text into it unescaped."""
    st.markdown(f'<div class="callout {kind}"><div class="t">{html.escape(title)}</div>{body_html}</div>',
                unsafe_allow_html=True)


def label(text):
    st.markdown(f'<div class="section-label">{html.escape(text)}</div>', unsafe_allow_html=True)


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
                     "Suggested by": src})
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


# ----------------------------------------------------------------------------- tab: try it
def render_try(bundle):
    st.markdown('<div class="lede">Upload a table with a category column to predict. The tool measures a few '
                "basic facts about it, estimates whether LightGBM may be beaten, based on patterns in the benchmark, and lets you test a "
                "cheap two-model shortlist on your own data.</div>", unsafe_allow_html=True)

    label("Your data")
    source = st.radio("Data", ["Upload a CSV", "Use an example dataset"], horizontal=True,
                      label_visibility="collapsed")
    df, default_target, name_label = None, None, None
    if source == "Upload a CSV":
        up = st.file_uploader("CSV file (classification: one column is the label)", type="csv")
        if up is not None:
            df, name_label = read_csv(up), up.name
    else:
        ex = st.selectbox("Example", ["Breast cancer (569 rows)", "Wine (178 rows, small)",
                                      "Digits (1,797 rows, 10 classes)"])
        df, default_target = load_example(ex)
        name_label = ex

    if df is not None:
        cols = list(df.columns)
        target = st.selectbox("Target column (what you want to predict)", cols,
                              index=cols.index(default_target) if default_target in cols else len(cols) - 1)
        drop_cols = st.multiselect("Columns to ignore (IDs, free text, dates)", [c for c in cols if c != target])
        st.caption(f"{len(df):,} rows, {len(cols)} columns")

        if st.button("Analyze dataset", type="primary"):
            key = (name_label, target, tuple(drop_cols))
            try:
                with st.spinner("Profiling the dataset (a few seconds)..."):
                    st.session_state["analysis"] = analyze(df, target, drop_cols, bundle)
                    st.session_state["key"] = key
                    st.session_state["verify"] = None
            except ValueError as e:
                st.session_state["analysis"] = None
                st.error(str(e))

    a = st.session_state.get("analysis")
    if a is None:
        return
    f = a["feats"]
    for n in a["notes"]:
        st.warning(n)

    label("1 · Dataset profile")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows used", f"{int(f['n_rows']):,}")
    c2.metric("Features", int(f["n_features"]))
    c3.metric("Classes", int(f["n_classes"]))
    c4.metric("Largest class", f"{a['majority']:.0%}")
    c5, c6, _, _ = st.columns(4)
    c5.metric("Categorical", f"{f['frac_categorical']:.0%}")
    c6.metric("Missing", f"{f['missing_frac']:.1%}")

    label("2 · Is LightGBM likely to be beaten?")
    rec = bundle["band_table"][a["band"]]
    band_name = bundle["band_names"][a["band"]]
    callout(band_name.lower(), f"Risk: {band_name}",
            f"Among unseen datasets that our risk model put in this band, LightGBM was beaten by more than 1 point "
            f"of balanced accuracy on <b>{rec['observed_rate']:.0%}</b> (95% interval {rec['ci_low']:.0%} to "
            f"{rec['ci_high']:.0%}, {rec['n_datasets']} datasets). Across all {N_UNSEEN} unseen datasets that "
            f"happened {bundle['base_rate']:.0%} of the time.")
    if rec["n_datasets"] < 20:
        st.caption(f"Only {rec['n_datasets']} unseen datasets fell in this band, so the rate above is rough.")

    ss = bundle["size_stats"]
    if f["n_rows"] <= 1500:
        callout("note", "Size",
                f"{int(f['n_rows']):,} rows. On unseen datasets of 1,500 rows or fewer, a simple model (logistic "
                f"regression, SVM or kNN) was the best of the six on <b>{ss['small_share']:.0%}</b> of them, against "
                f"{ss['large_share']:.0%} for larger datasets. This is about how often a simple model wins, not by "
                "how much, and which simple model wins varies, so it is not advice to switch to logistic regression.")
    else:
        callout("note", "Size",
                f"{int(f['n_rows']):,} rows. On unseen datasets above 1,500 rows, a simple model was the best of the "
                f"six on <b>{ss['large_share']:.0%}</b> of them, against {ss['small_share']:.0%} for smaller datasets.")

    imp = pd.Series(bundle["risk_clf"].feature_importances_, index=bundle["feature_cols"]).sort_values(ascending=False)
    with st.expander("What the risk model looks at"):
        top = imp.head(5)
        st.dataframe(pd.DataFrame({"Meta-feature": top.index, "Importance (overall)": top.round(3).values,
                                   "Your dataset": [round(float(f[k]), 3) for k in top.index]}),
                     hide_index=True)
        st.caption("Importances describe the model overall, not why it scored your particular dataset.")

    label("3 · Two algorithms to try")
    fixed = ["logreg", "lgbm"]
    selector = a["ranking"][:2]
    callout("pick", "Start here: Logistic Regression + LightGBM",
            f"Across {N_UNSEEN} unseen datasets, this pair came within 1 point of the best of all six models on "
            f"<b>{PAIR_CLOSE:.0%}</b> of them (LightGBM alone: {LGBM_CLOSE:.0%}), and saved about "
            f"<b>{PAIR_SAVED:.0%}</b> of the training time compared with training all six (average of measured "
            "times). The pair was picked on the discovery datasets, so the comparison on unseen data is exploratory.")
    callout("exp", "Experimental: the selector's own pick for your data",
            f"<b>{html.escape(NAMES[selector[0]])} + {html.escape(NAMES[selector[1]])}</b>. In held-out tests this "
            f"per-dataset prediction did slightly worse than the fixed pair: mean regret {SEL_PAIR_REGRET} vs "
            f"{FIXED_PAIR_REGRET}, within 1 point of the best on {SEL_CLOSE:.0%} vs {PAIR_CLOSE:.0%} of datasets, and "
            f"about {SEL_SAVED:.0%} vs {PAIR_SAVED:.0%} training time saved. The regret difference is not "
            "statistically significant, which is why the fixed pair comes first.")

    also = st.checkbox("Also train the remaining models (slower)")
    if st.button("Verify on my data"):
        which = list(dict.fromkeys(fixed + selector))
        if also:
            which += [m for m in bundle["algos"] if m not in which]
        with st.spinner("Training with 5-fold cross-validation..."):
            st.session_state["verify"] = run_verify(a["X"], a["y"], which, fixed, selector)
    v = st.session_state.get("verify")
    if v is not None:
        st.dataframe(v, hide_index=True)
        msg = verdict(v)
        if msg:
            callout("note", "Verdict", html.escape(msg))
        st.caption("Cross-validated balanced accuracy on the data above (a stratified sample of at most 5,000 rows "
                   "if yours is larger). Default hyperparameters, as in the study.")

    with st.expander("Experimental: six-way ranking from the selector (did not beat fixed defaults)"):
        st.write("Predicted order, best first: " + " > ".join(NAMES[k] for k in a["ranking"]))
        st.caption(f"On unseen datasets this selector's mean regret was {CHOOSER_REGRET} against {LGBM_REGRET} for "
                   "always using LightGBM, a difference that is not statistically significant, and a fixed "
                   "Logistic Regression + LightGBM pair did better than its top-2. Treat it as a curiosity, not advice.")


# ----------------------------------------------------------------------------- tab: the study
def step(n, title, body):
    st.markdown(f'<div class="step"><div class="n">{n}</div><div class="b"><b>{title}</b><br>{body}</div></div>',
                unsafe_allow_html=True)


def render_study(bundle):
    st.markdown('<div class="lede">A plain-words tour of what this project asked, how it was tested, what '
                "worked, what didn't, and how the idea changed along the way.</div>", unsafe_allow_html=True)

    label("What was the question?")
    st.markdown(
        "You have a table of data and you want a model that sorts rows into categories (spam or not, sick or "
        "healthy). There are many algorithms to pick from, and trying all of them takes time. So: **can a few "
        "cheap facts about the dataset, like how many rows it has or how lopsided the classes are, tell us which "
        "algorithm will work best, without training them all?**\n\n"
        "This idea is called *meta-learning* for algorithm selection. It is an established research area, so "
        "this project does not claim to invent it. The goal was to measure honestly how well it works.")

    label("How was it tested?")
    st.markdown(
        f"- **{bundle['n_train']} real datasets** from OpenML, a public library of machine-learning datasets.\n"
        "- **Six algorithms** on each: logistic regression, kNN, random forest, extra trees, LightGBM and an "
        "RBF-SVM, all with default settings.\n"
        "- **Score:** balanced accuracy from 5-fold cross-validation (it treats small and large classes fairly).\n"
        "- **20 cheap facts per dataset** (rows, columns, class balance, how skewed the numbers are, and a few "
        "quick test models). These are the *meta-features*.\n"
        "- **A split to keep us honest:** 147 datasets for exploring and 159 locked away for testing. We wrote our "
        "predictions down *before* opening the 159, so a lucky pattern could not sneak through.")
    with st.expander("What does \"regret\" mean?"):
        st.write("On each dataset, regret is how much accuracy you give up compared with the best of the six. "
                 "Always picking the best would give 0. Lower is better. A regret of 0.0084 means about 0.8 "
                 "accuracy points lost on average.")

    label("How the idea evolved")
    step(1, "First try: pick the single best algorithm",
         "We trained a selector to choose the best of six from the 20 facts. On the 159 unseen datasets it tied "
         "with simply always using LightGBM (regret 0.0082 vs 0.0084; the difference is too small to tell apart "
         "from luck). One reason: LightGBM is already within 1 point of the best on about 77% of datasets, so "
         "there is little room to improve.")
    step(2, "New question: where does LightGBM lose?",
         "On the 147 exploration datasets, LightGBM was beaten by more than 1 point on about 29%. When it lost, a "
         "simple model (logistic regression, SVM or kNN) was usually the winner. The one fact that stood out was "
         "dataset size: smaller datasets, more losses.")
    step(3, "Lock in four predictions, then test on the 159 unseen datasets",
         "Simple models win more often on small data: <b>yes</b> (57% vs 24%). Simple models cause most of "
         "LightGBM's losses: <b>yes</b> (68%). The selector beats LightGBM: <b>no</b>. We can predict when "
         "LightGBM will lose: <b>yes</b>, modestly (AUC 0.775, where 0.5 is a coin flip).")
    step(4, "A cheaper idea: just use a fixed pair",
         "Training only Logistic Regression + LightGBM came within 1 point of the best of all six on about 89% of "
         "unseen datasets, for roughly a quarter of the training time. It did at least as well as the learned "
         "selector. This was an exploratory check run after the main test.")
    step(5, "Is the pattern real or a fluke of one cutoff?",
         "We repeated the analysis with three definitions of \"LightGBM loses\" (0.5, 1 and 2 points). The "
         "direction held every time. The ability to predict losses ranged from AUC 0.65 to 0.78 depending on the "
         "definition, so that range is what we report.")
    step(6, "Can we name the exact replacement?",
         "Partly: logistic regression is the easiest to predict. But switching models based on those predictions "
         "gained nothing on average, because the gains on some datasets were cancelled by bigger mistakes on "
         "others.")

    label("Four questions, four answers")
    st.markdown(
        "| Question | Answer |\n|---|---|\n"
        "| Can the facts pick the best algorithm better than always-LightGBM? | **No.** A tie. |\n"
        "| Can they predict when LightGBM will be beaten? | **Yes, modestly**, mostly through dataset size. |\n"
        "| Can they identify the exact replacement? | **Partly**, and switching does not help. |\n"
        "| Does a fixed Logistic Regression + LightGBM pair work anyway? | **Yes**, within 1 point of the best on about 89% of unseen datasets. |")

    label("What did not work")
    st.markdown(
        "- The learned selector never significantly beat LightGBM.\n"
        "- \"Use logistic regression on small data\" was **worse** than LightGBM on average, because the winning "
        "simple model varies from dataset to dataset.\n"
        "- Switching to the predicted replacement was a tie at best.\n\n"
        "\"A boosted-tree model plus a linear baseline is a strong default\" is something practitioners already "
        "believe. What this project adds is measured evidence and effect sizes across 306 datasets.")

    
    label("Limits to keep in mind")
    st.markdown(
        "- OpenML contains near-duplicate datasets, so the 306 are not fully independent. That may make "
        "held-out results look slightly better than they are.\n"
        f"- {N_UNSEEN} test datasets cannot detect differences smaller than about 0.15 accuracy points.\n"
        "- Default settings only. Tuned models might change the picture, and this was not tested.\n"
        "- Datasets capped at 5,000 rows and 200 features; classification only; balanced accuracy only.\n"
        "- The 1,500-row cutoff was chosen after looking at the exploration data.\n"
        "- The models inside this app were retrained on all 306 datasets, so the numbers above come from the "
        "147-train, 159-test setup, not from the app's own models.")

    with st.expander("What does AUC mean?"):
        st.write("Pick one dataset where LightGBM lost and one where it didn't, at random. AUC is the chance that "
                 "the model gave the \"lost\" one a higher risk score. 0.5 is a coin flip and 1.0 is perfect. "
                 "0.775 means it gets that pairing right about three times in four.")


# ----------------------------------------------------------------------------- page
st.set_page_config(page_title="Which model should I try first?", page_icon=None, layout="centered")
st.markdown(CSS, unsafe_allow_html=True)

bundle = load_bundle()
missing = [k for k in REQUIRED_KEYS if k not in bundle]
if missing:
    st.error(f"final_models.joblib is missing expected entries: {missing}. "
             "Re-run `python -m src.final_models` or tell the developer.")
    st.stop()
if sklearn.__version__ != bundle["sklearn_version"]:
    st.warning(f"Models were saved with scikit-learn {bundle['sklearn_version']} but this app runs "
               f"{sklearn.__version__}. Results may be unreliable.")

st.markdown(f'<div class="kicker">Meta-learning · {bundle["n_train"]} OpenML datasets · held-out tested</div>',
            unsafe_allow_html=True)
st.title("Which model should I try first?")
st.caption("This tool does not claim to pick the single best model. In held-out tests, no learned selector beat "
           "simple fixed defaults.")

tab_try, tab_study = st.tabs(["Try it", "The study"])
with tab_try:
    render_try(bundle)
with tab_study:
    render_study(bundle)