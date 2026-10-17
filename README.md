# Which model should I try first?

### A meta-learning experiment on 306 tabular datasets (and what happened when it didn't work)

**[Try it here!](https://meta-selector.streamlit.app/)**

**Short version:** I tried to build a model that picks the best ML algorithm for a new dataset. It didn't significantly beat just using LightGBM. But the failure taught me when LightGBM actually loses, and a simple two-model pair turned out to be a strong shortcut (an exploratory result).

---

## How this started

Every time I start a classification problem I hit the same question: *which model do I use?* Logistic regression, random forest, LightGBM, SVM... and the usual answer is "just try them all."

So I wondered: **what if there was a model that helps you select the model?** Look at your dataset (how many rows, how many columns, how unbalanced the classes are), and predict which algorithm will do well, without training all of them.

It turns out this is a real research area called *meta-learning* (tools like auto-sklearn use ideas from it), so I'm not claiming I invented anything. What I wanted to find out was how well the idea actually works when you test it properly.

## What I did

1. **Collected data.** 306 classification datasets from OpenML: 59 usable ones from the CC18 benchmark suite plus 247 randomly sampled others. Datasets with more than 200 features were excluded, and larger ones were sampled down to 5,000 rows.
2. **Benchmarked six models** on every dataset: logistic regression, kNN, random forest, extra trees, LightGBM and an RBF-SVM, with near-default settings and 5-fold cross-validation (balanced accuracy).
3. **Measured 20 cheap facts about each dataset** ("meta-features"): rows, columns, class balance, skewness, a few quick test models, and so on.
4. **Trained a selector** to predict the best model from those facts.
5. **Kept myself honest.** I explored on 147 datasets, wrote my hypotheses down, and only then tested them on 159 datasets I hadn't looked at.

## What happened (the honest part)

**The selector didn't work.** On the 159 unseen datasets it was not significantly better than "always use LightGBM". LightGBM is already within 1 accuracy point of the best model on about 77% of datasets, so there probably isn't much room to improve.

But the failure was useful. Asking *where does LightGBM lose?* led to the real findings:

| Question | Result on the 159 unseen datasets |
|---|---|
| Do simple models (logreg / SVM / kNN) win more on small datasets? | **Yes.** 57% on datasets with 1,500 rows or fewer vs 24% on larger ones (p = 0.0002) |
| When LightGBM loses, is a simple model usually the winner? | **Yes.** 68% of its losses (25 of 37) |
| Does the selector beat always-LightGBM? | **No significant difference.** Regret 0.0082 vs 0.0084 (95% CI for the difference: −0.0016 to +0.0013) |
| Can we predict *when* LightGBM will be beaten? | **Yes, modestly.** AUC 0.775 [0.694, 0.847] at the 1-point threshold I set in advance |
| Does a fixed **Logistic Regression + LightGBM** pair work anyway? | **Yes** (exploratory). Within 1 point of the best of six on 89% of datasets, with about 75% less measured training time on average (5-fold CV fit and predict time, one machine) |

*Regret = how much accuracy you give up compared with the best of the six on that dataset. Lower is better.*

### What didn't work (also worth knowing)

- The selector I built never significantly beat the LightGBM default.
- "Use logistic regression on small data" is **bad advice**: it scored worse than LightGBM on average, probably because which simple model wins changes from dataset to dataset.
- Predicting the exact replacement for LightGBM and switching to it gained nothing on average. Small wins were cancelled by bigger mistakes.

### How I'd describe it

Not "an AI that picks your model". More like: *a measured study of when a strong default breaks, with an interactive tool built around the evidence.* "A boosted-tree model plus a linear baseline is a strong default" is something practitioners already believe. What this adds is numbers, effect sizes and confidence intervals across 306 datasets.

---

## The app

A small Streamlit app built on the study. Upload a CSV, pick the target column, and it:

- profiles your data,
- shows how often LightGBM was beaten on unseen benchmark datasets in the same risk band (low / medium / high: 8% / 32% / 62%, from 74 / 72 / 13 datasets, so the high band is small), with intervals. These rates come from OpenML datasets and may not transfer to your data,
- suggests the Logistic Regression + LightGBM pair to try first,
- lets you check the shortlist on your own data with cross-validation.

There's also a **The study** tab that tells the story above in plain words.

The app doesn't claim to pick the best model. Cross-validation scores describe your uploaded file only and don't guarantee performance on future data. The app code doesn't store uploads, but the hosting platform may keep logs, so please don't upload sensitive data to a public demo.

```bash
streamlit run app/app.py
```

---

## Technical details

**Design.** Pattern-finding on 147 "discovery" datasets only. Four hypotheses written down, then trained on discovery and tested once on 159 "confirmation" datasets. Later experiments (fixed pair, threshold sensitivity, replacement prediction) were run after the confirmation set had been used, so they are labelled **exploratory**.

**Models.** Six sklearn/LightGBM classifiers with near-default settings. Preprocessing is fitted inside each CV fold (median impute + scaling for numeric, most-frequent impute + one-hot for categorical). Datasets over 5,000 rows are stratified-subsampled to 5,000; datasets with more than 200 features were excluded.

**Selector.** Random forest regressor predicting each algorithm's score relative to the dataset mean, evaluated leave-one-dataset-out and on the held-out set. The "LightGBM loses" classifier is a random forest with the label "LightGBM is more than 1 point behind the best of six". Risk bands (cutoffs 0.25 / 0.50, fixed in advance) are reported as observed loss rates (8% / 32% / 62%), not raw probabilities, since forest probabilities aren't calibrated.

**Robustness.** The simple-model result doesn't depend on that threshold. The "LightGBM loses" relationships kept their direction at 0.5-, 1- and 2-point definitions. On unseen data the link between dataset size and LightGBM losses (n_rows AUC) was significant at all three, while the plain small-vs-large loss-rate comparison pointed the same way but was not significant (p about 0.10 to 0.12). The 2-point setting had only 19 test losses, so its interval is wide. Loss-prediction AUC ranged from 0.65 to 0.78 across them, so that range is the fair summary.

### Limitations

- OpenML has near-duplicate datasets, so the 306 aren't fully independent. This may flatter held-out results.
- The 95% interval for the selector's regret difference versus always-LightGBM was roughly −0.16 to +0.13 accuracy points, so effects of that size can't be ruled out either way.
- Near-default hyperparameters only; tuning could change the conclusions (not tested).
- Classification only, balanced accuracy only, datasets sampled down to 5,000 rows, training times from one machine.
- The 1,500-row cutoff was chosen after looking at the discovery data.
- The fixed pair was picked from 15 candidate pairs on the discovery set. Clean, but exploratory.
- The deployed app models were retrained on all 306 datasets, so the numbers above come from the 147-train / 159-test setup, not from the app's own models.

### Reproduce

Python 3.13.

```bash
python -m venv venv
venv\Scripts\activate            # Windows (use `source venv/bin/activate` on Mac/Linux)
pip install -r requirements.txt
```

All results are already saved in `results/` (every number above comes from the `*_output.txt` files there), so you can read them without re-running anything. To re-run the analysis scripts (fast, fixed seeds):

```bash
python -m src.analyze
python -m src.confirm
python -m src.shortlist
python -m src.sensitivity
python -m src.replacement lodo     # slowest, a few minutes
python -m src.final_models         # retrains the app models
```

The benchmark itself (`python -m src.benchmark cc18|extra`, then `python -m src.metafeatures`) downloads datasets from OpenML and takes a long time. `results/meta_dataset_discovery.csv` defines which datasets are "discovery", and `results/extra_ids.txt` pins which extra OpenML datasets were sampled. Please don't overwrite them.

### Repository layout

```
src/        benchmark, meta-features, selector, and one script per experiment
results/    benchmark scores, meta-features, saved outputs of every experiment
app/        Streamlit app and the trained models
```

---

*Built as a learning project. Feedback and corrections are welcome.*