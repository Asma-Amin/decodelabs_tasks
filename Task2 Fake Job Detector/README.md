# Fraudulent Job Posting Detector

**DecodeLabs AI Engineering Internship (Batch 2026) — Project 2: Data Classification Using AI**

## Why this project (and not the usual Iris/Titanic demo)

The brief asked for a basic classification model: load a dataset, split it into
train/test, apply a classification algorithm. Almost every intern building this
milestone reaches for the Iris flowers dataset or Titanic survival dataset —
useful for learning the mechanics, but they don't say anything about the person
who built them.

I'm actively job-hunting for remote junior IT roles myself, and **fake job
postings are a real, well-documented problem** in exactly the kind of remote/
entry-level market I'm applying into (upfront "processing fees," package-
reshipping scams, "$2000/week, no experience" listings). So instead of a toy
dataset, this project builds a **fraudulent job posting classifier** — the same
category of problem researchers have studied on the real-world EMSCAD /
"Real or Fake Job Postings" dataset (17,880 postings, ~800 fraudulent), just
built end-to-end by me.

## Honesty note on the data

I don't have live internet access to pull down the original 50MB Kaggle/EMSCAD
CSV inside this environment, so `generate_dataset.py` builds a **synthetic but
realistic** dataset instead: 320 job postings with the same categories of
"red flag" signals that real fraud-detection research on this problem uses
(vague company names, no company profile/logo, generic free-email domains,
requests for upfront payment, urgent "act now" language, unrealistically high
pay for no experience, etc.), combined with randomized noise so the two
classes overlap and the task isn't trivially easy. Fraud rate is set to 35%
(real-world EMSCAD is closer to 5%) so the model has enough fraud examples to
learn from with only 320 rows — that trade-off is deliberate, not hidden.

**What this means in practice:** the results below show the pipeline working
correctly and the classifiers finding real signal, not a production-ready
scam detector. Swapping in the real EMSCAD CSV (same column philosophy: a
`description` text field + structured flags) would work with this same code
almost unchanged — that's a natural next step, noted below.

## What the project actually does

| Step (required by the brief) | Where |
|---|---|
| Load and understand a dataset | `load_data()` + EDA printout in `classify.py` |
| Split data into training and testing sets | `train_test_split(..., stratify=y)` |
| Apply a simple classification algorithm | `LogisticRegression` (baseline) |

And beyond the brief, to make it internship-portfolio strong:

- **Mixed-feature pipeline** — TF-IDF on the free-text job description
  *combined* with scaled numeric features and one-hot encoded categoricals via
  `ColumnTransformer`. Real datasets are rarely just numbers or just text.
- **Four algorithms trained and compared**, not one: Logistic Regression,
  Random Forest, Linear SVM, and a calibrated Linear SVM (for probability
  scores). Different algorithms trade off differently on this kind of
  imbalanced problem.
- **Evaluation beyond accuracy** — precision, recall, F1, ROC-AUC, and a
  confusion matrix. For fraud detection, accuracy alone is misleading: a model
  that predicts "legitimate" every time on a 65/35 split still looks 65%
  "accurate" while catching zero fraud.
- **5-fold cross-validation** on top of the single train/test split, so the
  reported performance isn't a lucky split.
- **Explainability, both global and per-prediction** — `classify.py` prints
  the words/phrases with the strongest learned weight toward "fraudulent"
  overall (e.g. `personal email`, `skip queue`, `limited time`). `predict.py`
  goes further and explains *each individual prediction*: it shows exactly
  which signals in that specific posting pushed it toward fraudulent or
  legitimate, e.g. `no company profile/about section -> FRAUDULENT`.
- **Model persistence** — the best model (by F1-score) is saved with
  `joblib` and reloaded by `predict.py` and `test_model.py`.
- **A usable CLI** (`predict.py`) — paste in a real posting's details and get
  a live prediction with a confidence score and a plain-English "why," instead
  of the model only living inside a notebook. `--demo` and `--legit-demo`
  flags score a ready-made fraud/legit example without typing anything in.
- **A sanity-test suite** (`test_model.py`) — five hand-written postings
  (2 obvious scams, 2 clearly legitimate, 1 deliberately ambiguous) checked
  against the trained model. Runs standalone (`python3 test_model.py`) or,
  if `pytest` is installed, as real parametrized test cases
  (`pytest test_model.py -v`).

## Results (this run)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Linear SVM | 0.953 | 0.913 | 0.955 | 0.933 | 0.960 |
| Logistic Regression | 0.938 | 0.909 | 0.909 | 0.909 | 0.960 |
| Calibrated Linear SVM | 0.938 | 0.950 | 0.864 | 0.905 | 0.956 |
| Random Forest | 0.906 | 0.900 | 0.818 | 0.857 | 0.972 |

**Best model: Linear SVM** (by F1-score) — confusion matrix on the held-out
test set: 40/42 legitimate postings correctly identified, 21/22 fraudulent
postings correctly caught, 2 false positives, 1 false negative.

See `outputs/model_comparison.png` for the chart and `outputs/model_comparison.csv`
for the raw numbers. Numbers will shift slightly if you regenerate the dataset
with a different random seed.

## Web demo (Streamlit)

```bash
streamlit run app.py
```

Opens at http://localhost:8501 with four pages:

- **Check a Posting** - paste a posting, get a verdict, a risk score, the signals
  behind it, and a plain-language red-flag scan. One-click fake/legit examples.
- **Batch Scanner** - upload a CSV and get every posting ranked by risk, with a
  downloadable results file (a CSV template is provided).
- **Model Dashboard** - dataset overview, 4-model comparison, confusion matrix,
  and the words/signals the model learned.
- **About** - method, honest limitations, and how to spot scams yourself.

The app builds the dataset and trains the models by itself on first launch
(about 20 seconds) if they are missing, so no extra setup is needed.

## How to run it (command line)

```bash
pip install -r requirements.txt

python3 generate_dataset.py   # builds data/job_postings.csv
python3 classify.py           # trains + compares models, saves the best one
python3 predict.py --demo     # scores a sample suspicious posting
python3 predict.py --legit-demo  # scores a sample legitimate posting
python3 predict.py            # interactive: enter your own posting's details
python3 test_model.py         # sanity-checks the model on 5 hand-written cases
```

If you have `pytest` installed, `pytest test_model.py -v` runs the same
checks as real test cases with pass/fail output per case.

### If `pip install` wants to build a package from source

If pip warns about compiling matplotlib (or anything else) from a `.tar.gz`
instead of installing a pre-built wheel, that almost always means your Python
version is very new and a pre-built wheel doesn't exist for it yet. Two
options, in order of preference:

1. Use a virtual environment on an older, well-supported Python (3.11 or
   3.12): `python3.11 -m venv venv && source venv/bin/activate` (Linux/Mac)
   or `venv\Scripts\activate` (Windows), then `pip install -r requirements.txt`.
2. Force pip to only use pre-built wheels and fail fast instead of compiling:
   `pip install --only-binary :all: -r requirements.txt`.

Either way, no version in `requirements.txt` is pinned exactly — any
reasonably recent version satisfies it.

## Project structure

```
fakejob/
├── generate_dataset.py   # builds the synthetic dataset (documented, reproducible)
├── classify.py            # full pipeline: EDA -> split -> train -> compare -> save
├── app.py                  # Streamlit web demo (4 pages)
├── .streamlit/config.toml  # app theme
├── predict.py              # CLI to score a new posting + explain why
├── test_model.py           # sanity-check test suite (standalone or pytest)
├── data/job_postings.csv   # generated dataset (320 rows)
├── model/best_model.joblib # saved best pipeline (created after running classify.py)
├── outputs/                # comparison chart + metrics CSV
└── requirements.txt
```

## Limitations & next steps

- Synthetic data, not the real EMSCAD set — flagged above, not hidden.
- 320 rows is small for text classification; real EMSCAD's 17,880 rows would
  let TF-IDF and Random Forest generalize much better.
- No hyperparameter tuning (e.g. `GridSearchCV`) yet — a natural Project 3
  extension.
- The `test_model.py` sanity suite is intentionally small (5 cases) and
  deliberately includes one ambiguous case that fails — that's documented
  behavior, not a bug: it shows the model's real limits instead of hiding them.
- Could wrap `predict.py` in a small Flask/FastAPI endpoint or Streamlit UI
  for a live demo link.
