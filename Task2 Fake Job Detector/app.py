"""
app.py  -  Fraudulent Job Posting Detector (Streamlit web demo)
DecodeLabs AI Engineering Internship - Project 2: Data Classification Using AI

Run:  streamlit run app.py

Pages
  1. Check a Posting   - paste one posting, get verdict + risk score + "why"
  2. Batch Scanner     - upload a CSV, get postings ranked by risk
  3. Model Dashboard   - dataset, metrics, what the model learned
  4. About             - how it works, honest limitations, safety tips

The app builds its own dataset + model on first launch if they are missing
(or if the saved model was made with a different scikit-learn version).
"""

import io
import os
import re
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "data" / "job_postings.csv"
MODEL_PATH = BASE / "model" / "best_model.joblib"
METRICS_PATH = BASE / "outputs" / "model_comparison.csv"
CHART_PATH = BASE / "outputs" / "model_comparison.png"

st.set_page_config(page_title="Fake Job Posting Detector", page_icon="🛡️", layout="wide")

FLAG_COLS = ["has_company_profile", "has_company_logo", "requests_upfront_payment",
             "urgent_hiring_language", "requires_no_experience"]
MODEL_COLS = (["description", "years_experience_required"] + FLAG_COLS +
              ["employment_type", "contact_email_domain", "salary_listed"])
EMPLOYMENT_TYPES = ["Full-time", "Part-time", "Contract", "Temporary"]
FREE_EMAILS = {"gmail.com", "yahoo.com", "hotmail.com", "outlook.com"}

FLAG_PHRASES = {
    "has_company_profile": ("has a company profile/about section", "no company profile/about section"),
    "has_company_logo": ("shows a company logo", "no company logo"),
    "requests_upfront_payment": ("asks for upfront payment/bank details", "does not ask for upfront payment"),
    "urgent_hiring_language": ("uses urgent hiring language", "no urgent hiring language"),
    "requires_no_experience": ("advertised as 'no experience needed'", "does not advertise 'no experience needed'"),
    "salary_listed": ("salary/pay range listed", "no salary/pay range listed"),
}

# Plain-language rule scan of the description text (separate from the ML model)
RED_FLAG_PATTERNS = {
    "Asks for money or fees": r"processing fee|registration fee|\bfee of\b|\bdeposit\b|pay (us|upfront)",
    "Asks for bank / financial details": r"bank (account )?details|credit card|social security|routing number",
    "Cheque / equipment-money trick": r"check in advance|send you a check|purchase your own equipment",
    "Urgency / pressure": r"act now|limited (time|spots)|\burgent(ly)?\b|immediate start|hurry|today only",
    "Unrealistic earnings": r"\$\s?\d{3,5}\s?/\s?(week|day)|earn up to|guaranteed",
    "Skips normal hiring steps": r"no experience|no interview|no resume|hire on the spot|you'?re hired",
    "Personal email / chat app contact": r"personal email|whatsapp|telegram",
    "Package reshipping": r"reship|repackage|forward (the )?packages",
}

FAKE_EXAMPLE = {
    "desc": "URGENT HIRING! Earn $2500/week working from home, no experience needed. Send us your "
            "bank account details today so we can process your first paycheck. Limited spots, act now!",
    "years": 0, "profile": False, "logo": False, "upfront": True, "urgent": True, "noexp": True,
    "emp": "Part-time", "email": "gmail.com", "salary": True,
}
LEGIT_EXAMPLE = {
    "desc": "We are hiring a Junior Web Developer to join our engineering team. Candidates should have "
            "at least 1 year of experience with JavaScript and React. Please apply through our official "
            "careers page with your resume and GitHub portfolio. Interviews are two rounds: a technical "
            "screen and a team interview.",
    "years": 1, "profile": True, "logo": True, "upfront": False, "urgent": False, "noexp": False,
    "emp": "Full-time", "email": "corp.com", "salary": True,
}


# --------------------------------------------------------------------------
# Model / data loading (self-healing)
# --------------------------------------------------------------------------
def _run_script(name):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    subprocess.run([sys.executable, name], cwd=str(BASE), check=True,
                   capture_output=True, env=env)


@st.cache_resource(show_spinner="First launch: building dataset and training models (~20s)...")
def get_model():
    try:
        if MODEL_PATH.exists():
            bundle = joblib.load(MODEL_PATH)
            return bundle["pipeline"], bundle["model_name"]
    except Exception:
        pass  # e.g. saved with a different scikit-learn version -> retrain below
    if not DATA_PATH.exists():
        _run_script("generate_dataset.py")
    _run_script("classify.py")
    bundle = joblib.load(MODEL_PATH)
    return bundle["pipeline"], bundle["model_name"]


@st.cache_data
def get_dataset():
    return pd.read_csv(DATA_PATH) if DATA_PATH.exists() else None


# --------------------------------------------------------------------------
# Prediction + explanation helpers
# --------------------------------------------------------------------------
def score_df(pipe, df):
    """Return (predicted labels, risk scores 0-1). LinearSVC has no probabilities,
    so its decision score is squashed with a sigmoid -> a risk score, not a true probability."""
    if hasattr(pipe, "predict_proba"):
        risk = pipe.predict_proba(df)[:, 1]
    else:
        risk = 1 / (1 + np.exp(-pipe.decision_function(df)))
    return pipe.predict(df), risk


def linear_parts(pipe):
    model = pipe.named_steps["model"]
    if hasattr(model, "calibrated_classifiers_"):
        model = model.calibrated_classifiers_[0].estimator
    if not hasattr(model, "coef_"):
        return None
    pre = pipe.named_steps["preprocess"]
    names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
    return model.coef_[0], names, pre


def describe_feature(feat, row):
    if feat.startswith("contact_email_domain_"):
        return f"contact email domain is {feat.split('contact_email_domain_', 1)[1]}"
    if feat.startswith("employment_type_"):
        return f"employment type is {feat.split('employment_type_', 1)[1]}"
    if feat == "years_experience_required":
        return f"{int(row['years_experience_required'])} years of experience required"
    if feat in FLAG_PHRASES:
        yes, no = FLAG_PHRASES[feat]
        return yes if row[feat] == 1 else no
    return f"description mentions '{feat}'"


def explain_row(pipe, row_df, top_n=8):
    parts = linear_parts(pipe)
    if parts is None:
        return None
    coefs, names, pre = parts
    x = pre.transform(row_df)
    x = x.toarray()[0] if hasattr(x, "toarray") else np.asarray(x)[0]
    contrib = x * coefs
    order = np.argsort(-np.abs(contrib))
    row = row_df.iloc[0]
    out = [(describe_feature(names[i], row), float(contrib[i])) for i in order if abs(contrib[i]) > 1e-6]
    return out[:top_n]


def scan_red_flags(text):
    hits = []
    for label, pattern in RED_FLAG_PATTERNS.items():
        found = re.findall(pattern, text, flags=re.IGNORECASE)
        if found:
            m = re.search(pattern, text, flags=re.IGNORECASE)
            hits.append((label, m.group(0)))
    return hits


def clean_domain(raw):
    d = (raw or "").strip().lower()
    return d.split("@")[-1] if "@" in d else d


def risk_band(risk):
    if risk >= 0.65:
        return "high"
    if risk <= 0.35:
        return "low"
    return "borderline"


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
def load_example(ex):
    st.session_state.update(ex)


def page_check(pipe, model_name):
    st.title("🛡️ Check a Job Posting")
    st.caption("Paste a posting, answer a few quick questions, and the model estimates how likely it is to be a scam.")

    defaults = dict(desc="", years=0, profile=False, logo=False, upfront=False, urgent=False,
                    noexp=False, emp="Full-time", email="gmail.com", salary=True)
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

    b1, b2, _ = st.columns([1, 1, 3])
    b1.button("🚩 Load fake example", on_click=load_example, args=(FAKE_EXAMPLE,))
    b2.button("✅ Load legit example", on_click=load_example, args=(LEGIT_EXAMPLE,))

    left, right = st.columns([3, 2])
    with left:
        st.text_area("Job description", key="desc", height=190,
                     placeholder="Paste the full text of the job posting here...")
        c1, c2 = st.columns(2)
        c1.selectbox("Employment type", EMPLOYMENT_TYPES, key="emp")
        c2.number_input("Years of experience required", 0, 30, key="years")
        st.text_input("Contact email (or just the domain)", key="email",
                      help="Free providers like gmail.com / yahoo.com are a common scam signal.")
    with right:
        st.markdown("**Quick checks about the listing**")
        st.toggle("Has a company profile / about section", key="profile")
        st.toggle("Shows a company logo", key="logo")
        st.toggle("Lists a salary / pay range", key="salary")
        st.toggle("Asks for money or bank details", key="upfront")
        st.toggle("Uses urgent language (\"act now\")", key="urgent")
        st.toggle("Advertised as \"no experience needed\"", key="noexp")

    if not st.button(" Analyze posting", type="primary"):
        return
    ss = st.session_state
    if not ss.desc.strip():
        st.warning("Please paste a job description first (or load an example).")
        return

    row = pd.DataFrame([{
        "description": ss.desc,
        "years_experience_required": ss.years,
        "has_company_profile": int(ss.profile), "has_company_logo": int(ss.logo),
        "requests_upfront_payment": int(ss.upfront), "urgent_hiring_language": int(ss.urgent),
        "requires_no_experience": int(ss.noexp), "employment_type": ss.emp,
        "contact_email_domain": clean_domain(ss.email), "salary_listed": int(ss.salary),
    }])[MODEL_COLS]
    pred, risk = score_df(pipe, row)
    risk, pred = float(risk[0]), int(pred[0])
    band = risk_band(risk)

    st.divider()
    r1, r2 = st.columns([2, 3])
    with r1:
        if pred == 1:
            st.error("### 🚩 Likely FRAUDULENT")
        elif band == "borderline":
            st.warning("### ⚠️ Looks legitimate, but borderline")
        else:
            st.success("### ✅ Likely LEGITIMATE")
        st.metric("Risk score", f"{risk:.0%}")
        st.progress(min(max(risk, 0.0), 1.0))
        if band == "borderline":
            st.caption("The model is unsure. Verify the company independently before sharing any details.")
        st.caption(f"Model: {model_name}. The risk score is a model estimate, not a guarantee.")

    with r2:
        st.markdown("**Why the model decided this**")
        expl = explain_row(pipe, row)
        if expl is None:
            st.info("This model type has no readable weights, so no per-signal explanation is available.")
        else:
            for text, c in expl[:6]:
                st.markdown(f"{'🔴' if c > 0 else '🟢'} {text}  \n"
                            f"<span style='color:#6B7280;font-size:0.8em'>"
                            f"{'pushes toward fraud' if c > 0 else 'pushes toward legitimate'}</span>",
                            unsafe_allow_html=True)

    hits = scan_red_flags(ss.desc)
    if clean_domain(ss.email) in FREE_EMAILS:
        hits.append(("Free-email contact", clean_domain(ss.email)))
    with st.expander(f"Plain-language red-flag scan ({len(hits)} found)", expanded=bool(hits)):
        if hits:
            for label, snippet in hits:
                st.markdown(f"⚠️ **{label}** - matched: `{snippet}`")
        else:
            st.write("No classic scam phrases found in the text.")


TEMPLATE_CSV = (
    "description,years_experience_required,has_company_profile,has_company_logo,"
    "requests_upfront_payment,urgent_hiring_language,requires_no_experience,"
    "employment_type,contact_email_domain,salary_range\n"
    "\"Earn $2000/week from home, no experience. Send bank details now!\",0,0,0,1,1,1,Part-time,gmail.com,$3000-$5000/month\n"
    "\"Hiring a QA Tester. 2 years experience. Apply via our careers portal.\",2,1,1,0,0,0,Full-time,corp.com,\n"
)


def prepare_batch(df):
    df = df.copy()
    if "description" not in df.columns:
        raise ValueError("The CSV needs at least a 'description' column.")
    df["description"] = df["description"].fillna("").astype(str)
    for c in ["years_experience_required"] + FLAG_COLS:
        df[c] = pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0).astype(int)
    df["employment_type"] = df.get("employment_type", "Full-time")
    df["employment_type"] = df["employment_type"].fillna("Full-time").astype(str)
    df["contact_email_domain"] = df.get("contact_email_domain", "unknown")
    df["contact_email_domain"] = df["contact_email_domain"].fillna("unknown").astype(str).map(clean_domain)
    if "salary_listed" not in df.columns:
        df["salary_listed"] = df["salary_range"].notna().astype(int) if "salary_range" in df.columns else 0
    return df


def page_batch(pipe):
    st.title("📋 Batch Scanner")
    st.caption("Upload a CSV of postings and get them ranked from most to least suspicious.")
    st.download_button("⬇️ Download CSV template", TEMPLATE_CSV, "postings_template.csv", "text/csv")
    st.caption("Only `description` is required; any other column you leave out is filled with a neutral default.")

    up = st.file_uploader("Upload a CSV", type="csv")
    use_sample = st.button("Or try 30 random postings from the built-in dataset")
    raw = None
    if up is not None:
        raw = pd.read_csv(up)
    elif use_sample or st.session_state.get("batch_sample") is not None:
        if use_sample:
            data = get_dataset()
            st.session_state["batch_sample"] = data.sample(30, random_state=None).drop(columns=[]).reset_index(drop=True)
        raw = st.session_state["batch_sample"]
    if raw is None:
        return
    try:
        df = prepare_batch(raw)
    except ValueError as e:
        st.error(str(e))
        return

    pred, risk = score_df(pipe, df[MODEL_COLS])
    out = raw.copy()
    out.insert(0, "verdict", np.where(pred == 1, "FRAUDULENT", "legitimate"))
    out.insert(0, "risk_score", np.round(risk, 3))
    out = out.sort_values("risk_score", ascending=False).reset_index(drop=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Postings scanned", len(out))
    m2.metric("Flagged as fraudulent", int((pred == 1).sum()))
    m3.metric("Average risk", f"{risk.mean():.0%}")
    if "fraudulent" in raw.columns:
        acc = float((pred == raw["fraudulent"].astype(int).values).mean())
        st.info(f"This file includes true labels: the model got **{acc:.0%}** of these right.")

    st.dataframe(out, hide_index=True,
                 column_config={"risk_score": st.column_config.ProgressColumn(
                     "risk_score", min_value=0.0, max_value=1.0, format="%.2f")})
    st.download_button("⬇️ Download ranked results", out.to_csv(index=False), "ranked_results.csv", "text/csv")


def page_dashboard(pipe, model_name):
    st.title("📊 Model Dashboard")
    data = get_dataset()
    t1, t2, t3 = st.tabs(["Dataset", "Model comparison", "What the model learned"])

    with t1:
        if data is None:
            st.info("Dataset not found.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Postings", len(data))
            c2.metric("Fraudulent", int(data["fraudulent"].sum()))
            c3.metric("Legitimate", int((1 - data["fraudulent"]).sum()))
            st.bar_chart(data["fraudulent"].map({0: "legitimate", 1: "fraudulent"}).value_counts())
            st.dataframe(data.head(50), hide_index=True)

    with t2:
        if METRICS_PATH.exists():
            m = pd.read_csv(METRICS_PATH)
            st.caption("Evaluated on a held-out 20% test split. Best model is chosen by F1-score.")
            show = m[["model", "accuracy", "precision", "recall", "f1", "roc_auc"]]
            st.dataframe(show, hide_index=True)
            st.bar_chart(show.set_index("model"))
            st.success(f"Model in use: **{model_name}**")
            if CHART_PATH.exists():
                st.image(str(CHART_PATH), caption="Metric comparison and confusion matrix of the best model")
        else:
            st.info("Run classify.py to generate metrics.")

    with t3:
        parts = linear_parts(pipe)
        if parts is None:
            st.info("The selected model has no readable weights.")
        else:
            coefs, names, _ = parts
            text_idx = [i for i, n in enumerate(names)
                        if not (n.startswith("contact_email_domain_") or n.startswith("employment_type_")
                                or n in FLAG_PHRASES or n == "years_experience_required")]
            s = pd.DataFrame({"signal": [names[i] for i in text_idx], "weight": [coefs[i] for i in text_idx]})
            a, b = st.columns(2)
            a.markdown("**Words/phrases that signal FRAUD**")
            a.dataframe(s.sort_values("weight", ascending=False).head(12), hide_index=True)
            b.markdown("**Words/phrases that signal LEGITIMATE**")
            b.dataframe(s.sort_values("weight").head(12), hide_index=True)
            flags = pd.DataFrame({"signal": [n for n in names if n in FLAG_PHRASES or n.startswith("contact_email")],
                                  "weight": [coefs[i] for i, n in enumerate(names)
                                             if n in FLAG_PHRASES or n.startswith("contact_email")]})
            st.markdown("**Structured signals (flags and email domain)**")
            st.bar_chart(flags.set_index("signal"), horizontal=True)
            st.caption("Positive weight = pushes toward fraud. Note: flags are scaled, so a 0 value can still carry weight.")


def page_about():
    st.title("ℹ️ About this project")
    st.markdown("""
**Fraudulent Job Posting Detector** - DecodeLabs AI Engineering Internship, Project 2 (Data Classification Using AI).

**Pipeline:** load data -> stratified 80/20 train/test split -> TF-IDF on the description + scaled flags +
one-hot categoricals -> train 4 classifiers (Logistic Regression, Random Forest, Linear SVM, Calibrated SVM)
-> compare on accuracy, precision, recall, F1, ROC-AUC with 5-fold cross-validation -> keep the best.

**Honest limitations**
- The dataset is **synthetic** (320 generated postings with realistic red-flag patterns and added noise),
  so scores here do not predict real-world performance. Real fraud is rarer and subtler.
- It is a learning project, not a security product. Never rely on it alone.
- Some borderline legitimate postings (small companies using free email, no logo) can be flagged.
""")
    with st.expander("How to spot a fake job posting yourself"):
        st.markdown("""
- A real employer never asks you to **pay a fee** or send **bank details** before you are hired.
- Be wary of **"earn $X/week, no experience"** and pressure like **"act now"**.
- Check the company has a real website, LinkedIn page and an **official-domain email**, not gmail/yahoo.
- Be suspicious of **no interview** or hiring over WhatsApp/Telegram.
- Never **cash a cheque** and send money back, and never agree to **reship packages**.
""")


# --------------------------------------------------------------------------
def main():
    pipe, model_name = get_model()
    with st.sidebar:
        st.markdown("## 🛡️ Job Scam Detector")
        page = st.radio("Navigate", ["Check a Posting", "Batch Scanner", "Model Dashboard", "About"],
                        label_visibility="collapsed", key="page")
        st.divider()
        st.caption(f"Active model: **{model_name}**")
        st.caption("DecodeLabs AI Internship - Project 2")
    if page == "Check a Posting":
        page_check(pipe, model_name)
    elif page == "Batch Scanner":
        page_batch(pipe)
    elif page == "Model Dashboard":
        page_dashboard(pipe, model_name)
    else:
        page_about()


main()
