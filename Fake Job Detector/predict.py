"""
predict.py
----------
Interactive CLI: paste in a new job posting's details and get a
fraud-risk prediction from the trained model (model/best_model.joblib),
along with a plain-English explanation of WHY it decided that way.

Run:
  python3 predict.py                # interactive, type in your own posting
  python3 predict.py --demo         # scores a sample scammy posting
  python3 predict.py --legit-demo   # scores a sample legitimate posting
"""

import sys
import os
import joblib
import pandas as pd

MODEL_PATH = os.path.join("model", "best_model.joblib")

FRAUD_SAMPLE = {
    "description": (
        "EARN UP TO $2000/WEEK WORKING FROM HOME - NO EXPERIENCE NEEDED!!! "
        "Simply forward your bank details so we can process your first payment. "
        "Limited time opportunity, act now!"
    ),
    "years_experience_required": 0,
    "has_company_profile": 0,
    "has_company_logo": 0,
    "requests_upfront_payment": 1,
    "urgent_hiring_language": 1,
    "requires_no_experience": 1,
    "employment_type": "Part-time",
    "contact_email_domain": "gmail.com",
    "salary_listed": 1,
}

LEGIT_SAMPLE = {
    "description": (
        "We are hiring a Junior Web Developer to join our engineering team. "
        "Candidates should have at least 1 year of experience with JavaScript and React. "
        "Please apply through our official careers page with your resume and GitHub portfolio. "
        "Interviews are two rounds: a technical screen and a team interview."
    ),
    "years_experience_required": 1,
    "has_company_profile": 1,
    "has_company_logo": 1,
    "requests_upfront_payment": 0,
    "urgent_hiring_language": 0,
    "requires_no_experience": 0,
    "employment_type": "Full-time",
    "contact_email_domain": "corp.com",
    "salary_listed": 1,
}

NUMERIC_COLS = ["years_experience_required", "has_company_profile",
                 "has_company_logo", "requests_upfront_payment",
                 "urgent_hiring_language", "requires_no_experience", "salary_listed"]

# Human-friendly (yes_phrase, no_phrase) pairs, used in the explanation
FLAG_PHRASES = {
    "has_company_profile": ("has a company profile/about section", "no company profile/about section"),
    "has_company_logo": ("shows a company logo", "no company logo shown"),
    "requests_upfront_payment": ("asks for upfront payment/bank details", "does not ask for upfront payment"),
    "urgent_hiring_language": ("uses urgent hiring language", "no urgent hiring language"),
    "requires_no_experience": ("advertised as 'no experience needed'", "does not advertise 'no experience needed'"),
    "salary_listed": ("salary/pay range listed", "no salary/pay range listed"),
}


def load_model():
    if not os.path.exists(MODEL_PATH):
        print("No trained model found. Run these first:")
        print("  python3 generate_dataset.py && python3 classify.py")
        sys.exit(1)
    bundle = joblib.load(MODEL_PATH)
    return bundle["pipeline"], bundle["model_name"]


def _describe_flag(feat, posting):
    """Turn a raw feature + its value in this posting into a readable phrase,
    e.g. has_company_profile=0 -> 'no company profile/about section'."""
    if feat not in posting:
        return feat
    val = posting[feat]
    if feat == "years_experience_required":
        return f"{val} years of experience required"
    if feat in FLAG_PHRASES:
        yes_phrase, no_phrase = FLAG_PHRASES[feat]
        return yes_phrase if val == 1 else no_phrase
    return f"{feat} = {val}"


def explain(pipe, posting: dict, top_n=5):
    """Best-effort, model-agnostic explanation: only works cleanly for the
    linear models (Logistic Regression / Linear SVM) since their weights are
    directly readable. For Random Forest, this is skipped with a note --
    tree-based feature importance isn't per-prediction the same way."""
    try:
        model = pipe.named_steps["model"]
        # Unwrap CalibratedClassifierCV to get at the underlying linear model
        if hasattr(model, "calibrated_classifiers_"):
            model = model.calibrated_classifiers_[0].estimator
        if not hasattr(model, "coef_"):
            print("\n(Explanation skipped: this model type doesn't expose linear weights.)")
            return

        preprocess = pipe.named_steps["preprocess"]
        tfidf = preprocess.named_transformers_["text"]
        cat_encoder = preprocess.named_transformers_["cat"]

        text_features = list(tfidf.get_feature_names_out())
        num_features = NUMERIC_COLS
        cat_features = list(cat_encoder.get_feature_names_out(["employment_type", "contact_email_domain"]))
        all_features = text_features + num_features + cat_features
        coefs = model.coef_[0]

        row = pd.DataFrame([posting])
        transformed = preprocess.transform(row)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        contributions = transformed[0] * coefs[:transformed.shape[1]]

        ranked = sorted(zip(all_features, contributions), key=lambda x: -abs(x[1]))
        shown = [item for item in ranked if abs(item[1]) > 1e-6][:top_n]

        if not shown:
            print("\n(No strong individual signals -- a fairly borderline/neutral posting.)")
            return

        print(f"\nTop signals behind this prediction:")
        for feat, contribution in shown:
            direction = "-> FRAUDULENT" if contribution > 0 else "-> legitimate"
            if feat in NUMERIC_COLS:
                display = _describe_flag(feat, posting)
            elif feat.startswith("contact_email_domain_"):
                display = f"contact email domain is {feat.split('contact_email_domain_')[-1]}"
            elif feat.startswith("employment_type_"):
                display = f"employment type is {feat.split('employment_type_')[-1]}"
            else:
                display = f"description mentions '{feat}'"
            print(f"  {display:50s} {direction}")
    except Exception as e:
        print(f"\n(Explanation skipped: {e})")


def predict_one(pipe, model_name, posting: dict, show_explanation=True):
    row = pd.DataFrame([posting])
    pred = pipe.predict(row)[0]
    label = "FRAUDULENT" if pred == 1 else "legitimate"

    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(row)[0][1]
        conf_str = f"{proba:.1%} estimated fraud probability"
    else:
        score = pipe.decision_function(row)[0]
        conf_str = f"decision score = {score:.2f} (>0 leans fraudulent)"

    print(f"\nModel used : {model_name}")
    print(f"Prediction : {label}")
    print(f"Confidence : {conf_str}")

    if show_explanation:
        explain(pipe, posting)

    return pred


def ask(prompt, cast=str, default=None):
    raw = input(f"{prompt}" + (f" [{default}]: " if default is not None else ": "))
    if raw.strip() == "" and default is not None:
        return default
    try:
        return cast(raw)
    except ValueError:
        return default


def interactive():
    print("=" * 60)
    print("FRAUDULENT JOB POSTING DETECTOR - live prediction")
    print("=" * 60)
    print("Enter details for a job posting (press Enter to accept the default).\n")

    posting = {
        "description": ask("Full job description text"),
        "years_experience_required": ask("Years of experience required", int, 0),
        "has_company_profile": ask("Does the listing include a company profile/about section? (1=yes, 0=no)", int, 0),
        "has_company_logo": ask("Does the listing show a company logo? (1=yes, 0=no)", int, 0),
        "requests_upfront_payment": ask("Does it ask you to pay/send money/bank details upfront? (1=yes, 0=no)", int, 0),
        "urgent_hiring_language": ask("Uses urgent language ('act now', 'immediate start')? (1=yes, 0=no)", int, 0),
        "requires_no_experience": ask("Advertised as 'no experience needed'? (1=yes, 0=no)", int, 0),
        "employment_type": ask("Employment type (Full-time/Part-time/Contract/Temporary)", str, "Full-time"),
        "contact_email_domain": ask("Contact email domain (e.g. gmail.com or company.com)", str, "gmail.com"),
        "salary_listed": ask("Is a salary/pay range listed? (1=yes, 0=no)", int, 1),
    }
    pipe, model_name = load_model()
    predict_one(pipe, model_name, posting)


def demo(posting, title):
    pipe, model_name = load_model()
    print("=" * 60)
    print(title)
    print("=" * 60)
    print(f"Description: {posting['description']}")
    predict_one(pipe, model_name, posting)


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo(FRAUD_SAMPLE, "DEMO: scoring a sample suspicious posting")
    elif "--legit-demo" in sys.argv:
        demo(LEGIT_SAMPLE, "DEMO: scoring a sample legitimate posting")
    else:
        try:
            interactive()
        except (EOFError, KeyboardInterrupt):
            print("\nNo interactive input available - running demo instead.\n")
            demo(FRAUD_SAMPLE, "DEMO: scoring a sample suspicious posting")
