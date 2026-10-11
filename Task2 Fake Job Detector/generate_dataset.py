"""
generate_dataset.py
--------------------
Builds a synthetic-but-realistic "job postings" dataset for the
Fraudulent Job Posting Detector project (DecodeLabs AI Internship - Project 2).

Why synthetic instead of a downloaded Kaggle CSV?
  - Fully reproducible (fixed random seed), no external downloads needed.
  - Lets us control class balance and inject realistic "red-flag" signal
    patterns (the same signals real fraud-detection research uses on the
    EMSCAD / Kaggle "Real or Fake Job Posting" dataset), while still adding
    noise so the problem isn't trivially easy for the model.
  - Produces both structured features (numeric/categorical) AND free text,
    so the pipeline demonstrates handling a realistic mixed feature set.

Run:  python3 generate_dataset.py
Output: data/job_postings.csv
"""

import random
import csv
import os

random.seed(42)

N_SAMPLES = 320
FRAUD_RATIO = 0.35  # a bit higher than real-world (~5%) so the model has enough
                     # fraud examples to learn from -- noted honestly in the README

# ---------------------------------------------------------------------------
# Vocabulary pools
# ---------------------------------------------------------------------------

JOB_TITLES = [
    "Data Entry Clerk", "Customer Support Representative", "Software Engineer",
    "Marketing Coordinator", "Graphic Designer", "Virtual Assistant",
    "Financial Analyst", "HR Coordinator", "Content Writer", "Sales Executive",
    "Administrative Assistant", "Social Media Manager", "IT Support Specialist",
    "Accounts Payable Clerk", "Recruitment Consultant", "Project Coordinator",
    "Warehouse Associate", "Mystery Shopper", "Package Reshipping Agent",
    "Online Survey Panelist", "Junior Web Developer", "QA Tester",
    "Bookkeeper", "Research Assistant", "Product Manager",
]

LEGIT_COMPANY_NAMES = [
    "Meridian Analytics", "Northbridge Consulting", "Clearwater Software",
    "Atlas Retail Group", "Summit Health Partners", "Lahore Tech Solutions",
    "BluePeak Systems", "Riverstone Media", "Falcon Logistics",
    "Horizon Financial Services", "Cobalt Design Studio", "Ferris & Wren LLP",
]

VAGUE_COMPANY_NAMES = [
    "Global Career Solutions", "Prime Earnings Group", "WorldWide Remote Jobs",
    "Elite Opportunities LLC", "FastTrack Careers", "Premier Home Jobs",
    "International Talent Network", "Bright Future Employment",
    "Apex Global Ventures", "Rapid Hire Associates",
]

LOCATIONS = [
    "Remote", "Lahore, Pakistan", "Karachi, Pakistan", "Islamabad, Pakistan",
    "London, UK", "New York, NY", "Remote (Worldwide)", "Toronto, Canada",
    "Berlin, Germany", "Dubai, UAE",
]

LEGIT_DESC_SNIPPETS = [
    "We are looking for a {title} to join our growing team in {location}.",
    "You will collaborate with cross-functional teams to deliver measurable results.",
    "The successful candidate will have at least {yrs} years of relevant experience.",
    "We offer a structured onboarding program and clear performance review cycles.",
    "Our benefits package includes health insurance, paid leave, and a retirement plan.",
    "Interviews are conducted over two rounds: a technical screen and a team interview.",
    "Please apply through our official careers portal with your resume and portfolio.",
    "This role reports directly to the {title} team lead and works standard business hours.",
    "We value diversity and are an equal-opportunity employer.",
    "Compensation is based on experience and will be discussed during the interview stage.",
]

FRAUD_DESC_SNIPPETS = [
    "EARN UP TO ${amt}/WEEK WORKING FROM HOME - NO EXPERIENCE NEEDED!!!",
    "Immediate start! We hire on the spot, no interview required.",
    "You will receive a check in advance to purchase your own equipment.",
    "This is a limited time opportunity, act now before positions fill up!",
    "Simply forward your bank details so we can process your first payment.",
    "Work only 2 hours a day and earn a full-time salary guaranteed.",
    "A small processing/registration fee of ${fee} is required before onboarding.",
    "Contact our hiring manager directly at the personal email below to skip the queue.",
    "No resume needed - just fill out the quick form and you're hired!",
    "We will reship packages to you; simply repackage and forward them, all shipping paid.",
]

NEUTRAL_DESC_SNIPPETS = [
    "Duties include general office support and data organization tasks.",
    "Strong communication skills and attention to detail are required.",
    "Familiarity with common office software is a plus.",
    "This position may require occasional weekend availability.",
]

EMAIL_DOMAINS_LEGIT = ["company.com", "careers-company.com", "corp.com", "hr-company.com"]
EMAIL_DOMAINS_FREE = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com"]

EMPLOYMENT_TYPES = ["Full-time", "Part-time", "Contract", "Temporary"]


def rand_choice_weighted(pool, weights):
    return random.choices(pool, weights=weights, k=1)[0]


def build_row(idx, is_fraud):
    title = random.choice(JOB_TITLES)
    location = random.choice(LOCATIONS)
    yrs = random.choice([0, 1, 2, 3, 5])
    employment_type = random.choice(EMPLOYMENT_TYPES)

    # --- Structured "red flag" style features -----------------------------
    # Each has some probability of appearing for fraud vs. legit postings,
    # with intentional overlap (noise) so classes are not perfectly separable.
    if is_fraud:
        company_name = random.choice(VAGUE_COMPANY_NAMES)
        has_company_profile = random.random() < 0.15
        has_company_logo = random.random() < 0.20
        salary_suspiciously_high = random.random() < 0.55
        requests_upfront_payment = random.random() < 0.45
        urgent_hiring_language = random.random() < 0.70
        generic_email_domain = random.random() < 0.75
        vague_location = random.random() < 0.5
        requires_no_experience = random.random() < 0.65
        n_legit_snips = random.randint(0, 2)
        n_fraud_snips = random.randint(1, 3)
    else:
        company_name = random.choice(LEGIT_COMPANY_NAMES)
        has_company_profile = random.random() < 0.85
        has_company_logo = random.random() < 0.80
        salary_suspiciously_high = random.random() < 0.05
        requests_upfront_payment = random.random() < 0.02
        urgent_hiring_language = random.random() < 0.10
        generic_email_domain = random.random() < 0.15
        vague_location = random.random() < 0.1
        requires_no_experience = random.random() < 0.15
        n_legit_snips = random.randint(1, 3)
        n_fraud_snips = random.randint(0, 2)

    # --- Build description text --------------------------------------------
    snippets = []
    for _ in range(n_legit_snips):
        s = random.choice(LEGIT_DESC_SNIPPETS)
        snippets.append(s.format(title=title, location=location, yrs=yrs))
    for _ in range(n_fraud_snips):
        s = random.choice(FRAUD_DESC_SNIPPETS)
        snippets.append(s.format(amt=random.choice([800, 1200, 1500, 2500]),
                                  fee=random.choice([25, 40, 60, 99])))
    if random.random() < 0.4:
        snippets.append(random.choice(NEUTRAL_DESC_SNIPPETS))
    random.shuffle(snippets)
    description = " ".join(snippets)

    email_domain = random.choice(EMAIL_DOMAINS_FREE) if generic_email_domain else random.choice(EMAIL_DOMAINS_LEGIT)

    salary_range = ""
    if salary_suspiciously_high:
        low = random.choice([3000, 4000, 5000])
        high = low + random.choice([3000, 5000, 8000])
        salary_range = f"${low}-${high}/month"
    elif random.random() < 0.5:
        low = random.choice([300, 500, 700, 900])
        high = low + random.choice([200, 400, 600])
        salary_range = f"${low}-${high}/month"

    row = {
        "job_id": idx,
        "title": title,
        "company_name": company_name,
        "location": "Not specified" if vague_location else location,
        "employment_type": employment_type,
        "years_experience_required": 0 if requires_no_experience else yrs,
        "salary_range": salary_range,
        "has_company_profile": int(has_company_profile),
        "has_company_logo": int(has_company_logo),
        "contact_email_domain": email_domain,
        "requests_upfront_payment": int(requests_upfront_payment),
        "urgent_hiring_language": int(urgent_hiring_language),
        "requires_no_experience": int(requires_no_experience),
        "description": description,
        "fraudulent": int(is_fraud),
    }

    # --- Label noise -------------------------------------------------------
    # Real-world fraud data is messy: some legit postings look sketchy and
    # some scams are well-disguised. Flip a small fraction of the structured
    # "red flag" fields independently of the label so the classes overlap
    # and the problem isn't trivially separable (a model scoring a suspicious
    # 100% usually just means the data was too clean).
    flip_cols = ["has_company_profile", "has_company_logo", "requests_upfront_payment",
                 "urgent_hiring_language", "requires_no_experience"]
    for col in flip_cols:
        if random.random() < 0.18:
            row[col] = 1 - row[col]

    return row


def main():
    n_fraud = int(N_SAMPLES * FRAUD_RATIO)
    n_legit = N_SAMPLES - n_fraud
    rows = [build_row(i, True) for i in range(n_fraud)] + \
           [build_row(i, False) for i in range(n_fraud, N_SAMPLES)]
    random.shuffle(rows)
    for i, r in enumerate(rows):
        r["job_id"] = i + 1

    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "job_postings.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows -> {out_path}")
    print(f"  fraudulent=1: {n_fraud}  ({n_fraud/N_SAMPLES:.1%})")
    print(f"  fraudulent=0: {n_legit}  ({n_legit/N_SAMPLES:.1%})")


if __name__ == "__main__":
    main()
