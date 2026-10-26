# CareerCompass 🧭

CareerCompass is a student-focused career planning application built with Python, Streamlit, Scikit-learn, Pandas, and SQLite.

## Features

- Personal skill profile with 0–5 self-ratings (0 = not learned yet)
- Career role explorer for AI/ML Engineer, Data Analyst, Data Scientist, Python Developer, and Software Developer
- Transparent skill-gap analysis
- Learning roadmap suggestions and task tracking
- Internship description skill extraction and profile matching
- Optional TF-IDF + Linear SVM role classifier trained on an uploaded labelled CSV
- Local SQLite persistence

## Requirements

- Python 3.10 or newer recommended
- Internet connection only for installing dependencies (the app itself can run locally)

## Run locally

### Windows (PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local address, usually `http://localhost:8501`.

## Uploading your own labelled job dataset

The optional classifier expects a CSV with exactly these required column names (extra columns are okay):

```csv
description,role
"Use Python, pandas, SQL and dashboards to analyze business data","Data Analyst"
"Train machine learning models with Python and scikit-learn","AI/ML Engineer"
```

Use enough representative examples per role, check label quality, and avoid duplicate or near-duplicate descriptions across the train/test split. A larger, balanced, carefully labelled dataset is strongly recommended. Do not report performance on the tiny bundled demonstration examples as meaningful real-world performance.

## Project structure

```text
CareerCompass/
├── app.py
├── src/
│   ├── __init__.py
│   └── core.py
├── data/
├── tests/
├── requirements.txt
├── README.md
└── .gitignore
```

## Limitations

- Career skill profiles are starting assumptions, not universal hiring requirements.
- Self-ratings are subjective and should be validated through projects and assessments.
- Skill extraction uses a fixed dictionary and can miss synonyms, context, or skills not listed.
- The overlap score is not a probability of receiving an interview or job.
- The built-in job examples are illustrative only. Use a real, appropriately licensed and representative dataset before making claims about model quality.
- The app stores data locally in `careercompass.db`. Do not deploy a public shared instance with private user data without adding authentication, access controls, and a privacy review.

## Resume note

Describe only features you implemented and tested. Report model metrics only from a suitable evaluation dataset and clearly state how evaluation was performed.
