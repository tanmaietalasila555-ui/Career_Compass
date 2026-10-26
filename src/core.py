import json
import re
import sqlite3
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "careercompass.db"

DEFAULT_SKILLS = [
    "Python", "SQL", "Pandas", "NumPy", "Statistics", "Data Visualization",
    "Machine Learning", "Scikit-learn", "Deep Learning", "NLP",
    "Git and GitHub", "Excel", "Data Structures and Algorithms",
    "APIs", "Streamlit", "FastAPI"
]

CAREER_PROFILES = {
    "AI/ML Engineer": {
        "summary": "Build, evaluate, and integrate machine learning systems into software products.",
        "work": ["prepare data", "train and evaluate models", "build ML pipelines", "deploy models", "monitor performance"],
        "skills": {"Python": 4, "SQL": 3, "Pandas": 4, "NumPy": 4, "Statistics": 4, "Data Visualization": 3, "Machine Learning": 5, "Scikit-learn": 4, "Deep Learning": 3, "NLP": 3, "Git and GitHub": 4, "Excel": 2, "Data Structures and Algorithms": 3, "APIs": 3, "Streamlit": 3, "FastAPI": 3}
    },
    "Data Analyst": {
        "summary": "Turn data into insights using analysis, SQL, visualization, and communication.",
        "work": ["clean data", "write SQL queries", "analyze trends", "build dashboards", "communicate findings"],
        "skills": {"Python": 3, "SQL": 5, "Pandas": 4, "NumPy": 3, "Statistics": 4, "Data Visualization": 5, "Machine Learning": 2, "Scikit-learn": 2, "Deep Learning": 1, "NLP": 1, "Git and GitHub": 2, "Excel": 5, "Data Structures and Algorithms": 2, "APIs": 2, "Streamlit": 2, "FastAPI": 1}
    },
    "Data Scientist": {
        "summary": "Use statistics, programming, experimentation, and machine learning to answer data questions.",
        "work": ["explore datasets", "design experiments", "engineer features", "train models", "explain findings"],
        "skills": {"Python": 4, "SQL": 4, "Pandas": 4, "NumPy": 4, "Statistics": 5, "Data Visualization": 4, "Machine Learning": 5, "Scikit-learn": 4, "Deep Learning": 3, "NLP": 3, "Git and GitHub": 3, "Excel": 3, "Data Structures and Algorithms": 3, "APIs": 2, "Streamlit": 2, "FastAPI": 2}
    },
    "Python Developer": {
        "summary": "Develop maintainable Python applications, services, automation, and integrations.",
        "work": ["write application code", "design APIs", "debug and test", "work with databases", "use version control"],
        "skills": {"Python": 5, "SQL": 3, "Pandas": 2, "NumPy": 2, "Statistics": 1, "Data Visualization": 1, "Machine Learning": 1, "Scikit-learn": 1, "Deep Learning": 1, "NLP": 1, "Git and GitHub": 4, "Excel": 1, "Data Structures and Algorithms": 4, "APIs": 5, "Streamlit": 2, "FastAPI": 4}
    },
    "Software Developer": {
        "summary": "Design, build, test, and maintain software applications and services.",
        "work": ["implement features", "write tests", "debug issues", "work with APIs and databases", "collaborate with Git"],
        "skills": {"Python": 4, "SQL": 3, "Pandas": 1, "NumPy": 1, "Statistics": 1, "Data Visualization": 1, "Machine Learning": 1, "Scikit-learn": 1, "Deep Learning": 1, "NLP": 1, "Git and GitHub": 4, "Excel": 1, "Data Structures and Algorithms": 5, "APIs": 4, "Streamlit": 2, "FastAPI": 3}
    }
}

ROADMAP_TEMPLATES = {
    "Python": ["Solve 5 Python problems and build a small script."],
    "SQL": ["Practise SELECT, JOIN, GROUP BY, subqueries, and window functions."],
    "Pandas": ["Clean a CSV, handle missing values, group data, and export a summary."],
    "NumPy": ["Practise arrays, indexing, vectorized operations, and summary statistics."],
    "Statistics": ["Revise distributions, variance, correlation, sampling, and hypothesis testing."],
    "Data Visualization": ["Create a small dashboard with at least three appropriate charts."],
    "Machine Learning": ["Train one baseline model and compare it with a second algorithm."],
    "Scikit-learn": ["Build a Pipeline with preprocessing, training, and evaluation."],
    "Deep Learning": ["Learn tensors and train a small neural network on a suitable dataset."],
    "NLP": ["Clean text and compare keyword matching with TF-IDF similarity."],
    "Git and GitHub": ["Create a repository, write a README, and practise branches and pull requests."],
    "Excel": ["Practise pivot tables, lookup functions, charts, and data cleaning."],
    "Data Structures and Algorithms": ["Solve problems on arrays, strings, stacks, queues, trees, and graphs."],
    "APIs": ["Call a public API, validate its response, and handle errors."],
    "Streamlit": ["Build a multi-page app with forms, charts, and input validation."],
    "FastAPI": ["Create and test a small REST API with request validation."]
}

# Used only as a small illustrative fallback dataset. Replace with real, labelled data before drawing conclusions.
DEMO_JOBS = [
    ("Python, pandas, SQL, Excel, dashboard creation, data cleaning and reporting", "Data Analyst"),
    ("Analyze business metrics with SQL and Python, create reports and visualizations", "Data Analyst"),
    ("Power BI, Excel, SQL queries, data validation, stakeholder reports", "Data Analyst"),
    ("Explore datasets, statistics, pandas, hypothesis testing and data visualization", "Data Analyst"),
    ("Train machine learning models using Python, scikit-learn, pandas and model evaluation", "AI/ML Engineer"),
    ("Build ML pipelines, feature engineering, model deployment, APIs and Python", "AI/ML Engineer"),
    ("Classification regression clustering model evaluation and experiment tracking", "AI/ML Engineer"),
    ("Machine learning, NLP, deep learning, Python, APIs and production pipelines", "AI/ML Engineer"),
    ("Statistical modelling, feature engineering, Python, SQL and predictive analytics", "Data Scientist"),
    ("Analyze datasets, train predictive models, test hypotheses and communicate findings", "Data Scientist"),
    ("Python, pandas, numpy, statistics, scikit-learn, regression and classification", "Data Scientist"),
    ("Experiment design, model validation, exploratory analysis and machine learning", "Data Scientist"),
    ("Build Python applications, REST APIs, unit tests, SQL integration and debugging", "Python Developer"),
    ("Python scripting, automation, FastAPI, Git, databases and application development", "Python Developer"),
    ("Develop backend services in Python, integrate APIs, test and maintain code", "Python Developer"),
    ("Write maintainable Python, debug software, work with Git and databases", "Python Developer"),
    ("Data structures, algorithms, APIs, software testing, version control and Python", "Software Developer"),
    ("Implement software features, write tests, debug code and collaborate using Git", "Software Developer"),
    ("Design backend services, databases, APIs, data structures and algorithms", "Software Developer"),
    ("Build and maintain applications, test code, review pull requests and fix bugs", "Software Developer"),
]

SKILL_ALIASES = {
    "python": "Python", "python programming": "Python",
    "sql": "SQL", "structured query language": "SQL",
    "pandas": "Pandas", "numpy": "NumPy",
    "statistics": "Statistics", "statistical": "Statistics",
    "data visualization": "Data Visualization", "visualization": "Data Visualization", "visualisation": "Data Visualization",
    "machine learning": "Machine Learning", "ml": "Machine Learning",
    "scikit-learn": "Scikit-learn", "sklearn": "Scikit-learn",
    "deep learning": "Deep Learning", "neural network": "Deep Learning",
    "nlp": "NLP", "natural language processing": "NLP",
    "git": "Git and GitHub", "github": "Git and GitHub", "git and github": "Git and GitHub", "version control": "Git and GitHub",
    "excel": "Excel", "data structures and algorithms": "Data Structures and Algorithms", "dsa": "Data Structures and Algorithms",
    "api": "APIs", "apis": "APIs", "rest api": "APIs", "rest apis": "APIs",
    "streamlit": "Streamlit", "fastapi": "FastAPI"
}

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS profile (
            id INTEGER PRIMARY KEY CHECK (id = 1), payload TEXT NOT NULL
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL,
            skill TEXT, notes TEXT, completed INTEGER NOT NULL DEFAULT 0,
            created_at TEXT
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS job_analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
            description TEXT NOT NULL, skills TEXT NOT NULL, similarity REAL,
            matched TEXT NOT NULL, missing TEXT NOT NULL
        )""")
        conn.commit()

def save_profile(profile):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO profile (id, payload) VALUES (1, ?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload",
            (json.dumps(profile),)
        )
        conn.commit()

def load_profile():
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT payload FROM profile WHERE id=1").fetchone()
    return json.loads(row[0]) if row else None

def save_task(task):
    with sqlite3.connect(DB_PATH) as conn:
        if task.get("id") is not None:
            conn.execute(
                "UPDATE tasks SET title=?, skill=?, notes=?, completed=? WHERE id=?",
                (task["title"], task.get("skill", ""), task.get("notes", ""), int(task.get("completed", False)), int(task["id"]))
            )
        else:
            cur = conn.execute(
                "INSERT INTO tasks (title, skill, notes, completed, created_at) VALUES (?, ?, ?, ?, ?)",
                (task["title"], task.get("skill", ""), task.get("notes", ""), int(task.get("completed", False)), task.get("created_at", ""))
            )
            task["id"] = cur.lastrowid
        conn.commit()

def load_tasks():
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("SELECT id, title, skill, notes, completed, created_at FROM tasks ORDER BY id DESC").fetchall()
    return [{"id": r[0], "title": r[1], "skill": r[2], "notes": r[3], "completed": bool(r[4]), "created_at": r[5]} for r in rows]

def add_job_analysis(description, found, similarity, matched, missing):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO job_analyses (created_at, description, skills, similarity, matched, missing) VALUES (?, ?, ?, ?, ?, ?)",
            (datetime_now(), description, json.dumps(found), float(similarity), json.dumps(matched), json.dumps(missing))
        )
        conn.commit()

def datetime_now():
    from datetime import datetime
    return datetime.now().isoformat(timespec="seconds")

def load_job_analyses():
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("SELECT id, created_at, skills, similarity, matched, missing FROM job_analyses ORDER BY id DESC").fetchall()
    return [{"id": r[0], "created_at": r[1], "skills": json.loads(r[2]), "similarity": r[3], "matched": json.loads(r[4]), "missing": json.loads(r[5])} for r in rows]

def calculate_skill_gaps(user_skills, role):
    targets = CAREER_PROFILES.get(role, {}).get("skills", {})
    result = []
    for skill, target in targets.items():
        current = int(user_skills.get(skill, 0))
        result.append({
            "skill": skill, "current": current, "target": int(target),
            "gap": max(0, int(target) - current)
        })
    return result

def extract_skills(text):
    text_lower = re.sub(r"[^a-z0-9+#.\s-]", " ", str(text).lower())
    found = set()
    # Longest phrases first to reduce duplicate/partial matches.
    for alias in sorted(SKILL_ALIASES, key=len, reverse=True):
        pattern = r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])"
        if re.search(pattern, text_lower):
            found.add(SKILL_ALIASES[alias])
    return sorted(found)

def build_job_match(found_skills, user_skills):
    matched = sorted(s for s in found_skills if int(user_skills.get(s, 0)) >= 1)
    missing = sorted(s for s in found_skills if int(user_skills.get(s, 0)) < 1)
    similarity = len(matched) / len(found_skills) if found_skills else 0.0
    return {"matched": matched, "missing": missing, "similarity": similarity}

def load_jobs(csv_path=None):
    if csv_path:
        df = pd.read_csv(csv_path)
        if not {"description", "role"}.issubset(df.columns):
            raise ValueError("CSV needs columns named 'description' and 'role'.")
        return df.dropna(subset=["description", "role"])
    return pd.DataFrame(DEMO_JOBS, columns=["description", "role"])

def train_role_classifier(df):
    df = df.dropna(subset=["description", "role"]).copy()
    df["description"] = df["description"].astype(str)
    df["role"] = df["role"].astype(str)
    if df["role"].nunique() < 2:
        raise ValueError("At least two role labels are required.")
    if df["role"].value_counts().min() < 2:
        raise ValueError("Each role must have at least two examples.")
    X_train, X_test, y_train, y_test = train_test_split(
        df["description"], df["role"], test_size=0.25, random_state=42, stratify=df["role"]
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
        ("classifier", LinearSVC(class_weight="balanced", random_state=42))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "report": classification_report(y_test, pred, zero_division=0, output_dict=True),
        "test_size": int(len(y_test))
    }
    return model, metrics
