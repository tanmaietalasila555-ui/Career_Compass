import json
import sqlite3
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

from src.core import (
    CAREER_PROFILES, DEFAULT_SKILLS, ROADMAP_TEMPLATES,
    calculate_skill_gaps, extract_skills, build_job_match,
    load_jobs, train_role_classifier, DB_PATH, init_db,
    save_profile, load_profile, save_task, load_tasks,
    add_job_analysis, load_job_analyses
)

st.set_page_config(page_title="CareerCompass", page_icon="🧭", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
.hero {padding: 1.35rem 1.5rem; border-radius: 16px; background: linear-gradient(120deg,#eef2ff,#f0fdfa); margin-bottom: 1rem;}
.hero h1 {margin:0; color:#172554;}
.hero p {margin:.35rem 0 0 0; color:#334155;}
.small-note {color:#64748b; font-size:.9rem;}
</style>
""", unsafe_allow_html=True)

init_db()

if "profile" not in st.session_state:
    st.session_state.profile = load_profile() or {
        "name": "", "target_role": "AI/ML Engineer", "weekly_hours": 10,
        "skills": {skill: 0 for skill in DEFAULT_SKILLS}
    }
if "tasks" not in st.session_state:
    st.session_state.tasks = load_tasks()

st.markdown("""
<div class="hero">
  <h1>🧭 CareerCompass</h1>
  <p>Your personal career planner, skill-gap analyzer, internship matcher, and learning progress tracker.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Your workspace")
    page = st.radio("Navigate", [
        "Dashboard", "My Skill Profile", "Career Explorer",
        "Skill Gap & Roadmap", "Internship Matcher", "Progress Tracker", "About"
    ])
    st.divider()
    st.caption("Your profile and task progress are saved in a local SQLite database.")
    if st.button("Reset local data", type="secondary"):
        if DB_PATH.exists():
            DB_PATH.unlink()
        init_db()
        st.session_state.profile = {
            "name": "", "target_role": "AI/ML Engineer", "weekly_hours": 10,
            "skills": {skill: 0 for skill in DEFAULT_SKILLS}
        }
        st.session_state.tasks = []
        st.success("Local data reset. Refresh the page if needed.")

profile = st.session_state.profile
role = profile.get("target_role", "AI/ML Engineer")
skills = profile.get("skills", {})
gaps = calculate_skill_gaps(skills, role)

if page == "Dashboard":
    st.subheader("Your career dashboard")
    st.write(f"Welcome{', ' + profile['name'] if profile.get('name') else ''}! Here's your current snapshot.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Target role", role)
    c2.metric("Skills tracked", len(skills))
    c3.metric("Skills to develop", sum(1 for x in gaps if x["gap"] > 0))
    tasks = st.session_state.tasks
    completed = sum(1 for t in tasks if t.get("completed"))
    c4.metric("Tasks completed", f"{completed}/{len(tasks)}" if tasks else "0")
    st.divider()
    left, right = st.columns(2)
    with left:
        st.markdown("#### Skill profile")
        if skills:
            chart = pd.DataFrame({"Skill": list(skills.keys()), "Level": list(skills.values())}).set_index("Skill")
            st.bar_chart(chart)
        else:
            st.info("Add your skills in **My Skill Profile** to see your chart.")
    with right:
        st.markdown("#### Your next steps")
        if gaps:
            for item in sorted(gaps, key=lambda x: x["gap"], reverse=True)[:4]:
                if item["gap"] > 0:
                    st.write(f"**{item['skill']}** — current {item['current']}/5, target {item['target']}/5")
        else:
            st.info("Choose a target career to see skill gaps.")
        st.write("1. Update your actual skill levels.")
        st.write("2. Review the role requirements.")
        st.write("3. Paste an internship description to compare its skills.")
    st.caption("CareerCompass is a planning aid, not a hiring predictor. Skill levels and role profiles are estimates you can edit.")

elif page == "My Skill Profile":
    st.subheader("My skill profile")
    st.write("Rate your current confidence honestly: 0 = not learned yet, 1 = beginner, and 5 = confident independent use.")
    with st.form("profile_form"):
        name = st.text_input("Name (optional)", value=profile.get("name", ""))
        target_role = st.selectbox(
            "Primary target role", list(CAREER_PROFILES.keys()),
            index=list(CAREER_PROFILES.keys()).index(profile.get("target_role", "AI/ML Engineer"))
            if profile.get("target_role", "AI/ML Engineer") in CAREER_PROFILES else 0
        )
        weekly_hours = st.slider("Hours you can spend learning each week", 1, 30, int(profile.get("weekly_hours", 10)))
        st.markdown("#### Rate your skills")
        cols = st.columns(2)
        updated_skills = {}
        for i, skill in enumerate(DEFAULT_SKILLS):
            with cols[i % 2]:
                updated_skills[skill] = st.slider(skill, 0, 5, int(profile.get("skills", {}).get(skill, 0)), key=f"skill_{skill}")
        submitted = st.form_submit_button("Save profile", type="primary")
    if submitted:
        new_profile = {
            "name": name.strip(), "target_role": target_role,
            "weekly_hours": weekly_hours, "skills": updated_skills
        }
        save_profile(new_profile)
        st.session_state.profile = new_profile
        st.success("Profile saved.")
        st.rerun()

elif page == "Career Explorer":
    st.subheader("Career role explorer")
    chosen = st.selectbox("Choose a role to explore", list(CAREER_PROFILES.keys()), index=list(CAREER_PROFILES.keys()).index(role) if role in CAREER_PROFILES else 0)
    details = CAREER_PROFILES[chosen]
    st.markdown(f"### {chosen}")
    st.write(details["summary"])
    st.markdown("**Common work areas**")
    st.write(" • ".join(details["work"]))
    st.markdown("**Example skill profile**")
    role_df = pd.DataFrame([
        {"Skill": skill, "Suggested target level": level}
        for skill, level in details["skills"].items()
    ]).sort_values("Suggested target level", ascending=False)
    st.dataframe(role_df, use_container_width=True, hide_index=True)
    st.info("These skill levels are editable starting assumptions, not official or universal hiring requirements.")
    if st.button(f"Set {chosen} as my target role", type="primary"):
        updated = dict(profile)
        updated["target_role"] = chosen
        save_profile(updated)
        st.session_state.profile = updated
        st.success(f"Target role updated to {chosen}.")
        st.rerun()

elif page == "Skill Gap & Roadmap":
    st.subheader("Skill gap analysis")
    st.write(f"Comparing your profile with the suggested skill profile for **{role}**.")
    gap_df = pd.DataFrame(gaps)
    if not gap_df.empty:
        gap_df = gap_df.sort_values(["gap", "target"], ascending=[False, False])
        st.dataframe(gap_df.rename(columns={
            "skill": "Skill", "current": "Your level", "target": "Suggested target", "gap": "Gap"
        }), use_container_width=True, hide_index=True)
        st.markdown("#### Skill gap chart")
        st.bar_chart(gap_df.set_index("skill")[["current", "target"]])
    st.caption("Gap = max(0, suggested target level − your self-rated level). Adjust your ratings as you learn.")
    st.divider()
    st.markdown("#### Generate a weekly learning plan")
    hours = int(profile.get("weekly_hours", 10))
    st.write(f"Your available learning time: **{hours} hours/week**.")
    suggested = []
    for item in sorted(gaps, key=lambda x: x["gap"], reverse=True):
        if item["gap"] > 0:
            suggestions = ROADMAP_TEMPLATES.get(item["skill"], [f"Study and practise {item['skill']}"])
            suggested.append({"Skill": item["skill"], "Gap": item["gap"], "Practice idea": suggestions[0]})
    if suggested:
        plan_df = pd.DataFrame(suggested[:6])
        st.dataframe(plan_df, use_container_width=True, hide_index=True)
        if st.button("Add these tasks to my tracker", type="primary"):
            existing = {t["title"] for t in st.session_state.tasks}
            added = 0
            for row in suggested[:6]:
                title = f"Practise {row['Skill']}"
                if title not in existing:
                    task = {"title": title, "skill": row["Skill"], "notes": row["Practice idea"], "completed": False, "created_at": date.today().isoformat()}
                    save_task(task)
                    st.session_state.tasks.append(task)
                    added += 1
            st.success(f"Added {added} new task(s) to Progress Tracker.")
    else:
        st.success("Your self-rated skills meet or exceed the current suggested targets. Review the targets and practise with projects to validate your skills.")

elif page == "Internship Matcher":
    st.subheader("Internship description matcher")
    st.write("Paste a job description. CareerCompass identifies known skills and compares them with your self-reported profile.")
    description = st.text_area("Job or internship description", height=220, placeholder="Paste the responsibilities and requirements here...")
    if st.button("Analyze description", type="primary"):
        if len(description.strip()) < 30:
            st.error("Please paste a longer description so there is enough text to analyze.")
        else:
            found = extract_skills(description)
            result = build_job_match(found, skills)
            add_job_analysis(description, found, result["similarity"], result["matched"], result["missing"])
            a, b, c = st.columns(3)
            a.metric("Skills found", len(found))
            b.metric("Skills in your profile", len(result["matched"]))
            c.metric("Text overlap score", f"{result['similarity']:.0%}")
            st.caption("Text overlap is a simple skills-based indicator, not a probability of getting hired or an assessment of your full qualifications.")
            if found:
                match_df = pd.DataFrame([
                    {"Skill": s, "Your level": skills.get(s, 0), "Status": "In profile" if skills.get(s, 0) >= 1 else "Not rated"}
                    for s in found
                ])
                st.dataframe(match_df, use_container_width=True, hide_index=True)
                st.markdown("**Skills found in your profile**")
                st.write(", ".join(result["matched"]) if result["matched"] else "None of the extracted skills are currently rated in your profile.")
                st.markdown("**Skills to review or add to your profile**")
                st.write(", ".join(result["missing"]) if result["missing"] else "All extracted skills are represented in your profile.")
            else:
                st.warning("No skills from the built-in skill dictionary were detected. Try a description with specific technical skills.")
    st.divider()
    st.markdown("#### Optional: classify job role using an ML model")
    st.write("The bundled training examples are small illustrative examples for demonstrating the pipeline. They are not a reliable benchmark. For meaningful results, upload a sufficiently large, representative, correctly labelled dataset.")
    uploaded = st.file_uploader("Upload CSV with columns `description` and `role`", type=["csv"])
    if uploaded is not None:
        try:
            training_df = pd.read_csv(uploaded)
            if not {"description", "role"}.issubset(training_df.columns):
                st.error("CSV must include columns named `description` and `role`.")
            elif training_df.dropna(subset=["description", "role"])["role"].nunique() < 2:
                st.error("The dataset needs at least two different role labels.")
            else:
                cleaned = training_df.dropna(subset=["description", "role"]).copy()
                cleaned["description"] = cleaned["description"].astype(str)
                cleaned["role"] = cleaned["role"].astype(str)
                if cleaned["role"].value_counts().min() < 2:
                    st.error("Each role needs at least two examples to attempt a stratified train/test split.")
                else:
                    if st.button("Train classifier on uploaded data"):
                        try:
                            model, metrics = train_role_classifier(cleaned)
                            st.session_state["trained_model"] = model
                            st.session_state["trained_metrics"] = metrics
                            st.success("Classifier trained in this session.")
                        except Exception as exc:
                            st.error(f"Could not train model: {exc}")
                if "trained_model" in st.session_state and st.button("Predict role for pasted description"):
                    prediction = st.session_state["trained_model"].predict([description])[0]
                    st.info(f"Predicted role: **{prediction}**")
                    if "trained_metrics" in st.session_state:
                        st.write("Held-out test accuracy:", f"{st.session_state['trained_metrics']['accuracy']:.1%}")
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")

elif page == "Progress Tracker":
    st.subheader("Learning progress tracker")
    tasks = st.session_state.tasks
    if not tasks:
        st.info("You have no tasks yet. Generate a plan from **Skill Gap & Roadmap** or add a task below.")
    else:
        completed = sum(bool(t.get("completed")) for t in tasks)
        st.progress(completed / len(tasks))
        st.write(f"Completed **{completed} of {len(tasks)}** tasks.")
        for idx, task in enumerate(tasks):
            with st.container(border=True):
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.markdown(f"**{task['title']}**")
                    if task.get("skill"):
                        st.caption(f"Skill: {task['skill']}")
                    if task.get("notes"):
                        st.write(task["notes"])
                with col2:
                    new_state = st.checkbox("Done", value=bool(task.get("completed")), key=f"done_{task.get('id', idx)}")
                    if new_state != bool(task.get("completed")):
                        task["completed"] = new_state
                        save_task(task)
                        st.session_state.tasks = load_tasks()
                        st.rerun()
    st.divider()
    with st.form("add_task_form", clear_on_submit=True):
        st.markdown("#### Add a custom task")
        title = st.text_input("Task title")
        skill = st.selectbox("Related skill", ["General"] + DEFAULT_SKILLS)
        notes = st.text_input("Notes or resource")
        if st.form_submit_button("Add task"):
            if title.strip():
                task = {"title": title.strip(), "skill": "" if skill == "General" else skill, "notes": notes.strip(), "completed": False, "created_at": date.today().isoformat()}
                save_task(task)
                st.session_state.tasks = load_tasks()
                st.success("Task added.")
                st.rerun()
            else:
                st.error("Enter a task title.")

elif page == "About":
    st.subheader("About CareerCompass")
    st.markdown("""
**CareerCompass** is a local-first student career planning application built with Python, Streamlit, Scikit-learn, Pandas, and SQLite.

**What it demonstrates**
- Data modelling and persistence with SQLite.
- Transparent skill-gap calculations.
- Text-based skill extraction and matching.
- Optional TF-IDF + Linear SVM job-role classification on an uploaded labelled dataset.
- Interactive data visualization and progress tracking.

**Limitations**
- Skill levels and career profiles are simplified starting assumptions.
- Skill extraction uses a fixed dictionary and can miss synonyms or context.
- Text overlap is not a hiring probability.
- The optional ML model is only as reliable as the training data and should be evaluated before real use.
""")
    st.markdown("**Local data file:** `careercompass.db` (created automatically in the project folder).")
