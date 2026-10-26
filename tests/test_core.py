from src.core import calculate_skill_gaps, extract_skills, build_job_match

def test_skill_extraction():
    found = extract_skills("We need Python, SQL, Pandas and machine learning.")
    assert "Python" in found
    assert "SQL" in found
    assert "Pandas" in found
    assert "Machine Learning" in found

def test_gap_calculation():
    result = calculate_skill_gaps({"Python": 4}, "AI/ML Engineer")
    python = next(item for item in result if item["skill"] == "Python")
    assert python["current"] == 4
    assert python["target"] == 4
    assert python["gap"] == 0

def test_job_match():
    result = build_job_match(["Python", "SQL"], {"Python": 4, "SQL": 0})
    assert result["matched"] == ["Python"]
    assert result["missing"] == ["SQL"]
    assert result["similarity"] == 0.5
