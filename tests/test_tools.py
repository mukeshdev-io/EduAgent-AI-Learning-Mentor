# tests/test_tools.py
# --------------------
# Unit tests for EduAgent tools.
# Run with:  python -m pytest tests/ -v  (from inside the eduagent/ folder)
#
# These tests do NOT call the real LLM — they mock call_llm so tests
# run offline and fast.

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import pytest
from unittest.mock import patch


# ══════════════════════════════════════════════════════════════════════════════
# Sample 1 — Quiz generation and scoring (Machine Learning)
# ══════════════════════════════════════════════════════════════════════════════

SAMPLE_QUIZ_JSON = json.dumps([
    {
        "question": "Which algorithm is used to minimise loss in neural networks?",
        "options": ["A) Bubble Sort", "B) Gradient Descent", "C) Binary Search", "D) K-Means"],
        "answer": "B",
        "topic": "Optimisation"
    },
    {
        "question": "What does 'overfitting' mean?",
        "options": ["A) Model trains too fast", "B) Model memorises training data", "C) Model underpredicts", "D) Model is too simple"],
        "answer": "B",
        "topic": "Model Evaluation"
    },
    {
        "question": "Which library is commonly used for ML in Python?",
        "options": ["A) Flask", "B) Django", "C) scikit-learn", "D) NumPy"],
        "answer": "C",
        "topic": "Python for ML"
    },
])


def test_generate_quiz_returns_valid_structure():
    """
    generate_quiz() should return a list of dicts with the required keys.
    We mock the LLM so no API key is needed.
    """
    from tools import generate_quiz
    with patch("tools.call_llm", return_value=SAMPLE_QUIZ_JSON):
        questions = generate_quiz("Machine Learning", ["Optimisation"], n_questions=3)

    assert isinstance(questions, list)
    assert len(questions) == 3
    for q in questions:
        assert "question" in q
        assert "options" in q
        assert "answer" in q
        assert "topic" in q
        assert len(q["options"]) == 4
        assert q["answer"] in ("A", "B", "C", "D")


def test_generate_quiz_retries_on_bad_json():
    """
    generate_quiz() should retry if the LLM returns invalid JSON,
    and succeed on the third attempt.
    """
    from tools import generate_quiz
    call_count = [0]

    def flaky_llm(prompt, json_mode=False):
        call_count[0] += 1
        if call_count[0] < 3:
            return "This is not valid JSON {{{{"
        return SAMPLE_QUIZ_JSON

    with patch("tools.call_llm", side_effect=flaky_llm):
        questions = generate_quiz("ML", ["Optimisation"], n_questions=3)

    assert call_count[0] == 3
    assert len(questions) == 3


def test_score_quiz_identifies_weak_topics():
    """
    score_quiz() should mark a topic as 'weak' when the student answers < 60%.
    Expected: Optimisation is strong (answered correctly), Model Evaluation is weak.
    """
    from tools import score_quiz

    questions = json.loads(SAMPLE_QUIZ_JSON)
    # Student answers: Q0 correct (B), Q1 wrong (A), Q2 correct (C)
    answers = {"0": "B", "1": "A", "2": "C"}
    result = score_quiz(questions, answers)

    assert result["correct"] == 2
    assert result["total"] == 3
    assert abs(result["score"] - 2/3) < 0.01
    assert "Model Evaluation" in result["weak_topics"]
    assert "Optimisation" in result["strong_topics"]


# ══════════════════════════════════════════════════════════════════════════════
# Sample 2 — Study plan generation (Python Programming)
# ══════════════════════════════════════════════════════════════════════════════

SAMPLE_PLAN_JSON = json.dumps([
    {"day": 1, "date": "2025-08-01", "topic": "Functions", "task": "Read chapter + write 5 functions", "duration_mins": 60},
    {"day": 2, "date": "2025-08-02", "topic": "Loops",     "task": "Solve 10 loop exercises",          "duration_mins": 75},
    {"day": 3, "date": "2025-08-03", "topic": "OOP",       "task": "Build a small class hierarchy",    "duration_mins": 90},
])


def test_build_study_plan_structure():
    """
    build_study_plan() should return a list of day entries with required keys.
    """
    import db
    from tools import build_study_plan

    db.init_db()

    # Use a throwaway student ID so this test doesn't pollute real data
    test_sid = "stu_test_plan"
    db.save_student(test_sid, "Test User", "Python", "Pass exam", "2025-08-10")

    with patch("tools.call_llm", return_value=SAMPLE_PLAN_JSON):
        plan = build_study_plan(
            subject="Python",
            weak_topics=["Functions", "Loops"],
            exam_date="2025-08-10",
            student_id=test_sid,
        )

    assert isinstance(plan, list)
    assert len(plan) == 3
    for entry in plan:
        for key in ("day", "date", "topic", "task", "duration_mins"):
            assert key in entry, f"Missing key: {key}"


# ══════════════════════════════════════════════════════════════════════════════
# Sample 3 — Post-topic quiz evaluation (Data Structures)
# ══════════════════════════════════════════════════════════════════════════════

def test_evaluate_post_quiz_pass_on_improvement():
    """
    A student who improves by ≥ 15 percentage points should pass,
    even if they are below 60%.
    """
    from tools import evaluate_post_quiz
    result = evaluate_post_quiz(pre_score=0.30, post_score=0.50, topic="Binary Trees")
    assert result["passed"] is True
    assert "passed" in result["message"].lower() or "great" in result["message"].lower()


def test_evaluate_post_quiz_fail_below_threshold():
    """
    A student with post_score < 0.6 and no significant improvement should fail.
    """
    from tools import evaluate_post_quiz
    result = evaluate_post_quiz(pre_score=0.40, post_score=0.45, topic="Binary Trees")
    assert result["passed"] is False
    assert result["suggest_easier"] is True


def test_evaluate_post_quiz_pass_above_threshold():
    """
    A student who scores >= 60% outright should pass regardless of pre-score.
    """
    from tools import evaluate_post_quiz
    result = evaluate_post_quiz(pre_score=0.20, post_score=0.80, topic="Heaps")
    assert result["passed"] is True
    assert result["suggest_easier"] is False


# ══════════════════════════════════════════════════════════════════════════════
# Database tests
# ══════════════════════════════════════════════════════════════════════════════

def test_db_save_and_retrieve_student():
    """Saving a student profile should persist and be retrievable."""
    import db
    db.init_db()
    db.save_student("stu_dbtest", "Ravi", "Data Structures", "Pass", "2025-09-01")
    profile = db.get_student("stu_dbtest")
    assert profile is not None
    assert profile["subject"] == "Data Structures"
    assert profile["exam_date"] == "2025-09-01"


def test_db_quiz_attempt_and_history():
    """Saving quiz attempts should appear in history, latest-first."""
    import db
    db.init_db()
    db.save_quiz_attempt("stu_dbtest", "diagnostic", 0.5)
    db.save_quiz_attempt("stu_dbtest", "diagnostic", 0.7)
    history = db.get_quiz_history("stu_dbtest")
    assert len(history) >= 2
    # Most recent first
    assert history[0]["score"] == 0.7


def test_db_completed_topics():
    """Marking a topic complete should make it appear in completed list."""
    import db
    db.init_db()
    db.mark_topic_complete("stu_dbtest", "Linked Lists")
    completed = db.get_completed_topics("stu_dbtest")
    assert "Linked Lists" in completed
