# tools.py
# ---------
# Each "tool" is a plain Python function the agent can call.
# Tools are kept separate from agent logic so they're easy to test independently.
#
# Tools in this file:
#   generate_quiz(subject, topics, n_questions)  → list of MCQ dicts
#   score_quiz(questions, answers)               → score + weak topics
#   build_study_plan(subject, weak_topics, exam_date) → day-by-day plan
#   search_resources(topic, subject)             → list of real URLs
#   get_nudge_message(student_id, next_topic)    → personalized reminder text

import json
import re
import time
import urllib.request
from datetime import date, datetime
from typing import Optional

from llm import call_llm
from db import get_completed_topics, get_quiz_history


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 1 — QuizGeneratorTool
# ══════════════════════════════════════════════════════════════════════════════

# The exact JSON shape we expect from the LLM
_QUIZ_SCHEMA_EXAMPLE = """
[
  {
    "question": "What is gradient descent?",
    "options": ["A) An optimisation algorithm", "B) A data type", "C) A loss function", "D) A neural layer"],
    "answer": "A",
    "topic": "Gradient Descent"
  }
]
"""


def generate_quiz(subject: str, topics: list[str], n_questions: int = 6) -> list[dict]:
    """
    Ask the LLM to create an MCQ diagnostic quiz.

    Returns a list of question dicts, each with:
      question, options (list of 4), answer (A/B/C/D), topic

    Retries up to 3 times if the JSON is malformed.
    """
    topics_str = ", ".join(topics) if topics else subject

    prompt = f"""You are a quiz generator for a student studying {subject}.

Create exactly {n_questions} multiple-choice questions covering these topics: {topics_str}.

Rules:
- Each question MUST have exactly 4 options labelled A), B), C), D).
- The "answer" field must be a single letter: A, B, C, or D.
- Vary difficulty: roughly 1/3 easy, 1/3 medium, 1/3 hard.
- Return ONLY a valid JSON array. No markdown, no explanation.

Use exactly this shape:
{_QUIZ_SCHEMA_EXAMPLE}
"""

    for attempt in range(1, 4):  # up to 3 retries
        try:
            raw = call_llm(prompt, json_mode=True)
            # Strip markdown fences if the model added them anyway
            raw = re.sub(r"```(?:json)?|```", "", raw).strip()
            questions = json.loads(raw)

            # ── validate structure ──────────────────────────────────────────
            if not isinstance(questions, list) or len(questions) == 0:
                raise ValueError("Expected a non-empty JSON array.")
            for q in questions:
                for key in ("question", "options", "answer", "topic"):
                    if key not in q:
                        raise ValueError(f"Missing key '{key}' in question: {q}")
                if len(q["options"]) != 4:
                    raise ValueError("Each question must have exactly 4 options.")
                if q["answer"] not in ("A", "B", "C", "D"):
                    raise ValueError(f"Invalid answer '{q['answer']}' — must be A/B/C/D.")

            return questions

        except (json.JSONDecodeError, ValueError) as exc:
            if attempt == 3:
                raise RuntimeError(
                    f"Quiz generation failed after 3 attempts: {exc}"
                ) from exc
            time.sleep(1)  # brief pause before retry

    return []  # unreachable but satisfies type checker


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 2 — Gap Analysis (scores the quiz, finds weak topics)
# ══════════════════════════════════════════════════════════════════════════════

def score_quiz(questions: list[dict], student_answers: dict[str, str]) -> dict:
    """
    Compare the student's answers against the correct answers.

    Args:
        questions:       The list of MCQ dicts from generate_quiz().
        student_answers: Dict mapping question index (str) to chosen letter.
                         e.g. {"0": "A", "1": "C", ...}

    Returns:
        {
            "score": 0.67,           # fraction correct
            "correct": 4,
            "total": 6,
            "weak_topics": ["Gradient Descent", "Backpropagation"],
            "strong_topics": ["Linear Regression", ...]
        }
    """
    correct_count = 0
    topic_correct: dict[str, list[bool]] = {}

    for i, q in enumerate(questions):
        student_ans = student_answers.get(str(i), "").strip().upper()
        is_correct = student_ans == q["answer"].strip().upper()
        if is_correct:
            correct_count += 1

        topic = q.get("topic", "General")
        topic_correct.setdefault(topic, []).append(is_correct)

    total = len(questions)
    score = correct_count / total if total > 0 else 0.0

    # A topic is "weak" if the student got < 60 % of its questions right
    weak_topics = [
        t for t, results in topic_correct.items()
        if sum(results) / len(results) < 0.6
    ]
    strong_topics = [
        t for t, results in topic_correct.items()
        if sum(results) / len(results) >= 0.6
    ]

    return {
        "score": round(score, 3),
        "correct": correct_count,
        "total": total,
        "weak_topics": weak_topics,
        "strong_topics": strong_topics,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 3 — PlanBuilderTool
# ══════════════════════════════════════════════════════════════════════════════

def build_study_plan(subject: str, weak_topics: list[str],
                     exam_date: str, student_id: str) -> list[dict]:
    """
    Ask the LLM to create a day-by-day study plan.

    - Weak topics come first and get more days.
    - Already-completed topics are skipped.
    - Returns a list of dicts: [{day, date, topic, task, duration_mins}, ...]
    """
    completed = get_completed_topics(student_id)
    topics_to_cover = [t for t in weak_topics if t not in completed]

    if not topics_to_cover:
        # Nothing weak left — cover the whole subject broadly
        topics_to_cover = [f"{subject} — General Review"]

    today = date.today().isoformat()
    prompt = f"""You are an academic planner for a student studying {subject}.

Today's date: {today}
Exam date: {exam_date}
Topics the student is WEAK on (prioritise these): {', '.join(topics_to_cover)}
Topics already completed (SKIP these): {', '.join(completed) if completed else 'none'}

Create a realistic day-by-day study plan from {today} to {exam_date}.
- Each day should have ONE focused topic and ONE concrete task (e.g. "Watch intro video + solve 5 practice problems").
- Weak topics should appear first and may get 2 days each.
- Leave the last 2 days for full revision.
- Keep daily study to 60–90 minutes.
- Return ONLY a valid JSON array. No markdown, no explanation.

Exact shape (one object per day):
[
  {{"day": 1, "date": "YYYY-MM-DD", "topic": "...", "task": "...", "duration_mins": 75}},
  ...
]
"""

    for attempt in range(1, 4):
        try:
            raw = call_llm(prompt, json_mode=True)
            raw = re.sub(r"```(?:json)?|```", "", raw).strip()
            plan = json.loads(raw)

            if not isinstance(plan, list) or len(plan) == 0:
                raise ValueError("Expected a non-empty plan array.")
            for entry in plan:
                for key in ("day", "date", "topic", "task", "duration_mins"):
                    if key not in entry:
                        raise ValueError(f"Plan entry missing key '{key}'.")
            return plan

        except (json.JSONDecodeError, ValueError) as exc:
            if attempt == 3:
                raise RuntimeError(
                    f"Study plan generation failed after 3 attempts: {exc}"
                ) from exc
            time.sleep(1)

    return []


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 4 — ResourceSearchTool
# ══════════════════════════════════════════════════════════════════════════════

def _url_is_reachable(url: str, timeout: int = 4) -> bool:
    """Quick HEAD request to check if a URL actually loads."""
    try:
        req = urllib.request.Request(url, method="HEAD")
        req.add_header("User-Agent", "Mozilla/5.0")
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except Exception:
        return False


def search_resources(topic: str, subject: str, max_results: int = 3) -> list[dict]:
    """
    Search DuckDuckGo (no API key needed) for real learning resources.
    Returns up to max_results dicts: [{title, url, source}, ...]

    We use the duckduckgo_search library (DDGS) — no LLM-invented links.
    Links are validated before being returned.
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return [{"title": "Install duckduckgo-search",
                 "url": "https://pypi.org/project/duckduckgo-search/",
                 "source": "setup"}]

    query = f"{subject} {topic} tutorial free lecture site:youtube.com OR site:nptel.ac.in OR site:khanacademy.org OR site:coursera.org"

    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=10):
                url = r.get("href", "")
                title = r.get("title", "Resource")
                if url and _url_is_reachable(url):
                    results.append({
                        "title": title,
                        "url": url,
                        "source": _friendly_source(url),
                    })
                if len(results) >= max_results:
                    break
    except Exception as exc:
        # Network failure — return a helpful message instead of crashing
        return [{"title": f"Search unavailable ({exc})",
                 "url": "https://nptel.ac.in",
                 "source": "fallback"}]

    # If DuckDuckGo returned nothing reachable, provide NPTEL fallback
    if not results:
        results.append({
            "title": f"Search NPTEL for '{topic}'",
            "url": f"https://nptel.ac.in/search?q={topic.replace(' ', '+')}",
            "source": "NPTEL",
        })

    return results


def _friendly_source(url: str) -> str:
    """Extract a readable source name from a URL."""
    for domain, name in [
        ("youtube.com", "YouTube"),
        ("nptel.ac.in", "NPTEL"),
        ("khanacademy.org", "Khan Academy"),
        ("coursera.org", "Coursera"),
        ("mit.edu", "MIT OpenCourseWare"),
        ("edx.org", "edX"),
    ]:
        if domain in url:
            return name
    # Generic: extract domain
    match = re.search(r"https?://(?:www\.)?([^/]+)", url)
    return match.group(1) if match else "Web"


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 5 — NudgeTool
# ══════════════════════════════════════════════════════════════════════════════

def generate_nudge(student_id: str, subject: str,
                   days_inactive: int, next_topic: Optional[str]) -> str:
    """
    Generate a short, warm, personalised nudge message.

    Uses the student_id (anonymous) — the student's real name is NOT sent to the LLM.

    Args:
        student_id:    Anonymous ID (e.g. "stu_abc123")
        subject:       The subject being studied
        days_inactive: How many days since last activity (for tone adjustment)
        next_topic:    The next topic in their study plan (may be None)

    Returns:
        A friendly nudge string (2–4 sentences).
    """
    topic_hint = f"Your next topic is: {next_topic}." if next_topic else ""

    prompt = f"""You are a friendly, encouraging AI study coach.
A student (anonymous) studying {subject} has been inactive for {days_inactive} day(s).
{topic_hint}

Write a short (2–4 sentences) warm and motivating nudge message to get them back on track.
- Do NOT use the student's name (you don't have it).
- Be encouraging, not scolding.
- Mention the specific topic if provided.
- End with a concrete suggestion (e.g. "Try spending just 20 minutes on X today.").
- Plain text only; no markdown.
"""

    try:
        return call_llm(prompt).strip()
    except RuntimeError as exc:
        # Fallback if LLM is down — still useful for demo
        return (
            f"Hey! It looks like you haven't studied {subject} in {days_inactive} day(s). "
            f"{'Your next topic is ' + next_topic + '.' if next_topic else ''} "
            "Even 20 minutes today will keep the momentum going. You've got this! 💪"
        )


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 6 — Post-topic quiz result handler
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_post_quiz(pre_score: Optional[float],
                       post_score: float,
                       topic: str) -> dict:
    """
    Decide whether the student passes the topic or needs extra help.

    Rules:
      - Pass if post_score >= 0.6 (60 %)
      - Pass if pre_score exists and post_score improved by >= 0.15 (15 pp)

    Returns:
        {
            "passed": True/False,
            "message": "...",
            "suggest_easier": True/False   # if False, unlock next topic
        }
    """
    improved = (pre_score is not None) and (post_score - pre_score >= 0.15)
    passed = post_score >= 0.6 or improved

    if passed:
        return {
            "passed": True,
            "message": (
                f"✅ Great work! You passed the '{topic}' check — "
                f"score {round(post_score * 100)}%. Moving to the next topic."
            ),
            "suggest_easier": False,
        }
    else:
        return {
            "passed": False,
            "message": (
                f"📚 Score: {round(post_score * 100)}% on '{topic}'. "
                "That's okay — let's try a slightly easier resource before moving on."
            ),
            "suggest_easier": True,
        }
