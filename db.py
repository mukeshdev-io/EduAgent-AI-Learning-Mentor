# db.py
# ------
# All database logic lives here.
# We use SQLite — a single file, no server needed, great for demos.
#
# Tables:
#   students   — one row per student session (anonymous ID + profile)
#   quiz_attempts — every quiz taken, with the raw score
#   completed_topics — topics the student has finished
#   study_plans — the generated day-by-day plan (stored as JSON text)

import sqlite3
import json
import os
from datetime import datetime

# The database file is created next to this script
DB_PATH = os.path.join(os.path.dirname(__file__), "eduagent.db")


def _get_conn() -> sqlite3.Connection:
    """Open (or create) the SQLite database and return a connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # lets us access columns by name
    return conn


def init_db() -> None:
    """
    Create all tables if they don't already exist.
    Call this once at app startup.
    """
    conn = _get_conn()
    with conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS students (
                student_id   TEXT PRIMARY KEY,
                name         TEXT,       -- display name (not sent to LLM)
                subject      TEXT,
                goal         TEXT,
                exam_date    TEXT,
                created_at   TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS quiz_attempts (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id   TEXT,
                topic        TEXT,       -- 'diagnostic' for the first quiz
                score        REAL,       -- 0.0 – 1.0
                attempted_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (student_id) REFERENCES students(student_id)
            );

            CREATE TABLE IF NOT EXISTS completed_topics (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id   TEXT,
                topic        TEXT,
                completed_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (student_id) REFERENCES students(student_id)
            );

            CREATE TABLE IF NOT EXISTS study_plans (
                student_id   TEXT PRIMARY KEY,
                plan_json    TEXT,       -- the full plan as a JSON string
                created_at   TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (student_id) REFERENCES students(student_id)
            );

            CREATE TABLE IF NOT EXISTS last_active (
                student_id   TEXT PRIMARY KEY,
                active_at    TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (student_id) REFERENCES students(student_id)
            );
        """)
    conn.close()


# ── Student ────────────────────────────────────────────────────────────────────

def save_student(student_id: str, name: str, subject: str,
                 goal: str, exam_date: str) -> None:
    """Insert or replace a student profile row."""
    conn = _get_conn()
    with conn:
        conn.execute(
            """INSERT OR REPLACE INTO students
               (student_id, name, subject, goal, exam_date)
               VALUES (?, ?, ?, ?, ?)""",
            (student_id, name, subject, goal, exam_date),
        )
    conn.close()


def get_student(student_id: str) -> dict | None:
    """Return a student's profile dict, or None if not found."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM students WHERE student_id = ?", (student_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


# ── Quiz attempts ──────────────────────────────────────────────────────────────

def save_quiz_attempt(student_id: str, topic: str, score: float) -> None:
    """Record the result of one quiz (score is a fraction 0.0–1.0)."""
    conn = _get_conn()
    with conn:
        conn.execute(
            """INSERT INTO quiz_attempts (student_id, topic, score)
               VALUES (?, ?, ?)""",
            (student_id, topic, round(score, 3)),
        )
    conn.close()
    touch_last_active(student_id)


def get_quiz_history(student_id: str) -> list[dict]:
    """Return all quiz attempts for a student, most recent first."""
    conn = _get_conn()
    rows = conn.execute(
        """SELECT topic, score, attempted_at FROM quiz_attempts
           WHERE student_id = ? ORDER BY attempted_at DESC""",
        (student_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_latest_score(student_id: str, topic: str) -> float | None:
    """Return the most recent score for a specific topic, or None."""
    conn = _get_conn()
    row = conn.execute(
        """SELECT score FROM quiz_attempts
           WHERE student_id = ? AND topic = ?
           ORDER BY attempted_at DESC LIMIT 1""",
        (student_id, topic),
    ).fetchone()
    conn.close()
    return row["score"] if row else None


# ── Completed topics ───────────────────────────────────────────────────────────

def mark_topic_complete(student_id: str, topic: str) -> None:
    """Mark a topic as completed (called when post-quiz score is good enough)."""
    conn = _get_conn()
    with conn:
        conn.execute(
            """INSERT OR IGNORE INTO completed_topics (student_id, topic)
               VALUES (?, ?)""",
            (student_id, topic),
        )
    conn.close()


def get_completed_topics(student_id: str) -> list[str]:
    """Return a list of topic names the student has completed."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT topic FROM completed_topics WHERE student_id = ?",
        (student_id,),
    ).fetchall()
    conn.close()
    return [r["topic"] for r in rows]


# ── Study plan ─────────────────────────────────────────────────────────────────

def save_study_plan(student_id: str, plan: list) -> None:
    """Persist the generated study plan as JSON."""
    conn = _get_conn()
    with conn:
        conn.execute(
            """INSERT OR REPLACE INTO study_plans (student_id, plan_json)
               VALUES (?, ?)""",
            (student_id, json.dumps(plan)),
        )
    conn.close()


def get_study_plan(student_id: str) -> list | None:
    """Return the study plan list, or None if not yet generated."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT plan_json FROM study_plans WHERE student_id = ?",
        (student_id,),
    ).fetchone()
    conn.close()
    return json.loads(row["plan_json"]) if row else None


# ── Last-active tracking (for nudge logic) ─────────────────────────────────────

def touch_last_active(student_id: str) -> None:
    """Update the 'last seen' timestamp to now."""
    conn = _get_conn()
    with conn:
        conn.execute(
            """INSERT OR REPLACE INTO last_active (student_id, active_at)
               VALUES (?, datetime('now'))""",
            (student_id,),
        )
    conn.close()


def get_last_active(student_id: str) -> str | None:
    """Return the ISO timestamp of when the student was last active."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT active_at FROM last_active WHERE student_id = ?",
        (student_id,),
    ).fetchone()
    conn.close()
    return row["active_at"] if row else None


def set_last_active_days_ago(student_id: str, days: int) -> None:
    """
    Simulate inactivity by backdating the last-active timestamp.
    Used by the NudgeTool demo button.
    """
    conn = _get_conn()
    with conn:
        conn.execute(
            """INSERT OR REPLACE INTO last_active (student_id, active_at)
               VALUES (?, datetime('now', ?))""",
            (student_id, f"-{days} days"),
        )
    conn.close()
