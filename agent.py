# agent.py
# ---------
# The EduAgent state machine.
#
# Instead of LangGraph (which adds complexity), we use a plain Python
# state machine — simpler to explain in a presentation, easier to debug,
# and still fully "agentic" (plans, decides, remembers, acts autonomously).
#
# States:
#   PROFILE        → collect student info
#   DIAGNOSTIC     → generate + administer the first quiz
#   GAP_ANALYSIS   → score quiz, identify weak topics
#   PLAN           → build day-by-day study plan
#   RESOURCES      → search for free learning resources per topic
#   STUDYING       → student works through their plan
#   POST_QUIZ      → check mastery after finishing a topic
#   COMPLETE       → all topics done

from __future__ import annotations

import uuid
from datetime import date
from typing import Optional

import db
import tools


# ── State constants ─────────────────────────────────────────────────────────
STATE_PROFILE      = "profile"
STATE_DIAGNOSTIC   = "diagnostic"
STATE_GAP_ANALYSIS = "gap_analysis"
STATE_PLAN         = "plan"
STATE_RESOURCES    = "resources"
STATE_STUDYING     = "studying"
STATE_POST_QUIZ    = "post_quiz"
STATE_COMPLETE     = "complete"


class EduAgent:
    """
    A stateful AI learning-mentor agent for one student session.

    All state is stored in two places:
      - self._state  → the current step in the workflow (in-memory)
      - SQLite DB    → scores, plans, completed topics (persistent)

    The UI (app.py) calls agent methods and reads agent properties;
    it never touches the database or tools directly.
    """

    def __init__(self, student_id: Optional[str] = None):
        """
        Create or resume an agent session.

        If student_id is given AND a profile exists in the DB, we resume
        from where the student left off (STUDYING state).
        Otherwise we start fresh from PROFILE.
        """
        db.init_db()  # ensure tables exist (idempotent)

        if student_id and db.get_student(student_id):
            self.student_id = student_id
            self._state = STATE_STUDYING   # resume
        else:
            # New student — generate an anonymous ID (UUID, no PII)
            self.student_id = f"stu_{uuid.uuid4().hex[:8]}"
            self._state = STATE_PROFILE

        # In-memory working data (not persisted between app restarts
        # except via the DB helpers above)
        self.questions: list[dict] = []          # current quiz questions
        self.gap_result: dict = {}               # output of score_quiz()
        self.study_plan: list[dict] = []         # day-by-day plan
        self.resources: dict[str, list] = {}     # topic → resource list
        self.current_topic_index: int = 0        # pointer into study plan

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def state(self) -> str:
        return self._state

    @property
    def profile(self) -> Optional[dict]:
        return db.get_student(self.student_id)

    # ── Step 1: save profile ──────────────────────────────────────────────────

    def set_profile(self, name: str, subject: str,
                    goal: str, exam_date: str) -> None:
        """
        Save the student's profile and advance to DIAGNOSTIC state.
        The student's real name is stored locally only — never sent to the LLM.
        """
        db.save_student(self.student_id, name, subject, goal, exam_date)
        db.touch_last_active(self.student_id)
        self._state = STATE_DIAGNOSTIC

    # ── Step 2: generate diagnostic quiz ─────────────────────────────────────

    def generate_diagnostic_quiz(self) -> list[dict]:
        """
        Generate a 6-question MCQ quiz covering the student's subject.
        Stores the questions in self.questions and advances state.
        """
        profile = self.profile
        if not profile:
            raise RuntimeError("Profile not set. Call set_profile() first.")

        # Broad topics — we'll discover weak ones from the quiz results
        broad_topics = [profile["subject"]]
        self.questions = tools.generate_quiz(
            subject=profile["subject"],
            topics=broad_topics,
            n_questions=6,
        )
        self._state = STATE_GAP_ANALYSIS
        return self.questions

    # ── Step 3: score quiz and find gaps ─────────────────────────────────────

    def submit_quiz_answers(self, answers: dict[str, str]) -> dict:
        """
        Score the diagnostic quiz and identify weak topics.

        Args:
            answers: Dict mapping question index (str) to letter (A/B/C/D).

        Returns:
            The gap result dict from tools.score_quiz().
        """
        if not self.questions:
            raise RuntimeError("No quiz loaded. Call generate_diagnostic_quiz() first.")

        profile = self.profile
        self.gap_result = tools.score_quiz(self.questions, answers)

        # Persist the diagnostic score
        db.save_quiz_attempt(
            self.student_id, "diagnostic", self.gap_result["score"]
        )

        self._state = STATE_PLAN
        return self.gap_result

    # ── Step 4: build study plan ──────────────────────────────────────────────

    def build_plan(self) -> list[dict]:
        """
        Create a personalised day-by-day study plan based on weak topics.
        Saves the plan to DB and advances to RESOURCES state.
        """
        profile = self.profile
        if not profile:
            raise RuntimeError("Profile not set.")
        if not self.gap_result:
            raise RuntimeError("Gap analysis not done. Submit quiz answers first.")

        self.study_plan = tools.build_study_plan(
            subject=profile["subject"],
            weak_topics=self.gap_result.get("weak_topics", []),
            exam_date=profile["exam_date"],
            student_id=self.student_id,
        )
        db.save_study_plan(self.student_id, self.study_plan)
        db.touch_last_active(self.student_id)
        self._state = STATE_RESOURCES
        return self.study_plan

    # ── Step 5: search resources ──────────────────────────────────────────────

    def fetch_resources(self) -> dict[str, list]:
        """
        For each unique topic in the study plan, search for 3 free resources.
        Returns a dict: {topic: [resource_dicts]}.
        """
        profile = self.profile
        if not profile:
            raise RuntimeError("Profile not set.")
        if not self.study_plan:
            raise RuntimeError("Study plan not built.")

        unique_topics = list({entry["topic"] for entry in self.study_plan})
        self.resources = {}
        for topic in unique_topics:
            self.resources[topic] = tools.search_resources(
                topic=topic, subject=profile["subject"]
            )

        self._state = STATE_STUDYING
        db.touch_last_active(self.student_id)
        return self.resources

    # ── Step 6: post-topic quiz ───────────────────────────────────────────────

    def run_post_topic_quiz(self, topic: str,
                            answers: dict[str, str]) -> dict:
        """
        Generate and score a short (3-question) post-topic quiz.

        Returns:
            evaluate_post_quiz() result with passed/message/suggest_easier.
        """
        profile = self.profile
        if not profile:
            raise RuntimeError("Profile not set.")

        # Retrieve any previous diagnostic score for this topic
        pre_score = db.get_latest_score(self.student_id, topic)

        # Generate a focused 3-question quiz on just this topic
        post_questions = tools.generate_quiz(
            subject=profile["subject"],
            topics=[topic],
            n_questions=3,
        )
        result = tools.score_quiz(post_questions, answers)
        post_score = result["score"]

        db.save_quiz_attempt(self.student_id, topic, post_score)

        evaluation = tools.evaluate_post_quiz(pre_score, post_score, topic)
        if evaluation["passed"]:
            db.mark_topic_complete(self.student_id, topic)

        db.touch_last_active(self.student_id)
        return {**evaluation, "post_questions": post_questions, "score_result": result}

    # ── Step 7: nudge ─────────────────────────────────────────────────────────

    def get_nudge(self) -> str:
        """
        Check inactivity and generate a personalised nudge message.
        Returns the nudge string.
        """
        from datetime import datetime

        profile = self.profile
        if not profile:
            return "Set up your profile to get started!"

        last_active_str = db.get_last_active(self.student_id)
        if last_active_str:
            last_dt = datetime.fromisoformat(last_active_str)
            days_inactive = (datetime.utcnow() - last_dt).days
        else:
            days_inactive = 0

        # Find the next incomplete topic from the study plan
        plan = db.get_study_plan(self.student_id) or self.study_plan
        completed = set(db.get_completed_topics(self.student_id))
        next_topic = next(
            (entry["topic"] for entry in plan if entry["topic"] not in completed),
            None,
        )

        return tools.generate_nudge(
            student_id=self.student_id,
            subject=profile["subject"],
            days_inactive=days_inactive,
            next_topic=next_topic,
        )

    # ── Simulate inactivity (for demo) ────────────────────────────────────────

    def simulate_inactivity(self, days: int = 3) -> None:
        """
        Backdate the last-active timestamp so the nudge fires.
        Only used by the demo button in the UI.
        """
        db.set_last_active_days_ago(self.student_id, days)

    # ── Progress summary ──────────────────────────────────────────────────────

    def get_progress(self) -> dict:
        """
        Return a summary of the student's progress for the dashboard.

        Returns:
            {
                "quiz_history": [...],
                "completed_topics": [...],
                "total_plan_days": int,
                "completion_pct": float,
            }
        """
        plan = db.get_study_plan(self.student_id) or self.study_plan
        completed = db.get_completed_topics(self.student_id)
        unique_plan_topics = list({e["topic"] for e in plan}) if plan else []
        done_count = sum(1 for t in unique_plan_topics if t in completed)
        total = len(unique_plan_topics)

        return {
            "quiz_history": db.get_quiz_history(self.student_id),
            "completed_topics": completed,
            "total_plan_topics": total,
            "completion_pct": round(done_count / total * 100, 1) if total > 0 else 0.0,
        }
