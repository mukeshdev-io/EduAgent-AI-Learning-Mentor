# app.py
# -------
# EduAgent — Streamlit UI
#
# Run with:  streamlit run app.py
#
# This file handles ONLY the UI and navigation.
# All business logic lives in agent.py / tools.py / db.py.
#
# Page flow:
#   Sidebar always shows progress dashboard.
#   Main area shows the current step based on agent.state.

import sys
import os

# Make sure imports resolve whether run from inside or outside the folder
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
from datetime import date, timedelta

import db as dbmod                           # used in render_studying and render_post_quiz
from tools import generate_quiz, score_quiz, evaluate_post_quiz  # used in post-quiz flow

from agent import (
    EduAgent,
    STATE_PROFILE, STATE_DIAGNOSTIC, STATE_GAP_ANALYSIS,
    STATE_PLAN, STATE_RESOURCES, STATE_STUDYING,
    STATE_POST_QUIZ, STATE_COMPLETE,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduAgent — AI Learning Mentor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

DISCLAIMER = (
    "⚠️ **Disclaimer:** AI-generated content (quizzes, study plans, explanations) "
    "may contain errors. Always verify important facts with your textbook or teacher."
)


# ── Session state helpers ──────────────────────────────────────────────────────
# Streamlit re-runs the whole script on every interaction.
# We keep the agent object in st.session_state so it survives re-runs.

def get_agent() -> EduAgent:
    """Return the agent stored in session state, creating one if needed."""
    if "agent" not in st.session_state:
        st.session_state.agent = EduAgent()
    return st.session_state.agent


def reset_session() -> None:
    """Clear the session and start fresh (new student)."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


# ── Sidebar: progress dashboard ───────────────────────────────────────────────

def render_sidebar(agent: EduAgent) -> None:
    with st.sidebar:
        st.title("🎓 EduAgent")
        st.caption("AI Learning Mentor · AICTE–IBM SkillsBuild")
        st.divider()

        profile = agent.profile
        if profile:
            st.markdown(f"**Subject:** {profile['subject']}")
            st.markdown(f"**Exam date:** {profile['exam_date']}")
            st.markdown(f"**Goal:** {profile['goal']}")
            st.divider()

            progress = agent.get_progress()

            # Overall completion bar
            pct = progress["completion_pct"]
            st.markdown(f"**Plan progress: {pct}%**")
            st.progress(int(pct))

            # Completed topics
            if progress["completed_topics"]:
                st.markdown("**✅ Completed topics:**")
                for t in progress["completed_topics"]:
                    st.markdown(f"- {t}")

            # Quiz history chart
            history = progress["quiz_history"]
            if history:
                st.markdown("**📊 Quiz scores over time**")
                df = pd.DataFrame(history)
                df["score_pct"] = (df["score"] * 100).round(1)
                df["label"] = df["topic"] + "\n" + df["attempted_at"].str[:10]
                st.bar_chart(df.set_index("label")["score_pct"])

        else:
            st.info("Complete your profile to see progress here.")

        st.divider()
        st.caption(f"Session ID: `{agent.student_id}`")

        if st.button("🔄 Start New Session", use_container_width=True):
            reset_session()


# ── Step renderers ─────────────────────────────────────────────────────────────

def render_profile(agent: EduAgent) -> None:
    """Step 1: Collect student profile."""
    st.header("👋 Welcome to EduAgent")
    st.markdown(DISCLAIMER)
    st.markdown(
        "EduAgent is your **autonomous AI learning mentor**. "
        "It will assess your knowledge, build a personalised study plan, "
        "find free resources, track your progress, and check in on you."
    )
    st.divider()

    with st.form("profile_form"):
        name = st.text_input("Your name", placeholder="e.g. Priya Sharma")
        subject = st.text_input(
            "Subject to study",
            placeholder="e.g. Machine Learning, Python Programming, Data Structures"
        )
        goal = st.text_input(
            "Your goal",
            placeholder="e.g. Pass university exams, Get an internship, Complete a certification"
        )
        min_date = date.today() + timedelta(days=3)
        exam_date = st.date_input(
            "Exam / deadline date",
            value=date.today() + timedelta(days=30),
            min_value=min_date,
        )
        submitted = st.form_submit_button("🚀 Start Learning Journey", use_container_width=True)

    if submitted:
        if not name.strip() or not subject.strip() or not goal.strip():
            st.error("Please fill in all fields.")
            return
        with st.spinner("Setting up your profile..."):
            agent.set_profile(
                name=name.strip(),
                subject=subject.strip(),
                goal=goal.strip(),
                exam_date=str(exam_date),
            )
        st.rerun()


def render_diagnostic(agent: EduAgent) -> None:
    """Step 2: Generate and display the diagnostic quiz."""
    profile = agent.profile
    st.header(f"🧪 Diagnostic Quiz — {profile['subject']}")
    st.markdown(
        "Answer these questions honestly — there's no penalty for wrong answers. "
        "EduAgent uses your results to identify where to focus your study time."
    )
    st.markdown(DISCLAIMER)

    # Generate quiz only once per session
    if "diagnostic_questions" not in st.session_state:
        with st.spinner("Generating your personalised quiz... (this takes ~10 seconds)"):
            try:
                questions = agent.generate_diagnostic_quiz()
                st.session_state.diagnostic_questions = questions
            except RuntimeError as e:
                st.error(f"Could not generate quiz: {e}")
                return

    questions = st.session_state.diagnostic_questions
    st.divider()

    answers = {}
    with st.form("quiz_form"):
        for i, q in enumerate(questions):
            st.markdown(f"**Q{i+1}. {q['question']}**")
            choice = st.radio(
                label=f"q{i}",
                options=q["options"],
                index=None,
                label_visibility="collapsed",
                key=f"q_{i}",
            )
            # Map full option text back to letter (A/B/C/D)
            if choice:
                answers[str(i)] = choice[0]   # "A) ..." → "A"
            st.markdown("")

        submitted = st.form_submit_button("📝 Submit Quiz", use_container_width=True)

    if submitted:
        unanswered = [i for i in range(len(questions)) if str(i) not in answers]
        if unanswered:
            st.warning(f"Please answer all questions (missing: Q{[u+1 for u in unanswered]}).")
            return
        with st.spinner("Scoring your quiz..."):
            gap = agent.submit_quiz_answers(answers)
            st.session_state.gap_result = gap
        st.rerun()


def render_gap_analysis(agent: EduAgent) -> None:
    """Step 3: Show gap analysis results."""
    gap = st.session_state.get("gap_result") or agent.gap_result
    profile = agent.profile

    st.header("📊 Your Knowledge Gap Analysis")

    col1, col2, col3 = st.columns(3)
    col1.metric("Your Score", f"{gap['correct']}/{gap['total']}")
    col2.metric("Score %", f"{round(gap['score']*100)}%")
    col3.metric("Weak Topics Found", len(gap.get("weak_topics", [])))

    if gap.get("strong_topics"):
        st.success(f"✅ **Strong topics:** {', '.join(gap['strong_topics'])}")
    if gap.get("weak_topics"):
        st.warning(f"📌 **Topics to focus on:** {', '.join(gap['weak_topics'])}")
    else:
        st.info("You seem strong across all topics! We'll create a general review plan.")

    st.divider()
    if st.button("🗓️ Build My Study Plan →", use_container_width=True):
        with st.spinner("Building your personalised study plan..."):
            try:
                plan = agent.build_plan()
                st.session_state.study_plan = plan
            except RuntimeError as e:
                st.error(f"Plan generation failed: {e}")
                return
        st.rerun()


def render_plan(agent: EduAgent) -> None:
    """Step 4: Display the generated study plan."""
    plan = st.session_state.get("study_plan") or agent.study_plan
    st.header("🗓️ Your Personalised Study Plan")
    st.markdown(DISCLAIMER)

    if not plan:
        st.error("No study plan found. Please go back and build one.")
        return

    # Show plan as a table
    df = pd.DataFrame(plan)
    df["Day"] = df["day"].astype(str)
    df["Date"] = df["date"]
    df["Topic"] = df["topic"]
    df["Task"] = df["task"]
    df["Study Time"] = df["duration_mins"].astype(str) + " min"
    st.dataframe(
        df[["Day", "Date", "Topic", "Task", "Study Time"]],
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    if st.button("🔍 Find Free Resources for Each Topic →", use_container_width=True):
        with st.spinner("Searching the web for real learning resources... (may take 20–30 seconds)"):
            try:
                resources = agent.fetch_resources()
                st.session_state.resources = resources
            except RuntimeError as e:
                st.error(f"Resource search failed: {e}")
                return
        st.rerun()


def render_resources(agent: EduAgent) -> None:
    """Step 5: Show curated resources per topic."""
    resources = st.session_state.get("resources") or agent.resources
    plan = st.session_state.get("study_plan") or agent.study_plan

    st.header("📚 Free Learning Resources")
    st.markdown(
        "These links were found via DuckDuckGo search — "
        "**not generated by the AI**, so they are real URLs. "
        "Each was checked to be reachable."
    )
    st.markdown(DISCLAIMER)

    if not resources:
        st.warning("No resources loaded yet. Please rebuild the plan.")
        return

    unique_topics = list({e["topic"] for e in plan})
    for topic in unique_topics:
        with st.expander(f"📖 {topic}", expanded=True):
            topic_resources = resources.get(topic, [])
            if not topic_resources:
                st.write("No resources found — try searching NPTEL manually.")
            for r in topic_resources:
                st.markdown(f"- 🔗 [{r['title']}]({r['url']}) *(via {r['source']})*")

    st.divider()
    if st.button("✅ Start Studying →", use_container_width=True):
        agent._state = "studying"
        st.rerun()


def render_studying(agent: EduAgent) -> None:
    """
    Step 6 (main): The 'studying' dashboard.
    Shows plan, resources, nudge button, and post-topic quiz entry.
    """
    profile = agent.profile
    db_plan = dbmod.get_study_plan(agent.student_id)
    plan = db_plan or st.session_state.get("study_plan") or agent.study_plan
    resources = st.session_state.get("resources") or agent.resources

    st.header(f"📖 Studying: {profile['subject']}")
    st.markdown(DISCLAIMER)

    # ── Nudge section ───────────────────────────────────────────────────────
    with st.expander("🔔 Nudge Simulator (Demo feature)", expanded=False):
        st.markdown(
            "In a real app, this triggers automatically after 3 days of inactivity. "
            "Click the button below to simulate it for the demo."
        )
        if st.button("📅 Simulate 3 days of inactivity → get nudge"):
            agent.simulate_inactivity(days=3)
            with st.spinner("Generating your personalised nudge..."):
                nudge = agent.get_nudge()
            st.info(f"💬 **Your nudge:** {nudge}")

    st.divider()

    if not plan:
        st.warning("No study plan loaded. Please start from Profile.")
        return

    completed = set(dbmod.get_completed_topics(agent.student_id))

    # ── Plan topics with post-quiz buttons ─────────────────────────────────
    unique_topics = list({e["topic"] for e in plan})
    for topic in unique_topics:
        status = "✅" if topic in completed else "📌"
        with st.expander(f"{status} {topic}", expanded=(topic not in completed)):
            # Resources for this topic
            topic_resources = resources.get(topic, [])
            if topic_resources:
                st.markdown("**Free resources:**")
                for r in topic_resources:
                    st.markdown(f"- 🔗 [{r['title']}]({r['url']}) *(via {r['source']})*")
            else:
                st.markdown(
                    f"🔍 [Search NPTEL for '{topic}'](https://nptel.ac.in/search?q={topic.replace(' ', '+')})"
                )

            if topic not in completed:
                st.divider()
                st.markdown("**📝 Ready to test yourself? Take a post-topic quiz.**")
                if st.button(f"Start quiz: {topic}", key=f"start_pq_{topic}"):
                    st.session_state.post_quiz_topic = topic
                    st.session_state.post_quiz_questions = None
                    agent._state = STATE_POST_QUIZ
                    st.rerun()
            else:
                st.success("Topic completed! ✅")


def render_post_quiz(agent: EduAgent) -> None:
    """Step 7: Post-topic quiz for one topic."""
    topic = st.session_state.get("post_quiz_topic", "Unknown topic")
    profile = agent.profile

    st.header(f"🧩 Post-Topic Quiz: {topic}")
    st.markdown(DISCLAIMER)

    # Generate post-quiz questions only once
    if not st.session_state.get("post_quiz_questions"):
        with st.spinner(f"Generating a 3-question quiz on '{topic}'..."):
            try:
                questions = generate_quiz(
                    subject=profile["subject"],
                    topics=[topic],
                    n_questions=3,
                )
                st.session_state.post_quiz_questions = questions
            except RuntimeError as e:
                st.error(f"Could not generate quiz: {e}")
                return

    questions = st.session_state.post_quiz_questions
    answers = {}

    with st.form("post_quiz_form"):
        for i, q in enumerate(questions):
            st.markdown(f"**Q{i+1}. {q['question']}**")
            choice = st.radio(
                label=f"pq{i}",
                options=q["options"],
                index=None,
                label_visibility="collapsed",
                key=f"pq_{i}",
            )
            if choice:
                answers[str(i)] = choice[0]
        submitted = st.form_submit_button("📝 Submit", use_container_width=True)

    if submitted:
        unanswered = [i for i in range(len(questions)) if str(i) not in answers]
        if unanswered:
            st.warning("Please answer all questions.")
            return

        result = score_quiz(questions, answers)
        pre = dbmod.get_latest_score(agent.student_id, topic)
        eval_result = evaluate_post_quiz(pre, result["score"], topic)

        dbmod.save_quiz_attempt(agent.student_id, topic, result["score"])
        if eval_result["passed"]:
            dbmod.mark_topic_complete(agent.student_id, topic)
            st.success(eval_result["message"])
        else:
            st.warning(eval_result["message"])
            if eval_result["suggest_easier"]:
                st.markdown(
                    "💡 **Suggested action:** Try this simpler resource and re-take the quiz:\n"
                    f"🔍 [Search Khan Academy for '{topic}'](https://www.khanacademy.org/search?search_again=1&page_search_query={topic.replace(' ', '+')})"
                )

        if st.button("← Back to Study Dashboard"):
            agent._state = "studying"
            st.session_state.post_quiz_topic = None
            st.session_state.post_quiz_questions = None
            st.rerun()


# ── Main router ────────────────────────────────────────────────────────────────

def main() -> None:
    agent = get_agent()
    render_sidebar(agent)

    state = agent.state

    if state == STATE_PROFILE:
        render_profile(agent)
    elif state == STATE_DIAGNOSTIC:
        render_diagnostic(agent)
    elif state == STATE_GAP_ANALYSIS:
        render_gap_analysis(agent)
    elif state == STATE_PLAN:
        render_plan(agent)
    elif state == STATE_RESOURCES:
        render_resources(agent)
    elif state == STATE_STUDYING:
        render_studying(agent)
    elif state == STATE_POST_QUIZ:
        render_post_quiz(agent)
    else:
        st.header("🎉 You've completed your study plan!")
        st.success(
            "Congratulations — all topics are marked complete. "
            "Good luck on your exam! 🍀"
        )
        if st.button("Start a new subject"):
            reset_session()


if __name__ == "__main__":
    main()
