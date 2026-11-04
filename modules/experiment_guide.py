"""Student journey, session-only progress, and guided experiments."""

import streamlit as st


JOURNEY = {
    "basics": ("Learn the basics", "Complete the introduction in Beginner Guide."),
    "dataset": ("Explore a dataset", "Open Show Dataset Explanation in Simulation."),
    "experiment": ("Experiment with models", "Run an experiment, apply a preset, or change a setting in Playground."),
    "compare": ("Compare results", "Review Model Comparison and mark it reviewed."),
    "knowledge": ("Test your knowledge", "Submit the beginner knowledge check or Post-test.")
}

CONTROL_LABELS = {
    "Number of Samples": "How many examples should we use? (Samples)",
    "Noise Level": "How much randomness should the data contain? (Noise)",
    "Polynomial Degree": "How complex should the curve be? (Polynomial Degree)",
    "Ridge Alpha": "How strongly should we restrain coefficients? (Ridge Alpha)",
    "Testing Data (%)": "How much data should be used for testing? (%)"
}

CHALLENGES = {
    "1 · Is a straight line enough?": {
        "preset": "Linear relationship",
        "steps": "Apply the Linear relationship preset. Compare all three curves and Test MSE. Then change degree from 3 to 8 without changing the other controls.",
        "question": "Does a more complex curve necessarily predict the unseen data better? Which model ignores the degree control?",
        "hint": "The true pattern is straight. Linear ignores both degree and alpha. Extra flexibility need not improve test performance."
    },
    "2 · When does bending help?": {
        "preset": "Curved relationship",
        "steps": "Apply the Curved relationship preset. Set degree to 1, note Test MSE, then set it to 2. Leave the dataset, noise, samples and split unchanged.",
        "question": "Which curves and scores changed? Why can degree 2 represent this pattern better than a straight line?",
        "hint": "Curved synthetic data follows a quadratic equation. Degree changes Polynomial and Ridge, but not Linear."
    },
    "3 · Can restraint help a noisy curve?": {
        "preset": "Overfitting and regularization",
        "steps": "Apply the Overfitting and regularization preset. Try alpha 0.01, 1, 10 and 100, keeping everything else fixed. Compare training and testing scores, not just the smoothness of the curve.",
        "question": "Can training error increase while testing error decreases? Which two models ignore alpha?",
        "hint": "Only Ridge uses alpha. A stronger coefficient penalty may improve unseen predictions, but too much restraint can miss the trend. No fixed winner is guaranteed."
    }
}


def mark_step(step):
    if step not in JOURNEY:
        raise ValueError(f"Unknown learning step: {step}")
    progress = dict(st.session_state.get("learning_progress", {}))
    progress[step] = True
    st.session_state["learning_progress"] = progress


def navigate_to(section):
    """Use as a callback so navigation changes before the radio is created."""
    st.session_state["navigation_section"] = section


def render_journey():
    st.subheader("Your learning journey")
    progress = st.session_state.get("learning_progress", {})
    for number, (key, (title, task)) in enumerate(JOURNEY.items(), start=1):
        status = "✓" if progress.get(key) else str(number)
        st.markdown(f"**{status}. {title}** — {task}")
    st.caption("Work at your own pace. Pages are not locked, and a completed step does not mean you must stop exploring.")


def render_progress():
    progress = st.session_state.get("learning_progress", {})
    completed = sum(bool(progress.get(step)) for step in JOURNEY)
    st.subheader("Your learning progress")
    st.progress(completed / len(JOURNEY), text=f"{completed} / {len(JOURNEY)} steps completed")
    for step, (title, _) in JOURNEY.items():
        st.caption(f"{'✓' if progress.get(step) else '○'} {title}")
    st.caption("Progress and quiz results are saved for this browser session only. They are not stored in a database.")


def render_challenges(key_prefix):
    name = st.selectbox("Choose a guided challenge", list(CHALLENGES), key=f"{key_prefix}_challenge")
    challenge = CHALLENGES[name]
    st.markdown(f"**Start with the preset:** {challenge['preset']}")
    st.write(challenge["steps"])
    st.markdown(f"**Think about it:** {challenge['question']}")
    with st.expander("Show a hint"):
        st.write(challenge["hint"])
    st.caption("Apply the preset in Simulation's sidebar yourself, then change one control at a time. These instructions never overwrite your experiment settings.")
