"""Plain-language concepts and explanations of the actual selected dataset."""

import numpy as np
import pandas as pd
import streamlit as st

from modules.datasets import DATASETS
from modules.experiment_guide import mark_step, navigate_to, render_journey


BEGINNER_QUIZ = [
    {
        "question": "In Study Hours vs Exam Score, what is the input X?",
        "options": ["Study hours", "Exam score", "Prediction error"],
        "answer": "Study hours",
        "why": "X is the information given to the model. Exam score is the target Y we want to predict."
    },
    {
        "question": "Why keep some observations for testing?",
        "options": ["To check predictions on unseen examples", "To make training error zero", "To remove all noise"],
        "answer": "To check predictions on unseen examples",
        "why": "Testing is like an unseen exam. It checks more than memorizing the practice sheet."
    },
    {
        "question": "Which models change when you change Ridge alpha in this lab?",
        "options": ["All three", "Only Ridge", "Only Linear"],
        "answer": "Only Ridge",
        "why": "Only Ridge includes the coefficient penalty. Degree, in contrast, affects both Polynomial and Ridge in this lab."
    }
]

MODEL_EXPLANATIONS = {
    "Linear Regression": (
        "A ruler through the dots", "Finds a straight line connecting input to prediction.",
        "It learns a slope and an intercept from the training data. It can tilt or shift, but cannot bend. Neither polynomial degree nor Ridge alpha changes this model.",
        r"\hat{y} = b_0 + b_1x"
    ),
    "Polynomial Regression": (
        "A curve with more bending tools", "Adds powers of X to represent curved patterns.",
        "Degree 1 is a line. Degree 2 adds x²; degree 3 adds x³. More flexibility can capture a trend or follow noise too closely. Ridge alpha has no effect here.",
        r"\hat{y} = b_0 + b_1x + b_2x^2 + \cdots + b_dx^d"
    ),
    "Ridge Regression": (
        "A curve with restraint", "Discourages large coefficients with a penalty.",
        "Here Ridge uses the same polynomial features as Polynomial Regression. Degree controls its available powers of X; alpha controls the coefficient penalty. More restraint may help unseen predictions, but is not guaranteed to improve them.",
        r"\text{objective} = \sum_i(y_i-\hat{y}_i)^2 + \alpha\sum_j w_j^2"
    )
}

DATASET_QUESTIONS = {
    "Synthetic": "Do the dots follow a straight, curved or S-shaped trend? How much scatter is there around it?",
    "Salary vs Years of Experience": "As years of experience increase, does salary tend to increase? Do all observations lie exactly on a line?",
    "Study Hours vs Exam Score": "Do higher study hours tend to go with higher exam scores? Why might two students studying similar hours score differently?",
    "Diabetes BMI vs Disease Progression": "How much scatter remains at similar BMI values? Could other factors help explain disease progression?"
}


def render_dataset_explanation(dataset_name, data):
    info = DATASETS[dataset_name]
    if st.button("Show Dataset Explanation", key="show_dataset_explanation"):
        st.session_state["explained_dataset"] = dataset_name
        mark_step("dataset")
    if st.session_state.get("explained_dataset") != dataset_name:
        st.caption("New to datasets? Open the explanation to see real rows and identify X and Y.")
        return
    with st.container(border=True):
        st.markdown(f"**X · the input:** {info['x_label']}")
        st.markdown(f"**Y · the target:** {info['y_label']}")
        st.write(f"Our task is to predict {info['y_label']} using {info['x_label']}. A row is one example containing an input and its observed target.")
        preview = pd.DataFrame({
            f"X · {info['x_label']}": data["X"][:5].ravel(),
            f"Y · {info['y_label']}": data["y"][:5]
        })
        st.table(preview)
        st.caption("These are the first five actual rows of your current dataset, not made-up teaching values. The rows shown here are not necessarily the testing rows.")
        st.info(f"Think about it: {DATASET_QUESTIONS[dataset_name]}")
        st.caption("A trend can support prediction, but does not by itself prove that X causes Y. Different rows can have similar inputs and different targets.")


def render_model_panels():
    st.subheader("Part B · Meet the models")
    for name, (metaphor, summary, explanation, equation) in MODEL_EXPLANATIONS.items():
        with st.expander(f"How does this work? · {name}"):
            st.markdown(f"**{metaphor}.** {summary}")
            st.write(explanation)
            st.latex(equation)


def render_beginner_guide(display_quiz):
    st.header("Beginner Guide")
    st.write("No machine-learning background needed. Start with a familiar question: can we estimate an exam score from study hours?")
    render_journey()
    st.subheader("1 · Data is a collection of examples")
    st.write("Each row describes one observation. X is the input you know, such as hours studied. Y is the target you want to predict, such as exam score. Regression predicts a number, not a category.")
    st.subheader("2 · A prediction is an estimate, not a promise")
    hours = st.slider("Try study hours in this teaching example", 0, 10, 4, key="basics_hours")
    prediction = 30 + 6 * hours
    st.metric("Illustrative predicted score", f"{prediction} / 100")
    st.caption(f"Made-up teaching rule: score = 30 + 6 × hours. With {hours} hours, the estimate is {prediction}. This is not the fitted model for the real Study Hours dataset.")
    x = np.arange(11)
    st.line_chart(pd.DataFrame({"Study hours": x, "Illustrative score": 30 + 6 * x}).set_index("Study hours"))
    st.subheader("3 · Training is practice; testing is the unseen exam")
    st.write("The model learns from training examples. Testing examples are held aside to check predictions it did not practice on. An error is the observed target minus the prediction; an error of zero means the prediction matched that observation.")
    st.info("For example: observed score 70, prediction 65 → error 5 → squared error 25. MSE averages these squared errors. A single squared error is not yet the whole dataset's MSE.")
    if st.button("I've understood the basics", key="complete_basics"):
        mark_step("basics")
        st.success("Basics marked complete. Try the short knowledge check, then explore a dataset.")
    st.subheader("Quick knowledge check")
    display_quiz(BEGINNER_QUIZ, "beginner")
    st.button("Explore a dataset in Simulation →", on_click=navigate_to, args=("Simulation",))
