import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

from modules.datasets import DATASETS, load_dataset
from modules.presets import DEMONSTRATION_PRESETS
from modules.quiz_data import PRETEST, POSTTEST
from modules.presentation import render_theme
from modules.playground import render_playground
from modules.beginner_guide import render_beginner_guide, render_dataset_explanation, render_model_panels
from modules.explanations import render_metric_interpretation, render_model_comparison
from modules.experiment_guide import (
    CONTROL_LABELS, mark_step, navigate_to, render_challenges, render_journey, render_progress
)


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Regression Virtual Lab",
    page_icon="📈",
    layout="wide"
)

render_theme()


# ==================================================
# NAVIGATION
# ==================================================

st.sidebar.title("Virtual Lab Navigation")

section = st.sidebar.radio(
    "Select Section",
    [
        "Aim",
        "Beginner Guide",
        "Theory",
        "Pre-test",
        "Procedure",
        "Simulation",
        "Playground",
        "Model Comparison",
        "Post-test"
    ],
    key="navigation_section"
)


# ==================================================
# QUIZ FUNCTION
# ==================================================

def display_quiz(questions, quiz_key):
    st.subheader("Answer the following questions")
    saved = st.session_state.get("quiz_results", {}).get(quiz_key)

    with st.form(key=f"{quiz_key}_form"):
        answers = {}

        for i, question in enumerate(questions):
            st.markdown(f"**Q{i + 1}. {question['question']}**")

            widget_key = f"{quiz_key}_q{i}"
            if widget_key not in st.session_state:
                st.session_state[widget_key] = saved["answers"][i] if saved else None
            answers[i] = st.radio(
                "Choose your answer:",
                question["options"],
                index=None,
                key=widget_key,
                label_visibility="collapsed"
            )

        submitted = st.form_submit_button("Submit Quiz")

    if submitted:
        if any(answer is None for answer in answers.values()):
            st.warning("Please answer every question before submitting. Your previous submitted result, if any, is unchanged.")
        else:
            saved = {
                "answers": answers,
                "score": sum(answers[i] == question["answer"] for i, question in enumerate(questions)),
                "total": len(questions)
            }
            st.session_state["quiz_results"] = {**st.session_state.get("quiz_results", {}), quiz_key: saved}
            if quiz_key in ("beginner", "posttest"):
                mark_step("knowledge")

    if saved:
        st.markdown("---")
        st.subheader("Last submitted quiz results")

        for i, question in enumerate(questions):
            user_answer = saved["answers"][i]
            correct_answer = question["answer"]

            st.markdown(f"**Q{i + 1}. {question['question']}**")

            if user_answer == correct_answer:
                st.success("Correct!")
            else:
                st.error("Incorrect.")

            st.write(
                "**Your answer:** "
                + (
                    str(user_answer)
                    if user_answer is not None
                    else "Not answered"
                )
            )
            st.write(f"**Correct answer:** {correct_answer}")
            if "why" in question:
                st.caption(question["why"])
            st.markdown("---")

        st.subheader(f"Final Score: {saved['score']}/{saved['total']}")

        if questions:
            st.progress(saved["score"] / saved["total"])
        st.caption("You can change your answers and submit again. Results persist while you navigate within this browser session.")


# ==================================================
# DATA GENERATION
# ==================================================

def apply_demonstration_preset():
    """Set widget state in a callback, before the next run creates widgets."""
    preset = DEMONSTRATION_PRESETS[st.session_state["demonstration_preset"]]
    settings = preset["settings"]
    st.session_state["dataset_selection"] = settings["dataset"]
    synthetic_settings = {
        setting: settings[setting] for setting in ["pattern", "n_samples", "noise"]
    }
    st.session_state["synthetic_settings"] = synthetic_settings
    for setting, value in synthetic_settings.items():
        st.session_state[f"synthetic_{setting}"] = value
    st.session_state["polynomial_degree"] = settings["degree"]
    st.session_state["ridge_alpha"] = settings["alpha"]
    st.session_state["test_size"] = settings["test_size"]
    # Explicit application also refreshes an already matching experiment.
    st.session_state.pop("regression_settings", None)
    st.session_state["preset_was_applied"] = True


def generate_dataset(n_samples, noise, pattern):
    rng = np.random.default_rng(42)
    X = rng.uniform(-3, 3, n_samples)

    if pattern == "Linear":
        y = 4 * X + 5
    elif pattern == "Curved":
        y = 2 * X**2 + 3 * X + 5
    else:
        y = 8 * np.sin(X) + 2 * X + 5

    y = y + rng.normal(0, noise / 10, n_samples)

    return X.reshape(-1, 1), y


# ==================================================
# MODEL CREATION
# ==================================================

def build_model(model_type, degree, alpha):

    if model_type == "Linear Regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression())
        ])

    if model_type == "Polynomial Regression":
        return Pipeline([
            (
                "poly",
                PolynomialFeatures(
                    degree=degree,
                    include_bias=False
                )
            ),
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression())
        ])

    return Pipeline([
        (
            "poly",
            PolynomialFeatures(
                degree=degree,
                include_bias=False
            )
        ),
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alpha=alpha))
    ])


# ==================================================
# MODEL EVALUATION
# ==================================================

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    return {
        "model": model,
        "Train MSE": mean_squared_error(y_train, train_predictions),
        "Test MSE": mean_squared_error(y_test, test_predictions),
        "Train R²": r2_score(y_train, train_predictions),
        "Test R²": r2_score(y_test, test_predictions)
    }


# ==================================================
# SIMPLE MODEL ANALYSIS
# ==================================================

def display_model_analysis(
    model_name,
    train_mse,
    test_mse,
    train_r2,
    test_r2,
    target_label,
    settings
):
    st.markdown(f"#### {model_name} — Model Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Training MSE", f"{train_mse:.4f}")
        st.metric("Training R²", f"{train_r2:.4f}")

    with col2:
        st.metric("Testing MSE", f"{test_mse:.4f}")
        st.metric("Testing R²", f"{test_r2:.4f}")

    render_metric_interpretation(train_mse, test_mse, train_r2, test_r2, target_label)
    if model_name == "Linear Regression":
        st.info("Try next: if the dots form a curve, compare Polynomial at degree 2. Degree and Ridge alpha never change Linear's scores.")
    elif model_name == "Polynomial Regression":
        st.info(f"Current degree: {settings['degree']}. Try a lower degree if the curve chases individual dots. Ridge alpha does not affect this model.")
    else:
        st.info(f"Current degree: {settings['degree']}; alpha: {settings['alpha']:g}. Try changing alpha alone and watch testing error. Stronger penalties do not guarantee a better result.")


# ==================================================
# AIM
# ==================================================

if section == "Aim":

    st.header("Aim")

    st.write(
        "To implement Linear Regression, Polynomial Regression, and "
        "Ridge Regression models and compare their prediction accuracy "
        "and overfitting behavior."
    )

    st.subheader("Objectives")

    st.markdown("""
    - Understand the working of three regression models.
    - Train models using generated and curated offline datasets.
    - Evaluate models using MSE and R² score.
    - Visualize fitted curves.
    - Investigate underfitting and overfitting.
    - Understand the effects of polynomial degree and Ridge regularization.
    """)


    render_journey()
    st.button("Start with the Beginner Guide →", on_click=navigate_to, args=("Beginner Guide",))
    st.button("Open Playground →", on_click=navigate_to, args=("Playground",))


# ==================================================
# BEGINNER GUIDE AND THEORY
# ==================================================

elif section == "Beginner Guide":
    render_beginner_guide(display_quiz)


elif section == "Theory":

    st.header("Theory")

    st.subheader("1. Linear Regression")

    st.write(
        "Linear Regression models the relationship between input "
        "and output using a straight line."
    )

    st.latex(r"y = b_0 + b_1x")

    st.write(
        "It is suitable when the relationship between the input "
        "and output is approximately linear."
    )

    st.subheader("2. Polynomial Regression")

    st.write(
        "Polynomial Regression adds powers of the input to represent "
        "curved relationships. Think of degree as the available bending "
        "tools: degree 1 is a line and degree 2 adds x²."
    )

    st.latex(r"y = b_0 + b_1x + b_2x^2 + \cdots + b_dx^d")

    st.write(
        "Higher polynomial degrees allow more complex curves, but an "
        "unnecessarily high degree can cause overfitting."
    )

    st.subheader("3. Ridge Regression")

    st.write(
        "Ridge Regression adds an L2 penalty to the squared-error loss "
        "to discourage large coefficients."
    )

    st.latex(
        r"\text{Loss} = \sum_{i=1}^{n}(y_i-\hat{y}_i)^2"
        r" + \alpha\sum_{j=1}^{p}w_j^2"
    )

    st.write(
        "The regularization parameter alpha controls the strength "
        "of the penalty. Larger values apply stronger regularization. "
        "In this lab Ridge also uses polynomial features: degree changes "
        "both Polynomial and Ridge; alpha changes Ridge only. "
        "Neither control changes Linear Regression."
    )
    st.subheader("Visual example · a ruler versus a curve")
    example_x = np.linspace(-3, 3, 61)
    st.line_chart(pd.DataFrame({
        "X": example_x, "Straight line": 2 * example_x + 1,
        "Quadratic curve": example_x**2 + 2 * example_x + 1
    }).set_index("X"))
    st.caption("Illustrative equations, not fitted experiment results. A straight line cannot follow every curved pattern; Ridge can use curved features while penalizing coefficients.")

    st.subheader("4. Evaluation Metrics")

    st.markdown("""
    - **Mean Squared Error (MSE):** Measures squared prediction errors.
      Lower is generally better.
    - **R² Score:** Measures how much variation in the target is
      explained by the model. 1 is perfect; 0 matches predicting that
      evaluation set's average; a negative value is worse than that reference.
      R² is not classification accuracy.
    - **Training vs testing:** Training is the practice sheet; testing is
      the unseen exam. Check testing performance, not training error alone.
    """)

    st.subheader("5. Overfitting and Underfitting")

    st.markdown("""
    - **Overfitting:** The model fits training data well but performs
      considerably worse on unseen data.
    - **Underfitting:** The model fails to capture the underlying pattern
      and performs poorly on both datasets.
    - **Good fit:** The model captures the pattern and performs
      consistently on training and testing data.
    """)


# ==================================================
# PRE-TEST
# ==================================================

elif section == "Pre-test":

    st.header("Pre-test")

    st.write(
        "Check your existing knowledge before starting the regression experiment."
    )

    display_quiz(PRETEST, "pretest")


# ==================================================
# PROCEDURE
# ==================================================

elif section == "Procedure":

    st.header("Procedure")
    render_journey()

    st.markdown("""
    1. New to regression? Start with **Beginner Guide** and its short quiz,
       then open the Simulation section.
    2. Optionally choose a **Demonstration Preset** in **Experiment Settings**
       and click **Apply Preset** for a linear, curved, or overfitting example.
       Selecting a preset alone changes nothing; after applying, you can tweak
       all settings manually or apply the preset again to restore its values.
    3. Choose a dataset in **Experiment Settings**: Synthetic, Salary vs
       Years of Experience, Study Hours vs Exam Score, or Diabetes BMI vs
       Disease Progression. Click **Show Dataset Explanation** to inspect
       actual rows, identify X and Y, and read the units and source.
    4. For **Synthetic** only, select Linear, Curved, or S-shaped and set
       the number of samples and noise level. Real datasets use all their
       original rows without added noise or generated samples.
    5. Open the model explanation panels, then select a polynomial degree.
    6. Set the Ridge regularization parameter alpha.
    7. Choose the testing-data percentage.
    8. Results update automatically when the dataset or settings change.
       Click **Run Experiment** to rerun the current experiment.
    9. Inspect the fitted curves and evaluation metrics.
    10. Read the automatic score interpretations. Open **Model Comparison**
        to compare training and testing errors on the same saved split.
    11. Repeat with different settings and datasets. Diabetes uses only
        standardized BMI to predict quantitative disease progression one
        year after baseline; BMI alone is not expected to give a strong fit.
    12. Try an independent challenge below, then take the Post-test.
    13. For free exploration, open **Playground**: all controls appear above
        the live results. Change one parameter and watch the curves and
        score deltas below. This never overwrites your Simulation experiment.
    """)
    st.subheader("Guided experiments and independent challenges")
    render_challenges("procedure")


# ==================================================
# SIMULATION
# ==================================================

elif section == "Simulation":

    st.header("Interactive Regression Simulation")

    st.write(
        "Adjust parameters to observe fitted curves, evaluation metrics, "
        "and model analysis."
    )
    st.info("New here? Follow Parts A–D below. Results update automatically; Run Experiment explicitly reruns the current settings. Use Beginner Guide for X, Y and prediction basics.")

    with st.sidebar:

        st.subheader("Experiment Settings")
        st.caption("Part C · Choose one change at a time, then inspect the curve and testing scores.")
        # Widget values are cleaned up by Streamlit when navigating to another page.
        # Restore the last experiment rather than silently replacing it with defaults.
        last_settings = st.session_state.get("regression_settings", {})
        for key, value in {
            "dataset_selection": last_settings.get("dataset", "Synthetic"),
            "polynomial_degree": last_settings.get("degree", 3),
            "ridge_alpha": last_settings.get("alpha", 1.0),
            "test_size": last_settings.get("test_size", 20)
        }.items():
            st.session_state.setdefault(key, value)

        preset_name = st.selectbox(
            "Demonstration Preset",
            list(DEMONSTRATION_PRESETS),
            key="demonstration_preset"
        )
        st.caption(DEMONSTRATION_PRESETS[preset_name]["description"])
        st.caption(
            "Optional: click Apply Preset to load this case, then tweak any "
            "setting manually. Selecting a preset alone does not change settings."
        )
        st.button("Apply Preset", on_click=apply_demonstration_preset)

        dataset_name = st.selectbox(
            "Dataset",
            list(DATASETS),
            key="dataset_selection",
            help="All models use the chosen dataset. Curated datasets load all original rows offline, without generated noise."
        )

        pattern = n_samples = noise = None

        if dataset_name == "Synthetic":
            # Keep these preferences even while their widgets are hidden.
            synthetic_settings = st.session_state.get(
                "synthetic_settings",
                {"pattern": "Linear", "n_samples": 120, "noise": 15}
            )
            for setting, value in synthetic_settings.items():
                widget_key = f"synthetic_{setting}"
                if widget_key not in st.session_state:
                    st.session_state[widget_key] = value

            patterns = ["Linear", "Curved", "S-shaped"]

            pattern = st.selectbox(
                "Data Pattern",
                patterns,
                key="synthetic_pattern"
            )

            n_samples = st.slider(
                CONTROL_LABELS["Number of Samples"],
                min_value=50,
                max_value=300,
                step=10,
                key="synthetic_n_samples",
                help="Total examples before splitting, not just training examples. Affects all three models."
            )
            st.caption("More examples provide more evidence. Some are held out for testing; scores need not improve on every split.")

            noise = st.slider(
                CONTROL_LABELS["Noise Level"],
                min_value=0,
                max_value=50,
                key="synthetic_noise",
                help="Scatters targets for every model. Noise standard deviation = this value ÷ 10. Synthetic only."
            )
            st.caption("Noise scatters the dots around the true pattern. It can change all four scores for every model.")

            st.session_state["synthetic_settings"] = {
                "pattern": pattern,
                "n_samples": n_samples,
                "noise": noise
            }

        for key, default in {
            "polynomial_degree": 3, "ridge_alpha": 1.0, "test_size": 20
        }.items():
            st.session_state.setdefault(key, default)

        degree = st.slider(
            CONTROL_LABELS["Polynomial Degree"],
            min_value=1,
            max_value=12,
            key="polynomial_degree",
            help="Highest power of X. Changes Polynomial and Ridge, never Linear. Degree 1 is a line; degree 2 adds x²."
        )
        st.caption("Extra flexibility may follow a curve or chase noise. Watch testing error as well as training error.")

        alpha = st.slider(
            CONTROL_LABELS["Ridge Alpha"],
            min_value=0.01,
            max_value=100.0,
            step=0.1,
            key="ridge_alpha",
            help="Only Ridge uses this coefficient penalty. Polynomial and Linear ignore alpha."
        )
        st.caption("Higher alpha restrains large coefficients more strongly. Too much restraint can miss the pattern.")

        test_size = st.slider(
            CONTROL_LABELS["Testing Data (%)"],
            min_value=10,
            max_value=40,
            key="test_size",
            help="The unseen exam. More testing data leaves fewer training observations and changes all models' evaluation sets."
        )
        st.caption("Training = practice; testing = an unseen exam. Changing this split can change all four scores for every model.")

        run_button = st.button(
            "Run Experiment",
            type="primary",
            use_container_width=True
        )

    current_settings = {
        "dataset": dataset_name,
        "pattern": pattern,
        "n_samples": n_samples,
        "noise": noise,
        "degree": degree,
        "alpha": alpha,
        "test_size": test_size
    }

    # Recompute when settings change or the button is clicked.
    if (
        run_button
        or "regression_results" not in st.session_state
        or st.session_state.get("regression_settings") != current_settings
    ):

        if dataset_name == "Synthetic":
            X, y = generate_dataset(
                n_samples=n_samples,
                noise=noise,
                pattern=pattern
            )
            plot_title = f"Regression Models — {pattern} Synthetic Dataset"
        else:
            X, y = load_dataset(dataset_name)
            plot_title = f"Regression Models — {dataset_name}"

        dataset_info = DATASETS[dataset_name]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size / 100,
            random_state=42
        )

        model_names = [
            "Linear Regression",
            "Polynomial Regression",
            "Ridge Regression"
        ]

        results = {}

        for model_name in model_names:

            model = build_model(
                model_name,
                degree,
                alpha
            )

            results[model_name] = evaluate_model(
                model,
                X_train,
                X_test,
                y_train,
                y_test
            )

        st.session_state["regression_results"] = results

        st.session_state["regression_data"] = {
            "dataset": dataset_name,
            "x_label": dataset_info["x_label"],
            "y_label": dataset_info["y_label"],
            "plot_title": plot_title,
            "X": X,
            "y": y,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test
        }

        st.session_state["regression_settings"] = current_settings

    preset_was_applied = st.session_state.pop("preset_was_applied", False)
    if run_button or preset_was_applied:
        mark_step("experiment")

    results = st.session_state["regression_results"]
    data = st.session_state["regression_data"]

    dataset_info = DATASETS[dataset_name]
    st.subheader("Part A · Understand your dataset")
    st.subheader(f"Dataset: {dataset_name}")
    st.write(dataset_info["description"])
    st.caption(
        f"{len(data['y'])} samples | "
        f"X: {data['x_label']} | Y: {data['y_label']}"
    )
    if dataset_name != "Synthetic":
        st.caption(
            "All original rows are loaded offline and split into training "
            "and testing data. No samples are generated and no noise is added."
        )
        st.markdown(
            f"Source: [{dataset_info['source_name']}]"
            f"({dataset_info['source_url']})"
        )

    render_dataset_explanation(dataset_name, data)
    render_model_panels()
    with st.expander("Part C · What should I try first?"):
        render_challenges("simulation")

    st.subheader("Part D · Understand your results")
    # --------------------------------------------------
    # FITTED CURVE GRAPH
    # --------------------------------------------------

    st.subheader("Fitted Curve Comparison")

    X_curve = np.linspace(
        data["X"].min(),
        data["X"].max(),
        400
    ).reshape(-1, 1)

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.scatter(
        data["X_train"].ravel(),
        data["y_train"],
        label="Training Data",
        alpha=0.7
    )

    ax.scatter(
        data["X_test"].ravel(),
        data["y_test"],
        label="Testing Data",
        marker="x",
        alpha=0.8
    )

    for model_name, result in results.items():

        predictions = result["model"].predict(X_curve)

        ax.plot(
            X_curve.ravel(),
            predictions,
            label=model_name,
            linewidth=2
        )

    ax.set_xlabel(data["x_label"])
    ax.set_ylabel(data["y_label"])
    ax.set_title(data["plot_title"])
    ax.legend()
    ax.grid(True, alpha=0.25)

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Inspect whether each curve captures the data pattern without "
        "unnecessary bends or missing important trends."
    )

    # --------------------------------------------------
    # METRICS TABLE
    # --------------------------------------------------

    st.subheader("Model Evaluation Metrics")

    metrics_df = pd.DataFrame([
        {
            "Model": model_name,
            "Train MSE": result["Train MSE"],
            "Test MSE": result["Test MSE"],
            "Train R²": result["Train R²"],
            "Test R²": result["Test R²"]
        }
        for model_name, result in results.items()
    ])

    st.dataframe(
        metrics_df.style.format({
            "Train MSE": "{:.4f}",
            "Test MSE": "{:.4f}",
            "Train R²": "{:.4f}",
            "Test R²": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------
    # MODEL ANALYSIS
    # --------------------------------------------------

    st.subheader("Model Analysis")

    for model_name, result in results.items():

        with st.expander(
            f"{model_name} — Analysis",
            expanded=True
        ):

            display_model_analysis(
                model_name=model_name,
                train_mse=result["Train MSE"],
                test_mse=result["Test MSE"],
                train_r2=result["Train R²"],
                test_r2=result["Test R²"],
                target_label=data["y_label"],
                settings=current_settings
            )

    # --------------------------------------------------
    # DOWNLOAD RESULTS
    # --------------------------------------------------

    st.subheader("Download Results")

    csv_data = metrics_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Metrics as CSV",
        data=csv_data,
        file_name="regression_results.csv",
        mime="text/csv"
    )


# ==================================================
# MODEL COMPARISON AND POST-TEST
# ==================================================

elif section == "Playground":
    render_playground(generate_dataset, build_model, evaluate_model)


elif section == "Model Comparison":
    render_model_comparison()


elif section == "Post-test":

    st.header("Post-test")

    st.write(
        "Test your understanding of regression models after completing "
        "the experiment."
    )

    display_quiz(POSTTEST, "posttest")


# ==================================================
# FOOTER
# ==================================================

st.sidebar.markdown("---")
with st.sidebar:
    render_progress()
st.sidebar.caption("Regression Virtual Lab")
