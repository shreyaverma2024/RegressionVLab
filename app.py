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
from modules.quiz_data import PRETEST, POSTTEST


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Regression Virtual Lab",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Regression Virtual Lab")
st.caption(
    "Implement Linear, Polynomial, and Ridge Regression "
    "and compare prediction accuracy and overfitting behavior."
)


# ==================================================
# NAVIGATION
# ==================================================

st.sidebar.title("Virtual Lab Navigation")

section = st.sidebar.radio(
    "Select Section",
    [
        "Aim",
        "Theory",
        "Pre-test",
        "Procedure",
        "Simulation",
        "Post-test"
    ]
)


# ==================================================
# QUIZ FUNCTION
# ==================================================

def display_quiz(questions, quiz_key):
    st.subheader("Answer the following questions")

    with st.form(key=f"{quiz_key}_form"):
        answers = {}

        for i, question in enumerate(questions):
            st.markdown(f"**Q{i + 1}. {question['question']}**")

            answers[i] = st.radio(
                "Choose your answer:",
                question["options"],
                index=None,
                key=f"{quiz_key}_q{i}",
                label_visibility="collapsed"
            )

        submitted = st.form_submit_button("Submit Quiz")

    if submitted:
        score = 0
        st.markdown("---")
        st.subheader("Quiz Results")

        for i, question in enumerate(questions):
            user_answer = answers[i]
            correct_answer = question["answer"]

            st.markdown(f"**Q{i + 1}. {question['question']}**")

            if user_answer == correct_answer:
                score += 1
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
            st.markdown("---")

        st.subheader(f"Final Score: {score}/{len(questions)}")

        if questions:
            st.progress(score / len(questions))


# ==================================================
# DATA GENERATION
# ==================================================

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
    test_r2
):
    st.markdown(f"#### {model_name} — Model Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Training MSE", f"{train_mse:.4f}")
        st.metric("Training R²", f"{train_r2:.4f}")

    with col2:
        st.metric("Testing MSE", f"{test_mse:.4f}")
        st.metric("Testing R²", f"{test_r2:.4f}")

    # Compare training and testing errors
    error_difference = test_mse - train_mse

    # Testing error is considered significantly higher if
    # it exceeds training error by more than 25%.
    error_threshold = 0.25 * max(abs(train_mse), 1e-9)

    st.markdown("---")
    st.subheader("Result Analysis")

    # UNDERFITTING:
    # Both training and testing R² scores are low.
    if train_r2 < 0.50 and test_r2 < 0.50:
        st.warning("⚠️ Underfitting")

        st.write(
            "Both training and testing performance are poor. "
            "The model may be too simple to learn the pattern "
            "in the data."
        )

    # OVERFITTING:
    # Testing error is significantly higher than training error.
    elif error_difference > error_threshold:
        st.error("🔴 Overfitting")

        st.write(
            "The testing error is significantly higher than the "
            "training error. The model may have learned the training "
            "data too closely and may not perform well on unseen data."
        )

    # GOOD FIT:
    # Errors are close and both R² scores are reasonable.
    elif (
        abs(error_difference) <= error_threshold
        and train_r2 >= 0.50
        and test_r2 >= 0.50
    ):
        st.success("✅ Good Fit")

        st.write(
            "Training and testing errors are relatively close, "
            "and both R² scores indicate a reasonable fit. "
            "The model performs consistently on training and testing data."
        )

    # OTHER CASES
    else:
        st.info("ℹ️ Needs Improvement")

        st.write(
            "The model does not clearly match the underfitting, "
            "overfitting, or good-fit conditions. Try changing the "
            "model type, polynomial degree, or Ridge alpha."
        )

    st.caption(
        "These are simplified teaching rules. MSE depends on the "
        "scale of the target values, so consider the R² scores and "
        "fitted curve along with the errors."
    )


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


# ==================================================
# THEORY
# ==================================================

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
        "curved relationships."
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
        "of the penalty. Larger values apply stronger regularization."
    )

    st.subheader("4. Evaluation Metrics")

    st.markdown("""
    - **Mean Squared Error (MSE):** Measures squared prediction errors.
      Lower is generally better.
    - **R² Score:** Measures how much variation in the target is
      explained by the model. Values closer to 1 are generally better.
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

    st.markdown("""
    1. Open the Simulation section.
    2. Choose a dataset in **Experiment Settings**: Synthetic, Salary vs
       Years of Experience, Study Hours vs Exam Score, or Diabetes BMI vs
       Disease Progression. Read its description, units, and source.
    3. For **Synthetic** only, select Linear, Curved, or S-shaped and set
       the number of samples and noise level. Real datasets use all their
       original rows without added noise or generated samples.
    4. Select a polynomial degree.
    5. Set the Ridge regularization parameter alpha.
    6. Choose the testing-data percentage.
    7. Results update automatically when the dataset or settings change.
       Click **Run Experiment** to rerun the current experiment.
    8. Inspect the fitted curves and evaluation metrics.
    9. Compare training and testing errors for each model.
    10. Repeat with different settings and datasets. Diabetes uses only
        standardized BMI to predict quantitative disease progression one
        year after baseline; BMI alone is not expected to give a strong fit.
    """)


# ==================================================
# SIMULATION
# ==================================================

elif section == "Simulation":

    st.header("Interactive Regression Simulation")

    st.write(
        "Adjust parameters to observe fitted curves, evaluation metrics, "
        "and model analysis."
    )

    with st.sidebar:

        st.subheader("Experiment Settings")

        dataset_name = st.selectbox(
            "Dataset",
            list(DATASETS),
            key="dataset_selection"
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
                "Number of Samples",
                min_value=50,
                max_value=300,
                step=10,
                key="synthetic_n_samples"
            )

            noise = st.slider(
                "Noise Level",
                min_value=0,
                max_value=50,
                key="synthetic_noise"
            )

            st.session_state["synthetic_settings"] = {
                "pattern": pattern,
                "n_samples": n_samples,
                "noise": noise
            }

        degree = st.slider(
            "Polynomial Degree",
            min_value=1,
            max_value=12,
            value=3
        )

        alpha = st.slider(
            "Ridge Alpha",
            min_value=0.01,
            max_value=100.0,
            value=1.0,
            step=0.1
        )

        test_size = st.slider(
            "Testing Data (%)",
            min_value=10,
            max_value=40,
            value=20
        )

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

    results = st.session_state["regression_results"]
    data = st.session_state["regression_data"]

    dataset_info = DATASETS[dataset_name]
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
                test_r2=result["Test R²"]
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
# POST-TEST
# ==================================================

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
st.sidebar.caption("Regression Virtual Lab")
