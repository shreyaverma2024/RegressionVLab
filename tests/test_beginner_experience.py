from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from modules.beginner_guide import BEGINNER_QUIZ
from modules.datasets import DATASETS, load_dataset
from modules.explanations import interpret_metrics
from modules.experiment_guide import CONTROL_LABELS
from modules.quiz_data import POSTTEST


APP = Path(__file__).resolve().parent.parent / "app.py"


def navigate(app, page):
    app.sidebar.radio(key="navigation_section").set_value(page).run()
    assert not app.exception
    return app


def button(app, label):
    return next(item for item in app.button if item.label == label)


def slider(app, original_label):
    return next(item for item in app.sidebar.slider if item.label == CONTROL_LABELS[original_label])


def new_app():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    return app


@pytest.mark.parametrize("test_r2,expected", [(0.92, "92.0%"), (0, "none"), (-0.25, "below zero"), (1, "100.0%")])
def test_actual_metric_interpretation_handles_positive_zero_negative_and_perfect_r2(test_r2, expected):
    result = interpret_metrics(8, 12.5, 0.95, test_r2)
    assert expected in result["r2"]
    assert "12.5000" in result["mse"]
    assert "8.0000" in result["mse"]
    assert result["rmse"] == pytest.approx(np.sqrt(12.5))


def test_metric_warnings_are_qualified_not_definitive_diagnoses():
    result = interpret_metrics(1, 10, 0.99, 0.4)
    assert "Possible overfitting" in result["status"]
    assert "0.5900" in result["gap"]
    assert "test set" in result["gap"]
    limited = interpret_metrics(10, 11, 0.2, 0.25)
    assert "input may not contain enough information" in limited["gap"]
    assert interpret_metrics(0, 0, 1, 1)["rmse"] == 0
    assert interpret_metrics(1, float("nan"), 0.5, 0.4)["rmse"] is None


def test_navigation_keeps_glass_design_and_existing_simulation_plot():
    app = new_app()
    expected = ["Aim", "Beginner Guide", "Theory", "Pre-test", "Procedure", "Simulation", "Playground", "Model Comparison", "Post-test"]
    assert app.sidebar.radio[0].options == expected
    for page in expected:
        navigate(app, page)
        assert any("backdrop-filter: blur" in item.value for item in app.markdown)
        assert any("Small changes." in item.value for item in app.markdown)
    navigate(app, "Simulation")
    assert not app.tabs  # Original single simulation, not the removed mini-lab.
    assert any(item.value == "Fitted Curve Comparison" for item in app.subheader)
    assert len(app.dataframe) == 1


@pytest.mark.parametrize("dataset", list(DATASETS))
def test_dataset_explanation_uses_real_rows_and_stays_compatible_with_current_results(dataset):
    app = navigate(new_app(), "Simulation")
    app.selectbox(key="dataset_selection").select(dataset).run()
    original_results = app.session_state["regression_results"]
    button(app, "Show Dataset Explanation").click().run()
    assert not app.exception
    assert app.session_state["learning_progress"]["dataset"]
    assert app.session_state["regression_results"] is original_results
    table = app.table[0].value
    data = app.session_state["regression_data"]
    np.testing.assert_array_equal(table.iloc[:, 0], data["X"][:5].ravel())
    np.testing.assert_array_equal(table.iloc[:, 1], data["y"][:5])
    assert DATASETS[dataset]["x_label"] in table.columns[0]
    assert DATASETS[dataset]["y_label"] in table.columns[1]
    if dataset != "Synthetic":
        X, y = load_dataset(dataset)
        np.testing.assert_array_equal(data["X"], X)
        np.testing.assert_array_equal(data["y"], y)


def test_dataset_button_does_not_show_stale_explanation_after_switch():
    app = navigate(new_app(), "Simulation")
    button(app, "Show Dataset Explanation").click().run()
    app.selectbox(key="dataset_selection").select("Study Hours vs Exam Score").run()
    assert not app.exception
    assert len(app.table) == 0
    button(app, "Show Dataset Explanation").click().run()
    assert "Study time (hours)" in app.table[0].value.columns[0]


def test_simulation_interpretations_match_each_models_calculated_scores():
    app = navigate(new_app(), "Simulation")
    app.selectbox(key="demonstration_preset").select("Overfitting and regularization").run()
    button(app, "Apply Preset").click().run()
    results = app.session_state["regression_results"]
    for result in results.values():
        expected = interpret_metrics(result["Train MSE"], result["Test MSE"], result["Train R²"], result["Test R²"])
        assert any(expected["r2"] == item.value for item in app.markdown)
        assert any(expected["mse"] == item.value for item in app.markdown)
        assert any(expected["gap"] == item.value for item in app.markdown)
    assert any("Degree and Ridge alpha never change Linear" in item.value for item in app.info)


def test_comparison_without_experiment_is_safe_and_has_navigation():
    app = navigate(new_app(), "Model Comparison")
    assert any("No experiment is available yet" in item.value for item in app.info)
    button(app, "Go to Simulation").click().run()
    assert not app.exception
    assert app.sidebar.radio[0].value == "Simulation"


def test_saved_comparison_does_not_retrain_or_lose_settings_on_return():
    app = navigate(new_app(), "Simulation")
    app.selectbox(key="dataset_selection").select("Salary vs Years of Experience").run()
    slider(app, "Polynomial Degree").set_value(5)
    slider(app, "Ridge Alpha").set_value(10.0)
    slider(app, "Testing Data (%)").set_value(40)
    app.run()
    settings = app.session_state["regression_settings"].copy()
    results = app.session_state["regression_results"]
    navigate(app, "Model Comparison")
    assert app.session_state["regression_results"] is results
    assert app.session_state["regression_settings"] == settings
    ranked = app.dataframe[0].value
    assert ranked["Test MSE"].tolist() == sorted(result["Test MSE"] for result in results.values())
    for row in ranked.to_dict("records"):
        for metric in ("Train MSE", "Test MSE", "Train R²", "Test R²"):
            assert row[metric] == results[row["Model"]][metric]
    button(app, "I've compared the results").click().run()
    assert app.session_state["learning_progress"]["compare"]
    button(app, "Return to Simulation").click().run()
    assert not app.exception
    assert app.session_state["regression_results"] is results
    assert app.selectbox(key="dataset_selection").value == "Salary vs Years of Experience"
    assert slider(app, "Polynomial Degree").value == 5
    assert slider(app, "Ridge Alpha").value == 10.0
    assert slider(app, "Testing Data (%)").value == 40


def test_learning_progress_requires_actions_and_persists_through_navigation():
    app = new_app()
    assert "0 / 5" in app.sidebar.progress[0].text
    navigate(app, "Beginner Guide")
    app.slider(key="basics_hours").set_value(7).run()
    assert any(item.value == "72 / 100" for item in app.metric)
    assert "0 / 5" in app.sidebar.progress[0].text
    button(app, "I've understood the basics").click().run()
    assert "1 / 5" in app.sidebar.progress[0].text
    button(app, "Explore a dataset in Simulation →").click().run()
    assert "1 / 5" in app.sidebar.progress[0].text  # Automatic fitting is not an explicit exercise.
    button(app, "Show Dataset Explanation").click().run()
    assert "2 / 5" in app.sidebar.progress[0].text
    button(app, "Run Experiment").click().run()
    assert "3 / 5" in app.sidebar.progress[0].text
    navigate(app, "Aim")
    navigate(app, "Procedure")
    assert "3 / 5" in app.sidebar.progress[0].text
    navigate(app, "Simulation")
    assert "3 / 5" in app.sidebar.progress[0].text


@pytest.mark.parametrize("page,key,questions", [("Beginner Guide", "beginner", BEGINNER_QUIZ), ("Post-test", "posttest", POSTTEST)])
def test_quiz_requires_all_answers_preserves_results_and_allows_retry(page, key, questions):
    app = navigate(new_app(), page)
    button(app, "Submit Quiz").click().run()
    assert not app.exception
    assert any("Please answer every question" in item.value for item in app.warning)
    assert "quiz_results" not in app.session_state
    for i, question in enumerate(questions):
        app.radio(key=f"{key}_q{i}").set_value(question["answer"])
    button(app, "Submit Quiz").click().run()
    assert not app.exception
    saved = app.session_state["quiz_results"][key]
    assert saved["score"] == len(questions)
    assert app.session_state["learning_progress"]["knowledge"]
    navigate(app, "Aim")
    navigate(app, page)
    assert any(f"Final Score: {len(questions)}/{len(questions)}" == item.value for item in app.subheader)
    assert app.radio(key=f"{key}_q0").value == questions[0]["answer"]
    wrong = next(option for option in questions[0]["options"] if option != questions[0]["answer"])
    app.radio(key=f"{key}_q0").set_value(wrong)
    button(app, "Submit Quiz").click().run()
    assert not app.exception
    assert app.session_state["quiz_results"][key]["score"] == len(questions) - 1
