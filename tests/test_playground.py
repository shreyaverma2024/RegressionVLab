from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from streamlit.testing.v1 import AppTest

from modules.datasets import DATASETS, load_dataset
from modules.playground import DEFAULTS, MODEL_COLORS, describe_change
from modules.presets import DEMONSTRATION_PRESETS


APP = Path(__file__).resolve().parent.parent / "app.py"
METRICS = ("Train MSE", "Test MSE", "Train R²", "Test R²")


@pytest.fixture
def playground():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    app.sidebar.radio[0].set_value("Playground").run()
    assert not app.exception
    return app


def button(app, label):
    return next(item for item in app.button if item.label == label)


def assert_current_scores(app):
    assert not app.exception
    experiment = app.session_state["playground_experiment"]
    results, data, settings = (experiment[key] for key in ("results", "data", "settings"))
    expected_split = train_test_split(data["X"], data["y"], test_size=settings["test_size"] / 100, random_state=42)
    for key, expected in zip(("X_train", "X_test", "y_train", "y_test"), expected_split):
        np.testing.assert_array_equal(data[key], expected)
    for name, result in results.items():
        for split, prefix in (("train", "Train"), ("test", "Test")):
            predictions = result["model"].predict(data[f"X_{split}"])
            assert result[f"{prefix} MSE"] == pytest.approx(mean_squared_error(data[f"y_{split}"], predictions))
            assert result[f"{prefix} R²"] == pytest.approx(r2_score(data[f"y_{split}"], predictions))
        displayed = app.dataframe[0].value.set_index("Model")
        for metric in METRICS:
            assert displayed.loc[name, metric] == result[metric]
            assert np.isfinite(result[metric])
    return experiment


def test_playground_defaults_use_main_page_controls_and_live_results(playground):
    app = playground
    experiment = assert_current_scores(app)
    assert experiment["settings"] == DEFAULTS
    assert len(app.sidebar.slider) == 0
    assert len(app.sidebar.selectbox) == 0
    assert len(app.slider) == 5
    assert not any(item.label == "Run Experiment" for item in app.button)
    assert [item.value for item in app.subheader][:3] == ["1 · Make a change", "2 · See what happens", "3 · Understand and try again"]


@pytest.mark.parametrize("parameter,value", [("degree", 8), ("alpha", 25.0), ("noise", 40), ("n_samples", 50), ("test_size", 40)])
def test_every_numeric_parameter_refreshes_without_run_click(playground, parameter, value):
    app = playground
    before = app.session_state["playground_experiment"]
    app.slider(key=f"playground_{parameter}").set_value(value).run()
    after = assert_current_scores(app)
    assert after is not before
    assert after["settings"][parameter] == value
    assert app.session_state["learning_progress"]["experiment"]
    assert app.session_state["playground_transition"]["before_settings"] == before["settings"]
    if parameter in ("degree", "alpha"):
        for key in before["data"]:
            np.testing.assert_array_equal(before["data"][key], after["data"][key])
        unaffected = ("Linear Regression", "Polynomial Regression") if parameter == "alpha" else ("Linear Regression",)
        for name in unaffected:
            for metric in METRICS:
                assert before["results"][name][metric] == after["results"][name][metric]
        assert any("unchanged" in item.value or "same curves" in item.value for item in app.info)
    if parameter == "n_samples":
        assert len(after["data"]["y"]) == value
    if parameter == "test_size":
        assert len(after["data"]["y_test"]) == 48
    if parameter == "noise":
        np.testing.assert_array_equal(before["data"]["X"], after["data"]["X"])
        assert not np.array_equal(before["data"]["y"], after["data"]["y"])


def test_pattern_switch_updates_targets_and_real_datasets_preserve_every_row(playground):
    app = playground
    app.selectbox(key="playground_pattern").select("S-shaped").run()
    synthetic = assert_current_scores(app)
    assert synthetic["settings"]["pattern"] == "S-shaped"
    for dataset in list(DATASETS)[1:]:
        app.selectbox(key="playground_dataset").select(dataset).run()
        experiment = assert_current_scores(app)
        X, y = load_dataset(dataset)
        np.testing.assert_array_equal(experiment["data"]["X"], X)
        np.testing.assert_array_equal(experiment["data"]["y"], y)
        assert all(experiment["settings"][key] is None for key in ("pattern", "noise", "n_samples"))
        assert {widget.key for widget in app.slider} == {"playground_degree", "playground_alpha", "playground_test_size"}
        assert any(DATASETS[dataset]["source_url"] in item.value for item in app.markdown)
    app.selectbox(key="playground_dataset").select("Synthetic").run()
    restored = assert_current_scores(app)
    assert restored["settings"] == synthetic["settings"]
    for key in ("X", "y"):
        np.testing.assert_array_equal(restored["data"][key], synthetic["data"][key])


@pytest.mark.parametrize("preset_name", list(DEMONSTRATION_PRESETS))
def test_example_exactly_matches_the_existing_simulation(playground, preset_name):
    app = playground
    app.sidebar.radio[0].set_value("Simulation").run()
    app.selectbox(key="demonstration_preset").select(preset_name).run()
    button(app, "Apply Preset").click().run()
    simulation_results = app.session_state["regression_results"]
    simulation_data = app.session_state["regression_data"]
    simulation_settings = app.session_state["regression_settings"].copy()
    app.sidebar.radio[0].set_value("Playground").run()
    before = app.session_state["playground_experiment"]
    app.selectbox(key="playground_preset").select(preset_name).run()
    assert app.session_state["playground_experiment"] is before
    button(app, "Load example").click().run()
    experiment = assert_current_scores(app)
    assert experiment["settings"] == simulation_settings
    for name in MODEL_COLORS:
        for metric in METRICS:
            assert experiment["results"][name][metric] == simulation_results[name][metric]
    for key, values in experiment["data"].items():
        np.testing.assert_array_equal(values, simulation_data[key])
    assert app.session_state["regression_results"] is simulation_results
    assert app.session_state["regression_data"] is simulation_data
    assert app.session_state["regression_settings"] == simulation_settings


def test_metric_clicks_preserve_fit_and_show_only_current_scores(playground):
    app = playground
    app.slider(key="playground_alpha").set_value(15.0).run()
    experiment = app.session_state["playground_experiment"]
    transition = app.session_state["playground_transition"]
    for metric in METRICS:
        app.radio(key="playground_metric").set_value(metric).run()
        assert not app.exception
        assert app.session_state["playground_experiment"] is experiment
        assert app.session_state["playground_transition"] is transition
        cards = {item.label: item for item in app.metric}
        for name in MODEL_COLORS:
            value = experiment["results"][name][metric]
            assert cards[name].value == f"{value:.4g}"
            assert not cards[name].delta


def test_reset_and_navigation_preserve_page_isolation(playground):
    app = playground
    app.slider(key="playground_noise").set_value(40)
    app.slider(key="playground_degree").set_value(7)
    app.run()
    previous = assert_current_scores(app)
    app.sidebar.radio[0].set_value("Theory").run()
    app.sidebar.radio[0].set_value("Playground").run()
    assert app.session_state["playground_experiment"] is previous
    assert app.slider(key="playground_noise").value == 40
    assert app.slider(key="playground_degree").value == 7
    app.selectbox(key="playground_dataset").select("Study Hours vs Exam Score").run()
    app.sidebar.radio[0].set_value("Beginner Guide").run()
    app.sidebar.radio[0].set_value("Playground").run()
    assert app.selectbox(key="playground_dataset").value == "Study Hours vs Exam Score"
    button(app, "Reset Playground").click().run()
    assert_current_scores(app)
    assert app.session_state["playground_experiment"]["settings"] == DEFAULTS
    assert "playground_transition" not in app.session_state
    assert app.slider(key="playground_noise").value == DEFAULTS["noise"]
    assert app.slider(key="playground_degree").value == DEFAULTS["degree"]
    assert "regression_results" not in app.session_state
    assert "regression_settings" not in app.session_state


def test_change_explanation_does_not_claim_improvement_when_the_data_changes():
    assert "Only Ridge" in describe_change(DEFAULTS, {**DEFAULTS, "alpha": 10.0})
    assert "Linear stays unchanged" in describe_change(DEFAULTS, {**DEFAULTS, "degree": 4})
    assert "observations or task may be different" in describe_change(DEFAULTS, {**DEFAULTS, "noise": 30})
    assert "No parameters changed" in describe_change(DEFAULTS, DEFAULTS)
