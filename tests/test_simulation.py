from pathlib import Path

import numpy as np
import pytest
import streamlit as st
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from streamlit.testing.v1 import AppTest

from modules.datasets import DATASETS, load_dataset
from modules.experiment_guide import CONTROL_LABELS


APP_PATH = Path(__file__).resolve().parent.parent / "app.py"
METRICS = ["Train MSE", "Test MSE", "Train R²", "Test R²"]
SYNTHETIC_CONTROLS = {"Data Pattern", CONTROL_LABELS["Number of Samples"], CONTROL_LABELS["Noise Level"]}
EXPECTED_PRESETS = {
    "Linear relationship": {
        "dataset": "Synthetic", "pattern": "Linear", "n_samples": 200,
        "noise": 5, "degree": 3, "alpha": 1.0, "test_size": 20
    },
    "Curved relationship": {
        "dataset": "Synthetic", "pattern": "Curved", "n_samples": 200,
        "noise": 5, "degree": 2, "alpha": 1.0, "test_size": 20
    },
    "Overfitting and regularization": {
        "dataset": "Synthetic", "pattern": "Curved", "n_samples": 50,
        "noise": 50, "degree": 12, "alpha": 10.0, "test_size": 40
    }
}


def sidebar_widget(app, kind, label):
    return next(widget for widget in getattr(app.sidebar, kind) if widget.label == CONTROL_LABELS.get(label, label))


@pytest.fixture
def simulation(monkeypatch):
    plots = []
    original_pyplot = st.pyplot

    def capture_plot(fig, *args, **kwargs):
        ax = fig.axes[0]
        plots.append((ax.get_xlabel(), ax.get_ylabel(), ax.get_title()))
        return original_pyplot(fig, *args, **kwargs)

    monkeypatch.setattr(st, "pyplot", capture_plot)
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    assert not app.exception
    app.sidebar.radio[0].set_value("Simulation").run()
    assert not app.exception
    return app, plots


def assert_current_results(app, plots, name):
    assert not app.exception
    data = app.session_state["regression_data"]
    settings = app.session_state["regression_settings"]
    results = app.session_state["regression_results"]
    info = DATASETS[name]

    assert settings["dataset"] == data["dataset"] == name
    assert data["x_label"] == info["x_label"]
    assert data["y_label"] == info["y_label"]
    assert plots[-1] == (info["x_label"], info["y_label"], data["plot_title"])
    assert name in data["plot_title"]
    assert any(f"{len(data['y'])} samples" in caption.value for caption in app.caption)
    assert any(info["description"] in item.value for item in app.markdown)

    expected_split = train_test_split(
        data["X"], data["y"], test_size=settings["test_size"] / 100, random_state=42
    )
    for key, expected in zip(["X_train", "X_test", "y_train", "y_test"], expected_split):
        np.testing.assert_array_equal(data[key], expected)

    assert list(results) == [
        "Linear Regression", "Polynomial Regression", "Ridge Regression"
    ]
    curve = np.linspace(data["X"].min(), data["X"].max(), 400).reshape(-1, 1)
    for result in results.values():
        model = result["model"]
        assert model.n_features_in_ == 1
        assert np.isfinite(model.predict(data["X"])).all()
        assert np.isfinite(model.predict(curve)).all()
        for split, title in [("train", "Train"), ("test", "Test")]:
            predictions = model.predict(data[f"X_{split}"])
            actual = data[f"y_{split}"]
            assert result[f"{title} MSE"] == pytest.approx(mean_squared_error(actual, predictions))
            assert result[f"{title} R²"] == pytest.approx(r2_score(actual, predictions))
        assert all(np.isfinite(result[metric]) for metric in METRICS)

    displayed = app.dataframe[0].value
    assert displayed["Model"].tolist() == list(results)
    for metric in METRICS:
        np.testing.assert_allclose(displayed[metric], [r[metric] for r in results.values()])


def assert_synthetic_settings(app, plots, expected):
    assert_current_results(app, plots, "Synthetic")
    assert app.session_state["regression_settings"] == expected
    assert app.session_state["synthetic_settings"] == {
        key: expected[key] for key in ["pattern", "n_samples", "noise"]
    }
    controls = [
        ("selectbox", "Dataset", "dataset"),
        ("selectbox", "Data Pattern", "pattern"),
        ("slider", "Number of Samples", "n_samples"),
        ("slider", "Noise Level", "noise"),
        ("slider", "Polynomial Degree", "degree"),
        ("slider", "Ridge Alpha", "alpha"),
        ("slider", "Testing Data (%)", "test_size")
    ]
    for kind, label, setting in controls:
        assert sidebar_widget(app, kind, label).value == expected[setting]

    results = app.session_state["regression_results"]
    for name in ["Polynomial Regression", "Ridge Regression"]:
        assert results[name]["model"].named_steps["poly"].degree == expected["degree"]
    assert results["Ridge Regression"]["model"].named_steps["regressor"].alpha == expected["alpha"]

    rng = np.random.default_rng(42)
    X = rng.uniform(-3, 3, expected["n_samples"])
    if expected["pattern"] == "Linear":
        y = 4 * X + 5
    elif expected["pattern"] == "Curved":
        y = 2 * X ** 2 + 3 * X + 5
    else:
        y = 8 * np.sin(X) + 2 * X + 5
    y += rng.normal(0, expected["noise"] / 10, expected["n_samples"])
    np.testing.assert_array_equal(app.session_state["regression_data"]["X"], X.reshape(-1, 1))
    np.testing.assert_array_equal(app.session_state["regression_data"]["y"], y)


@pytest.mark.parametrize("starting_dataset", list(DATASETS))
@pytest.mark.parametrize("preset_name,expected", EXPECTED_PRESETS.items())
def test_presets_apply_all_settings_from_any_dataset(
    simulation, starting_dataset, preset_name, expected
):
    app, plots = simulation
    sidebar_widget(app, "selectbox", "Dataset").select(starting_dataset).run()
    previous_results = app.session_state["regression_results"]
    previous_data = app.session_state["regression_data"]
    previous_settings = app.session_state["regression_settings"].copy()
    previous_synthetic = app.session_state["synthetic_settings"].copy()

    # Selecting only previews the case; even a real dataset must stay untouched.
    sidebar_widget(app, "selectbox", "Demonstration Preset").select(preset_name).run()
    assert not app.exception
    assert app.session_state["regression_results"] is previous_results
    assert app.session_state["regression_data"] is previous_data
    assert app.session_state["regression_settings"] == previous_settings
    assert app.session_state["synthetic_settings"] == previous_synthetic
    assert sidebar_widget(app, "selectbox", "Dataset").value == starting_dataset

    sidebar_widget(app, "button", "Apply Preset").click().run()
    assert_synthetic_settings(app, plots, expected)
    assert app.session_state["regression_results"] is not previous_results
    assert app.session_state["regression_data"] is not previous_data


@pytest.mark.parametrize("preset_name,expected", EXPECTED_PRESETS.items())
def test_preset_manual_edits_persist_until_explicit_reapplication(
    simulation, preset_name, expected
):
    app, plots = simulation
    sidebar_widget(app, "selectbox", "Demonstration Preset").select(preset_name).run()
    sidebar_widget(app, "button", "Apply Preset").click().run()
    assert_synthetic_settings(app, plots, expected)
    preset_results = app.session_state["regression_results"]

    edited = {
        "dataset": "Synthetic", "pattern": "S-shaped", "n_samples": 160,
        "noise": 17, "degree": 4, "alpha": 2.0, "test_size": 30
    }
    sidebar_widget(app, "selectbox", "Data Pattern").select(edited["pattern"])
    for label, setting in [
        ("Number of Samples", "n_samples"), ("Noise Level", "noise"),
        ("Polynomial Degree", "degree"), ("Ridge Alpha", "alpha"),
        ("Testing Data (%)", "test_size")
    ]:
        sidebar_widget(app, "slider", label).set_value(edited[setting])
    app.run()
    assert_synthetic_settings(app, plots, edited)
    edited_results = app.session_state["regression_results"]
    assert edited_results is not preset_results

    app.run()
    assert_synthetic_settings(app, plots, edited)
    assert app.session_state["regression_results"] is edited_results
    other_preset = next(name for name in EXPECTED_PRESETS if name != preset_name)
    sidebar_widget(app, "selectbox", "Demonstration Preset").select(other_preset).run()
    assert_synthetic_settings(app, plots, edited)
    assert app.session_state["regression_results"] is edited_results
    sidebar_widget(app, "selectbox", "Demonstration Preset").select(preset_name).run()
    assert_synthetic_settings(app, plots, edited)
    assert app.session_state["regression_results"] is edited_results

    # The selected preset must not override the existing synthetic restoration.
    sidebar_widget(app, "selectbox", "Dataset").select("Study Hours vs Exam Score").run()
    assert_current_results(app, plots, "Study Hours vs Exam Score")
    sidebar_widget(app, "selectbox", "Dataset").select("Synthetic").run()
    assert_synthetic_settings(app, plots, edited)
    restored_results = app.session_state["regression_results"]

    sidebar_widget(app, "button", "Apply Preset").click().run()
    assert_synthetic_settings(app, plots, expected)
    assert app.session_state["regression_results"] is not restored_results

    # Applying an unchanged case is also an explicit refresh, like Run Experiment.
    reapplied_results = app.session_state["regression_results"]
    sidebar_widget(app, "button", "Apply Preset").click().run()
    assert_synthetic_settings(app, plots, expected)
    assert app.session_state["regression_results"] is not reapplied_results


def test_switching_refreshes_every_dataset_and_restores_synthetic(simulation):
    app, plots = simulation
    assert sidebar_widget(app, "selectbox", "Dataset").value == "Synthetic"
    assert app.session_state["regression_settings"]["n_samples"] == 120
    assert app.session_state["regression_settings"]["noise"] == 15
    assert app.session_state["regression_settings"]["pattern"] == "Linear"
    assert_current_results(app, plots, "Synthetic")

    sidebar_widget(app, "selectbox", "Data Pattern").select("S-shaped")
    sidebar_widget(app, "slider", "Number of Samples").set_value(180)
    sidebar_widget(app, "slider", "Noise Level").set_value(27)
    app.run()
    assert_current_results(app, plots, "Synthetic")
    synthetic_x = app.session_state["regression_data"]["X"].copy()
    synthetic_y = app.session_state["regression_data"]["y"].copy()
    previous_results = app.session_state["regression_results"]

    for name in list(DATASETS)[1:]:
        # No Run Experiment click: changing the selector must retrain immediately.
        sidebar_widget(app, "selectbox", "Dataset").select(name).run()
        assert_current_results(app, plots, name)
        assert app.session_state["regression_results"] is not previous_results
        previous_results = app.session_state["regression_results"]
        X, y = load_dataset(name)
        data = app.session_state["regression_data"]
        np.testing.assert_array_equal(data["X"], X)
        np.testing.assert_array_equal(data["y"], y)
        for setting in ["pattern", "n_samples", "noise"]:
            assert app.session_state["regression_settings"][setting] is None
        visible = {w.label for w in app.sidebar.selectbox} | {w.label for w in app.sidebar.slider}
        assert not visible & SYNTHETIC_CONTROLS
        assert any(DATASETS[name]["source_url"] in item.value for item in app.markdown)

    sidebar_widget(app, "selectbox", "Dataset").select("Synthetic").run()
    assert_current_results(app, plots, "Synthetic")
    assert app.session_state["regression_results"] is not previous_results
    assert sidebar_widget(app, "selectbox", "Data Pattern").value == "S-shaped"
    assert sidebar_widget(app, "slider", "Number of Samples").value == 180
    assert sidebar_widget(app, "slider", "Noise Level").value == 27
    np.testing.assert_array_equal(app.session_state["regression_data"]["X"], synthetic_x)
    np.testing.assert_array_equal(app.session_state["regression_data"]["y"], synthetic_y)


def test_original_synthetic_patterns_and_controls(simulation):
    app, plots = simulation
    for pattern in ["Linear", "Curved", "S-shaped"]:
        sidebar_widget(app, "selectbox", "Data Pattern").select(pattern)
        sidebar_widget(app, "slider", "Number of Samples").set_value(50)
        sidebar_widget(app, "slider", "Noise Level").set_value(0)
        app.run()
        assert_current_results(app, plots, "Synthetic")
        X = np.random.default_rng(42).uniform(-3, 3, 50)
        if pattern == "Linear":
            y = 4 * X + 5
        elif pattern == "Curved":
            y = 2 * X ** 2 + 3 * X + 5
        else:
            y = 8 * np.sin(X) + 2 * X + 5
        np.testing.assert_array_equal(app.session_state["regression_data"]["X"], X.reshape(-1, 1))
        np.testing.assert_array_equal(app.session_state["regression_data"]["y"], y)


def test_model_and_split_settings_still_refresh_real_data(simulation):
    app, plots = simulation
    name = "Salary vs Years of Experience"
    sidebar_widget(app, "selectbox", "Dataset").select(name).run()
    old_results = app.session_state["regression_results"]
    sidebar_widget(app, "slider", "Testing Data (%)").set_value(40)
    sidebar_widget(app, "slider", "Polynomial Degree").set_value(5)
    sidebar_widget(app, "slider", "Ridge Alpha").set_value(10.0)
    app.run()

    assert_current_results(app, plots, name)
    results = app.session_state["regression_results"]
    assert results is not old_results
    assert len(app.session_state["regression_data"]["y_test"]) == 12
    assert results["Polynomial Regression"]["model"].named_steps["poly"].degree == 5
    assert results["Ridge Regression"]["model"].named_steps["regressor"].alpha == 10.0

    # Unchanged settings reuse results; the explicit button still forces a rerun.
    app.run()
    assert not app.exception
    assert app.session_state["regression_results"] is results
    sidebar_widget(app, "button", "Run Experiment").click().run()
    assert_current_results(app, plots, name)
    assert app.session_state["regression_results"] is not results
