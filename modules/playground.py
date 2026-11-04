"""A main-page, live playground using the Simulation's existing model functions."""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split

from modules.datasets import DATASETS, load_dataset
from modules.experiment_guide import mark_step
from modules.explanations import metrics_frame, render_metric_interpretation
from modules.presets import DEMONSTRATION_PRESETS


DEFAULTS = {
    "dataset": "Synthetic", "pattern": "Curved", "n_samples": 120,
    "noise": 15, "degree": 2, "alpha": 1.0, "test_size": 20
}
MODEL_COLORS = {
    "Linear Regression": "#c46a2e",
    "Polynomial Regression": "#7851ce",
    "Ridge Regression": "#087e92"
}
PARAMETER_NAMES = {
    "dataset": "Dataset", "pattern": "Data pattern", "n_samples": "Samples",
    "noise": "Noise", "degree": "Polynomial degree", "alpha": "Ridge alpha",
    "test_size": "Testing percentage"
}


def set_playground_settings(settings):
    """Button callback: set only this page's widgets before they are created."""
    st.session_state["playground_preferences"] = dict(settings)
    for name, value in settings.items():
        st.session_state[f"playground_{name}"] = value


def apply_playground_preset():
    name = st.session_state["playground_preset"]
    set_playground_settings(DEMONSTRATION_PRESETS[name]["settings"])


def reset_playground():
    set_playground_settings(DEFAULTS)
    for key in ("playground_experiment", "playground_transition"):
        st.session_state.pop(key, None)


def run_playground(settings, generate_dataset, build_model, evaluate_model):
    """Use the supplied Simulation functions, preserving its data and fit behavior."""
    if settings["dataset"] == "Synthetic":
        X, y = generate_dataset(settings["n_samples"], settings["noise"], settings["pattern"])
    else:
        X, y = load_dataset(settings["dataset"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=settings["test_size"] / 100, random_state=42
    )
    results = {
        name: evaluate_model(build_model(name, settings["degree"], settings["alpha"]),
                             X_train, X_test, y_train, y_test)
        for name in MODEL_COLORS
    }
    return {
        "settings": dict(settings), "results": results,
        "data": dict(X=X, y=y, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    }


def describe_change(before, after):
    changes = [key for key in PARAMETER_NAMES if before[key] != after[key]]
    if not changes:
        return "No parameters changed. The same data and settings produce the same results."
    labels = "; ".join(f"{PARAMETER_NAMES[key]}: {before[key]} → {after[key]}" for key in changes)
    if set(changes) <= {"alpha"}:
        explanation = "Only Ridge responds to alpha. Linear and Polynomial keep exactly the same curves and all four scores."
    elif set(changes) <= {"degree", "alpha"}:
        explanation = "Polynomial degree affects Polynomial and Ridge; alpha affects Ridge only. Linear stays unchanged. More flexibility or restraint does not guarantee a better test score."
    else:
        explanation = "The data, targets or split changed, so all three models can respond. These deltas do not isolate a model improvement: the observations or task may be different."
    return f"{labels}. {explanation}"


def render_playground(generate_dataset, build_model, evaluate_model):
    st.header("Regression Playground")
    st.write("Change a parameter above. Watch the curves and scores update directly below—no Run button needed. Try one change at a time to discover what each control does.")
    st.caption("Your Playground is independent of Simulation and Model Comparison. Settings persist while you navigate within this browser session.")

    preferences = st.session_state.get("playground_preferences", DEFAULTS)
    for name, default in DEFAULTS.items():
        st.session_state.setdefault(f"playground_{name}", preferences.get(name, default))

    with st.container(border=True):
        st.subheader("1 · Make a change")
        preset_column, apply_column, reset_column = st.columns([2, 1, 1])
        with preset_column:
            st.selectbox("Try an example", list(DEMONSTRATION_PRESETS), key="playground_preset")
        with apply_column:
            st.button("Load example", on_click=apply_playground_preset, use_container_width=True)
        with reset_column:
            st.button("Reset Playground", on_click=reset_playground, use_container_width=True)

        dataset = st.selectbox("Dataset", list(DATASETS), key="playground_dataset")
        if dataset == "Synthetic":
            pattern_column, samples_column, noise_column = st.columns(3)
            with pattern_column:
                st.selectbox("Data pattern", ["Linear", "Curved", "S-shaped"], key="playground_pattern")
                st.caption("Change the underlying relationship for every model.")
            with samples_column:
                st.slider("Number of examples", 50, 300, step=10, key="playground_n_samples")
                st.caption("Total observations before splitting into practice and test data.")
            with noise_column:
                st.slider("Noise level", 0, 50, key="playground_noise")
                st.caption("Scatter around the pattern. Standard deviation = this value ÷ 10.")
        else:
            st.caption("Using every original row, with no generated noise. Pattern, samples and noise are synthetic-only controls; your synthetic choices are kept for later.")

        degree_column, alpha_column, split_column = st.columns(3)
        with degree_column:
            st.slider("Polynomial degree", 1, 12, key="playground_degree")
            st.caption("Bending tools · changes Polynomial and Ridge, not Linear.")
        with alpha_column:
            st.slider("Ridge alpha", 0.01, 100.0, step=0.1, key="playground_alpha")
            st.caption("Coefficient restraint · changes Ridge only.")
        with split_column:
            st.slider("Testing data (%)", 10, 40, key="playground_test_size")
            st.caption("Unseen exam · changes the training and testing sets for every model.")

    # Preserve even hidden synthetic controls without allowing widget cleanup to erase them.
    draft = {
        key: st.session_state.get(f"playground_{key}", preferences.get(key, value))
        for key, value in DEFAULTS.items()
    }
    st.session_state["playground_preferences"] = draft
    settings = dict(draft)
    if dataset != "Synthetic":
        settings.update(pattern=None, n_samples=None, noise=None)

    previous = st.session_state.get("playground_experiment")
    if previous is None or previous["settings"] != settings:
        experiment = run_playground(settings, generate_dataset, build_model, evaluate_model)
        if previous is not None:
            st.session_state["playground_transition"] = {
                "before_settings": previous["settings"],
                "description": describe_change(previous["settings"], settings)
            }
            mark_step("experiment")
        st.session_state["playground_experiment"] = experiment
    experiment = st.session_state["playground_experiment"]
    results, data = experiment["results"], experiment["data"]
    info = DATASETS[dataset]
    transition = st.session_state.get("playground_transition")

    st.subheader("2 · See what happens")
    if transition:
        st.info(f"Last parameter change · {transition['description']}")
    else:
        st.info("Start here: change degree from 2 to 1. Does Linear move? Then try changing only alpha—which curve responds?")
    st.caption(f"{dataset} · {len(data['y'])} observations · {len(data['y_train'])} training / {len(data['y_test'])} testing · seed 42")

    # Results immediately follow the controls, not another page or a hidden tab.
    fig, ax = plt.subplots(figsize=(11, 4.2))
    try:
        fig.patch.set_alpha(0)
        ax.set_facecolor((1, 1, 1, 0.35))
        ax.scatter(data["X_train"].ravel(), data["y_train"], color="#a69abf", s=24, alpha=0.65, label="Training · practice")
        ax.scatter(data["X_test"].ravel(), data["y_test"], color="#4b435d", marker="x", s=28, alpha=0.8, label="Testing · unseen")
        curve = np.linspace(data["X"].min(), data["X"].max(), 400).reshape(-1, 1)
        for (name, color), line_style in zip(MODEL_COLORS.items(), ("--", "-", "-.")):
            ax.plot(curve.ravel(), results[name]["model"].predict(curve), color=color, linestyle=line_style, linewidth=2.3, label=name)
        ax.set(xlabel=info["x_label"], ylabel=info["y_label"], title=f"Live results — {dataset}")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
    finally:
        plt.close(fig)

    metric = st.radio("Score to explore", ["Test MSE", "Test R²", "Train MSE", "Train R²"], horizontal=True, key="playground_metric")
    for column, name in zip(st.columns(3), results):
        with column:
            value = results[name][metric]
            st.metric(name, f"{value:.4g}")
    st.caption(f"{'Lower is better ↓' if 'MSE' in metric else 'Higher is better ↑'} · Training is practice; testing is the unseen exam.")
    st.caption("Compare scores on the same dataset and split. Raw MSE across different datasets/target units is not directly comparable.")

    frame = metrics_frame(results)
    st.dataframe(frame.style.format({name: "{:.4f}" for name in frame.columns if name != "Model"}), hide_index=True, use_container_width=True)
    st.subheader("3 · Understand and try again")
    st.write("Can you make the training error smaller but the testing error larger? Try degree 12 on 50 noisy examples. Then increase alpha and watch Ridge alone. Improvement is not guaranteed on every setting.")
    for name, result in results.items():
        with st.expander(f"Why these scores? · {name}"):
            render_metric_interpretation(result["Train MSE"], result["Test MSE"], result["Train R²"], result["Test R²"], info["y_label"])
    with st.expander("About this dataset"):
        st.write(info["description"])
        if dataset != "Synthetic":
            st.markdown(f"Source: [{info['source_name']}]({info['source_url']})")
    st.download_button("Download Playground scores", frame.to_csv(index=False).encode("utf-8"), "playground_scores.csv", "text/csv")
