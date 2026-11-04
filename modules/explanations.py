"""Interpret calculated metrics without changing the regression implementations."""

import math

import pandas as pd
import streamlit as st

from modules.experiment_guide import mark_step, navigate_to


def interpret_metrics(train_mse, test_mse, train_r2, test_r2):
    if not all(math.isfinite(value) for value in (train_mse, test_mse, train_r2, test_r2)):
        return {
            "r2": "An R² or MSE value is not finite. Check the data and predictions before interpreting the fit.",
            "mse": "A reliable error summary is unavailable for this run.",
            "gap": "Do not diagnose fitting behavior from undefined metrics.",
            "status": "Check the data", "severity": "warning", "rmse": None
        }
    if test_r2 < 0:
        r2_text = f"Test R² = {test_r2:.4f}. This is below zero: squared error is worse than always predicting this test set's average. R² can be negative; it is not classification accuracy."
    elif test_r2 == 0:
        r2_text = "Test R² = 0.0000. The model matches the squared error of always predicting this test set's average. It explains none of this set's variation under this measure."
    else:
        r2_text = f"Test R² = {test_r2:.4f}. The model explains approximately {100 * test_r2:.1f}% of the variation in this test set. That is not a percent of predictions that are correct."
    mse_text = (
        f"Test MSE = {test_mse:.4f}: the average squared prediction error. "
        f"Training MSE = {train_mse:.4f}. Lower MSE is generally better when comparing "
        "models on the same target scale and the same testing observations. Large mistakes carry extra weight."
    )
    gap = train_r2 - test_r2
    if gap > 0.15:
        status, severity = "Possible overfitting · practice exceeds the exam", "warning"
        gap_text = (
            f"Training R² ({train_r2:.4f}) exceeds testing R² ({test_r2:.4f}) by {gap:.4f}. "
            "The model may have learned training details that do not carry over. "
            "A small or unusual test set can also cause this gap."
        )
    elif train_r2 < 0.5 and test_r2 < 0.5:
        status, severity = "Limited fit · much variation remains", "info"
        gap_text = (
            f"Training R² is {train_r2:.4f} and testing R² is {test_r2:.4f}. Both are below 0.5. "
            "The model may be too simple, or the input may not contain enough information. "
            "More complexity cannot recover information that is missing from X."
        )
    elif test_r2 < 0:
        status, severity = "Testing predictions need attention", "warning"
        gap_text = "The testing score is below the average-target reference. Inspect the curve and data rather than relying on the training score alone."
    else:
        status, severity = "The pattern carries over on this split", "success"
        gap_text = (
            f"Training R² is {train_r2:.4f}; testing R² is {test_r2:.4f}. "
            "The drop is not large under this teaching rule. Check whether the actual prediction errors are acceptable for the task too."
        )
    return {"r2": r2_text, "mse": mse_text, "gap": gap_text,
            "status": status, "severity": severity, "rmse": math.sqrt(max(test_mse, 0))}


def render_metric_interpretation(train_mse, test_mse, train_r2, test_r2, target_label):
    interpretation = interpret_metrics(train_mse, test_mse, train_r2, test_r2)
    st.markdown("**What do these numbers mean?**")
    st.write(interpretation["r2"])
    st.write(interpretation["mse"])
    if interpretation["rmse"] is not None:
        st.caption(f"For scale: RMSE = √MSE = {interpretation['rmse']:.4f}, in the original units of {target_label}. Unlike MSE, RMSE is not in squared units.")
    getattr(st, interpretation["severity"])(interpretation["status"])
    st.write(interpretation["gap"])
    st.caption("Teaching heuristics: an R² drop greater than 0.15 flags a possible gap; both R² scores below 0.5 flag limited fit. These are not definitive diagnoses. Noise, missing predictors and the split also matter. All text above uses the actual calculated scores.")


def metrics_frame(results):
    return pd.DataFrame([
        {"Model": name, **{metric: result[metric] for metric in ("Train MSE", "Test MSE", "Train R²", "Test R²")}}
        for name, result in results.items()
    ])


def render_model_comparison():
    st.header("Model Comparison")
    st.write("Compare all three models from your most recent Simulation experiment. This page reads the saved results; it does not train another model or change your settings.")
    required = ("regression_results", "regression_data", "regression_settings")
    if not all(key in st.session_state for key in required):
        st.info("No experiment is available yet. Open Simulation and run an experiment or apply a preset first.")
        st.button("Go to Simulation", on_click=navigate_to, args=("Simulation",))
        return
    results, data, settings = (st.session_state[key] for key in required)
    st.subheader(f"Dataset: {settings['dataset']}")
    st.caption(f"Saved experiment · {len(data['y_train'])} training / {len(data['y_test'])} testing rows · degree {settings['degree']} · alpha {settings['alpha']:g} · testing {settings['test_size']}% · split seed 42")
    if settings["dataset"] == "Synthetic":
        st.caption(f"Synthetic settings · {settings['pattern']} · {settings['n_samples']} samples · noise {settings['noise']}")
    frame = metrics_frame(results)
    ranked = frame.sort_values("Test MSE", kind="stable").reset_index(drop=True)
    best_mse = float(ranked.loc[0, "Test MSE"])
    tied = frame.loc[(frame["Test MSE"] - best_mse).abs() <= max(1e-9, abs(best_mse) * 1e-9), "Model"].tolist()
    st.success(f"Lowest Test MSE on this split: {', '.join(tied)} ({best_mse:.4f}).")
    st.caption("A winner on one split is not a universal winner. Repeated tuning on this test set also makes it less independent; use validation or cross-validation for model selection in a real project.")
    st.dataframe(ranked.style.format({metric: "{:.4f}" for metric in ("Train MSE", "Test MSE", "Train R²", "Test R²")}), hide_index=True, use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.markdown("**MSE ↓ · smaller squared errors**")
        st.bar_chart(frame.set_index("Model")[["Train MSE", "Test MSE"]], stack=False)
    with right:
        st.markdown("**R² ↑ · more variation explained**")
        st.bar_chart(frame.set_index("Model")[["Train R²", "Test R²"]], stack=False)
    for name, result in results.items():
        with st.expander(f"Understand {name}"):
            render_metric_interpretation(result["Train MSE"], result["Test MSE"], result["Train R²"], result["Test R²"], data["y_label"])
    st.info("Degree changes Polynomial and Ridge, not Linear. Alpha changes Ridge only. Noise, samples, datasets and the split can affect all models. All four metrics can change for an affected model; none is guaranteed to improve.")
    if st.button("I've compared the results", key="complete_comparison"):
        mark_step("compare")
        st.success("Comparison marked complete. Try an independent challenge in Procedure or take the Post-test.")
    st.button("Return to Simulation", on_click=navigate_to, args=("Simulation",))
    st.download_button("Download comparison as CSV", ranked.to_csv(index=False).encode("utf-8"), "regression_comparison.csv", "text/csv")
