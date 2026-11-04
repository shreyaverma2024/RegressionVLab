"""Optional starting points for comparing the lab's regression models."""


DEMONSTRATION_PRESETS = {
    "Linear relationship": {
        "description": (
            "A linear model is adequate for this straight trend. Compare it "
            "with the more flexible polynomial and Ridge models."
        ),
        "settings": {
            "dataset": "Synthetic",
            "pattern": "Linear",
            "n_samples": 200,
            "noise": 5,
            "degree": 3,
            "alpha": 1.0,
            "test_size": 20
        }
    },
    "Curved relationship": {
        "description": (
            "A degree-2 polynomial can capture the curve that a straight line "
            "misses. Compare training and testing errors."
        ),
        "settings": {
            "dataset": "Synthetic",
            "pattern": "Curved",
            "n_samples": 200,
            "noise": 5,
            "degree": 2,
            "alpha": 1.0,
            "test_size": 20
        }
    },
    "Overfitting and regularization": {
        "description": (
            "A high-degree polynomial can overfit noisy, limited data. Ridge "
            "penalizes coefficients and may generalize better; compare training "
            "and testing errors rather than assuming a fixed ranking."
        ),
        "settings": {
            "dataset": "Synthetic",
            "pattern": "Curved",
            "n_samples": 50,
            "noise": 50,
            "degree": 12,
            "alpha": 10.0,
            "test_size": 40
        }
    }
}
