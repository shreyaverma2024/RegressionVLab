
PRETEST = [
    {
        "question": "1. Which model fits a straight-line relationship?",
        "options": [
            "Linear Regression",
            "Polynomial Regression only",
            "Ridge Regression only",
            "K-Means"
        ],
        "answer": "Linear Regression"
    },
    {
        "question": "2. What does a high polynomial degree allow?",
        "options": [
            "Only straight-line fitting",
            "More complex curves",
            "No training",
            "Removal of all noise"
        ],
        "answer": "More complex curves"
    },
    {
        "question": "3. What is overfitting?",
        "options": [
            "Good performance on all unseen data",
            "A model that ignores training data",
            "Learning training details that do not generalize",
            "Always having high training error"
        ],
        "answer": "Learning training details that do not generalize"
    },
    {
        "question": "4. What does Ridge Regression penalize?",
        "options": [
            "The number of rows",
            "Large model coefficients",
            "The test dataset",
            "The target column"
        ],
        "answer": "Large model coefficients"
    },
    {
        "question": "5. Which metric measures squared prediction error?",
        "options": ["R²", "MSE", "Accuracy", "Precision"],
        "answer": "MSE"
    }
]

POSTTEST = [
    {
        "question": "1. Which error should be checked for generalization?",
        "options": [
            "Training MSE only",
            "Testing MSE",
            "Number of features only",
            "Training time only"
        ],
        "answer": "Testing MSE"
    },
    {
        "question": "2. What can happen when polynomial degree is too high?",
        "options": [
            "Overfitting",
            "The model always becomes linear",
            "All errors become zero",
            "Regularization is automatic"
        ],
        "answer": "Overfitting"
    },
    {
        "question": "3. What usually happens when Ridge alpha increases?",
        "options": [
            "Coefficients are penalized more strongly",
            "All coefficients increase",
            "Training data is deleted",
            "Polynomial degree automatically increases"
        ],
        "answer": "Coefficients are penalized more strongly"
    },
    {
        "question": "4. What does a large train-test error gap often suggest?",
        "options": [
            "Overfitting",
            "Perfect generalization",
            "No model was trained",
            "The dataset has no features"
        ],
        "answer": "Overfitting"
    },
    {
        "question": "5. Which R² value is generally better on test data?",
        "options": ["0.95", "0.20", "-0.50", "-2.00"],
        "answer": "0.95"
    }
]
