```markdown
# Regression Virtual Lab

Regression Virtual Lab is an interactive app for learning how regression models fit data and make predictions. It is built with Streamlit and lets you compare Linear Regression, Polynomial Regression, and Ridge Regression.

Choose a dataset, adjust the experiment settings, and review the fitted curves and scores. Guided lessons and quizzes help you understand the results.

## Features

- Learning sections: Aim, Theory, Procedure, Pre-test, and Post-test
- Synthetic datasets with Linear, Curved, and S-shaped patterns
- Curated datasets for salary, student scores, and diabetes progression
- Interactive comparison of three regression models
- Graphs and training and test scores
- CSV download of model results

## Requirements

- Python 3 and pip
- A web browser

Use a Python version that works with the packages listed in `requirements.txt`.

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/shreyaverma2024/RegressionVLab.git
cd RegressionVLab
pip install -r requirements.txt
```

You can use a virtual environment to keep the project packages separate:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Then install the packages:

```bash
pip install -r requirements.txt
```

## Run the app

From the repository folder, start Streamlit:

```bash
streamlit run app.py
```

If that command is unavailable, try:

```bash
python -m streamlit run app.py
```

Open the local web address shown in the terminal. It is usually:

```text
http://localhost:8501
```

To stop the app, press `Ctrl+C` in the terminal.

## How to use the lab

1. Read **Aim** and **Theory** to learn the purpose and key concepts.
2. Take the **Pre-test** to check what you know.
3. Review the **Procedure** section.
4. Open **Simulation** from the navigation menu.
5. Choose a dataset from the sidebar.
6. Adjust the settings and select **Run Experiment**.
7. Compare the graph, scores, and model analysis.
8. Change settings and run the experiment again to see what changes.
9. Download the results as a CSV file if needed.
10. Take the **Post-test** to review what you learned.

For a clearer comparison, change one setting at a time.

## Datasets

| Dataset | What it contains |
|---|---|
| **Synthetic** | Generated sample data. Choose a Linear, Curved, or S-shaped pattern. |
| **Salary vs Experience** | Years of experience and salary for 30 examples. |
| **Student Study Hours vs Scores** | Study hours and exam scores for 25 examples. |
| **Diabetes BMI vs Disease Progression** | 442 examples using standardized BMI to predict a measure of disease progression. |

The sample count and noise settings apply only to Synthetic data. The curated datasets use their stored rows. The diabetes BMI values are standardized and unitless; they are not raw BMI measurements. See [`data/README.md`](data/README.md) for dataset sources and details.

## Simulation settings

| Setting | What it does |
|---|---|
| **Dataset** | Selects synthetic data or one of the curated datasets. |
| **Data Pattern** | Chooses Linear, Curved, or S-shaped synthetic data. |
| **Number of Samples** | Sets the number of synthetic data points. |
| **Noise Level** | Controls the amount of random variation in synthetic data. |
| **Polynomial Degree** | Controls how complex the Polynomial and Ridge model curves can be. |
| **Ridge Alpha** | Controls the strength of Ridge regularization. A higher value applies a stronger penalty. |
| **Testing Data (%)** | Sets how much data is held aside to evaluate the models. |

## Regression models

### Linear Regression

Linear Regression fits a straight line to the data. It works well when the relationship between the input and target is approximately linear.

```text
y = b₀ + b₁x
```

A straight line may not fit curved patterns well.

### Polynomial Regression

Polynomial Regression can fit curves by adding powers of the input, such as `x²` or `x³`. The polynomial degree controls how flexible the curve can be.

A higher degree can fit more complex patterns, but it can also follow random noise too closely. This is called **overfitting**.

### Ridge Regression

Ridge Regression adds a penalty that discourages very large model coefficients. This can help control overly complex fits.

The **Ridge Alpha** setting controls the penalty strength:

- A larger alpha applies a stronger penalty.
- A penalty that is too strong may make the model too simple.

## Understanding the scores

The app shows scores for both the training data and the test data.

- **Mean Squared Error (MSE):** Measures the size of prediction errors. Lower is generally better when comparing models on the same dataset.
- **R² score:** Describes how much variation in the target is explained by the model. A score closer to 1 is generally better.

Use the graph and both scores together:

- **Possible underfitting:** The model misses the main data pattern and performs poorly on both training and test data.
- **Possible overfitting:** The model performs much better on training data than test data, or its curve bends to follow noise.
- **Potentially good fit:** The curve follows the main pattern and training and test results are reasonably similar.

These are general learning guidelines, not fixed rules. Results depend on the dataset and experiment settings. MSE values also depend on the target scale, so compare models using the same dataset.

## Download results

In **Simulation**, select **Download Metrics as CSV** to save the current model scores. The CSV includes each model’s training and test MSE and R² scores. It does not include the dataset rows or the fitted models.

## Project structure

```text
RegressionVLab/
├── app.py                    Main Streamlit app
├── requirements.txt          Required Python packages
├── data/
│   ├── README.md             Dataset sources and notes
│   ├── Salary_Data.csv       Salary and experience data
│   └── student_scores.csv    Study hours and scores data
├── modules/
│   ├── datasets.py           Dataset information and loading
│   ├── quiz_data.py          Pre-test and post-test questions
│   └── regression_models.py  Data generation and regression models
└── tests/
    ├── test_datasets.py
    └── test_simulation.py
```

## Run the checks

With the required packages and `pytest` installed, run these commands from the repository folder:

```bash
python -m compileall app.py modules
python -m pytest tests/test_datasets.py tests/test_simulation.py
```

## Troubleshooting

- **The `streamlit` command is not found:** Install the requirements or run `python -m streamlit run app.py`.
- **A module cannot be imported:** Run the command from the repository folder.
- **A CSV file cannot be found:** Check that the `data` folder is present.
- **The app does not open automatically:** Open the local address printed in the terminal.

## License

No license file is currently included in the repository. Contact the repository owner before reusing or redistributing the project.
```
