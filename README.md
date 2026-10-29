# AI-Powered Automated Data Analysis and Machine Learning Insight Generation System

A runnable college ML project that accepts a CSV dataset and automatically produces data profiles, data-quality checks, visualizations, model comparisons, evaluation metrics, and downloadable insight reports.

## Features
- CSV upload and dataset preview
- Automatic column types, missing-value counts, duplicates, unique counts
- Cleaned CSV export (duplicate removal and empty-column removal)
- Histograms, box plots, categorical value counts, numeric correlation heatmap
- Classification and regression model training
- Preprocessing pipeline: median/mode imputation, scaling, one-hot encoding
- Model comparison and evaluation metrics
- Confusion matrix or actual-vs-predicted plot
- Downloadable test predictions and JSON insight report
- Built-in sample dataset for a quick demo

## Requirements
- Python 3.10 or newer recommended
- Internet connection for the first-time package installation

## Run on Windows
1. Extract the ZIP.
2. Open the extracted folder in VS Code.
3. Open Terminal → New Terminal.
4. Create a virtual environment (recommended):
   ```powershell
   py -m venv .venv
   .venv\\Scripts\\Activate.ps1
   ```
5. Install packages:
   ```powershell
   python -m pip install -r requirements.txt
   ```
6. Launch the app:
   ```powershell
   streamlit run app.py
   ```
7. A browser tab should open at `http://localhost:8501`.

If PowerShell blocks activation, run the install and launch commands without activating the environment, using `.venv\\Scripts\\python.exe -m pip install -r requirements.txt` and `.venv\\Scripts\\python.exe -m streamlit run app.py`.

## Run on macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

## Quick demo
- Start the app.
- Click **Load sample dataset** on the landing screen (or upload your own CSV).
- Open **Explore** to view distributions and correlations.
- Open **ML lab**, select `target` for the sample dataset, choose **Regression**, then click **Train and evaluate**.
- Open **Insights & export** to download a JSON report.

## How the ML pipeline works
1. Remove rows with missing target values (features are imputed).
2. Split into training and test sets.
3. Impute numeric missing values with the median and categorical missing values with the most frequent value.
4. Standardize numeric features and one-hot encode categorical features.
5. Train selected scikit-learn models and calculate holdout metrics.
6. Display model results and export predictions.

## Important limitations
- This is a local demonstration app, not a production deployment.
- Model choice and scores depend on the dataset, target column, class balance, and data leakage.
- Correlation is not causation. Review results before making real decisions.
- Avoid uploading confidential or personally identifiable data to public deployments.
