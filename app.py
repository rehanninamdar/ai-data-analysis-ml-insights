import io
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, mean_absolute_error,
    mean_squared_error, r2_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")
st.set_page_config(page_title="AI Data Insight", page_icon="📊", layout="wide")

st.markdown("""
<style>
.main {background-color: #0b1020;}
[data-testid="stMetric"] {background: #151d33; padding: 16px; border-radius: 12px; border: 1px solid #293653;}
div[data-testid="stDataFrame"] {border: 1px solid #293653; border-radius: 10px;}
.hero {padding: 22px 26px; border-radius: 16px; background: linear-gradient(120deg,#172554,#312e81); margin-bottom: 18px;}
.hero h1 {margin:0;color:#fff;font-size:2.1rem}
.hero p {color:#c7d2fe;margin:8px 0 0 0}
.small-note {color:#94a3b8;font-size:.9rem}
</style>
<div class="hero">
<h1>📊 AI-Powered Data Insight</h1>
<p>Upload a dataset. Automatically explore data, discover patterns, train a machine-learning model, and export a report.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Workspace")
    uploaded = st.file_uploader("Upload CSV dataset", type=["csv"])
    st.caption("Privacy note: analysis runs in this Streamlit session. Do not upload sensitive data to a public deployment.")
    st.divider()
    st.markdown("**Pipeline**")
    st.markdown("1. Data profiling\n2. Data quality checks\n3. Visual exploration\n4. ML training\n5. Insight report")

if uploaded is None:
    st.info("Start by uploading a CSV file, or use the built-in sample dataset below.")
    if st.button("Load sample dataset", type="primary"):
        from sklearn.datasets import load_diabetes
        sample = load_diabetes(as_frame=True)
        df = sample.frame
        df["target"] = sample.target
        st.session_state["sample_df"] = df
    if "sample_df" not in st.session_state:
        st.markdown("### What this project does")
        c1,c2,c3 = st.columns(3)
        c1.markdown("**🔎 Automated EDA**\n\nDataset shape, types, distributions, missing values, correlations and outliers.")
        c2.markdown("**🧠 Machine learning**\n\nChoose a target column and train classification or regression models.")
        c3.markdown("**📝 Explainable results**\n\nModel metrics, feature importance and downloadable cleaned data/report.")
        st.stop()
    df = st.session_state["sample_df"].copy()
else:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

if df.empty:
    st.error("The dataset has no rows.")
    st.stop()

# Normalize column names only for internal safety, preserve user-facing names
df.columns = [str(c).strip() for c in df.columns]
df = df.loc[:, ~pd.Index(df.columns).duplicated()].copy()

st.subheader("Dataset overview")
m1,m2,m3,m4 = st.columns(4)
m1.metric("Rows", f"{df.shape[0]:,}")
m2.metric("Columns", f"{df.shape[1]:,}")
m3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
m4.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🧾 Data preview", "🧹 Data quality", "📈 Explore", "🤖 ML lab", "📝 Insights & export"])

with tab1:
    st.markdown("#### First rows")
    st.dataframe(df.head(100), use_container_width=True)
    st.markdown("#### Column profile")
    profile = pd.DataFrame({
        "Column": df.columns,
        "Data type": [str(t) for t in df.dtypes],
        "Non-null": [int(df[c].notna().sum()) for c in df.columns],
        "Missing": [int(df[c].isna().sum()) for c in df.columns],
        "Unique values": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    st.dataframe(profile, use_container_width=True, hide_index=True)
    st.download_button("Download original dataset", df.to_csv(index=False).encode("utf-8"), "dataset.csv", "text/csv")

with tab2:
    st.markdown("#### Data quality checks")
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        st.success("No missing values detected.")
    else:
        st.warning(f"{len(missing)} columns contain missing values.")
        st.dataframe(missing.rename("Missing values").to_frame(), use_container_width=True)
        fig, ax = plt.subplots(figsize=(9, max(3, len(missing)*0.35)))
        missing.sort_values().plot(kind="barh", ax=ax)
        ax.set_title("Missing values by column")
        st.pyplot(fig)
        plt.close(fig)
    dup = int(df.duplicated().sum())
    st.write(f"**Duplicate rows:** {dup}")
    st.write("**Potential constant columns:**", [c for c in df.columns if df[c].nunique(dropna=False) <= 1] or "None")
    st.markdown("#### Create a cleaned copy")
    drop_duplicates = st.checkbox("Remove duplicate rows", value=True)
    drop_empty_cols = st.checkbox("Remove columns that are completely empty", value=True)
    clean_df = df.copy()
    if drop_duplicates:
        clean_df = clean_df.drop_duplicates()
    if drop_empty_cols:
        clean_df = clean_df.dropna(axis=1, how="all")
    st.caption(f"Cleaned copy: {clean_df.shape[0]} rows × {clean_df.shape[1]} columns. Missing cells are retained here and imputed automatically during ML.")
    st.download_button("Download cleaned CSV", clean_df.to_csv(index=False).encode("utf-8"), "cleaned_dataset.csv", "text/csv")

with tab3:
    st.markdown("#### Explore distributions and relationships")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    all_cols = df.columns.tolist()
    if not numeric_cols:
        st.info("No numeric columns found. Convert numeric-looking text columns in your source CSV to numbers to use numeric charts.")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            chosen_num = st.selectbox("Numeric column distribution", numeric_cols)
            fig, ax = plt.subplots(figsize=(7,4))
            sns.histplot(df[chosen_num].dropna(), kde=True, ax=ax)
            ax.set_title(f"Distribution of {chosen_num}")
            st.pyplot(fig)
            plt.close(fig)
        with col_b:
            chosen_box = st.selectbox("Outlier scan (box plot)", numeric_cols, index=min(1,len(numeric_cols)-1))
            fig, ax = plt.subplots(figsize=(7,4))
            sns.boxplot(x=df[chosen_box], ax=ax)
            ax.set_title(f"Spread and potential outliers: {chosen_box}")
            st.pyplot(fig)
            plt.close(fig)
        if len(numeric_cols) >= 2:
            st.markdown("#### Correlation heatmap")
            corr = df[numeric_cols].corr(numeric_only=True)
            fig, ax = plt.subplots(figsize=(min(12, 1+len(numeric_cols)*0.75), max(4, 1+len(numeric_cols)*0.55)))
            sns.heatmap(corr, cmap="coolwarm", center=0, annot=len(numeric_cols)<=10, fmt=".2f", ax=ax)
            st.pyplot(fig)
            plt.close(fig)
            pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().sort_values(key=abs, ascending=False)
            if len(pairs):
                st.write("**Strongest numeric relationships**")
                st.dataframe(pairs.head(8).rename("Correlation").to_frame(), use_container_width=True)
    cat_cols = [c for c in all_cols if c not in numeric_cols and df[c].nunique(dropna=True) <= 30]
    if cat_cols:
        cat = st.selectbox("Categorical column counts", cat_cols)
        counts = df[cat].fillna("(missing)").astype(str).value_counts().head(20)
        fig, ax = plt.subplots(figsize=(8,4))
        counts.sort_values().plot(kind="barh", ax=ax)
        ax.set_title(f"Most common values in {cat}")
        st.pyplot(fig)
        plt.close(fig)

with tab4:
    st.markdown("#### Train a model on your data")
    st.caption("The target column is what you want to predict. This demo compares several standard scikit-learn models using a holdout test set.")
    target = st.selectbox("Target column", df.columns.tolist(), index=len(df.columns)-1)
    task_guess = "Regression" if pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique() > 12 else "Classification"
    task = st.radio("Prediction task", ["Classification", "Regression"], index=0 if task_guess=="Classification" else 1, horizontal=True)
    test_size = st.slider("Test data percentage", 10, 40, 20, 5)
    model_choice = st.selectbox("Model", ["Compare models automatically", "Random Forest", "Decision Tree", "Linear / Logistic Regression", "Extra Trees"])
    run = st.button("🚀 Train and evaluate", type="primary")
    if run:
        work = df.dropna(subset=[target]).copy()
        if len(work) < 10:
            st.error("At least 10 rows with a non-missing target are recommended.")
        elif work[target].nunique() < 2:
            st.error("The target column needs at least two distinct values.")
        else:
            X = work.drop(columns=[target])
            y = work[target]
            # Remove unusable columns and high-cardinality identifiers
            X = X.drop(columns=[c for c in X.columns if X[c].nunique(dropna=True) <= 1], errors="ignore")
            if X.shape[1] == 0:
                st.error("No usable feature columns remain after removing constant columns.")
            else:
                cat_features = X.select_dtypes(include=["object","category","bool"]).columns.tolist()
                num_features = [c for c in X.columns if c not in cat_features]
                transformers = []
                if num_features:
                    transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_features))
                if cat_features:
                    transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_features))
                preprocess = ColumnTransformer(transformers, remainder="drop")
                if task == "Classification":
                    y = y.astype(str)
                    stratify = y if y.value_counts().min() >= 2 else None
                    if y.nunique() > max(20, int(len(y)*0.5)):
                        st.error("The selected target looks like a unique ID or free-text field. Choose a categorical label target.")
                        st.stop()
                    models = {
                        "Random Forest": RandomForestClassifier(n_estimators=180, random_state=42, class_weight="balanced"),
                        "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
                        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
                        "Extra Trees": ExtraTreesClassifier(n_estimators=180, random_state=42, class_weight="balanced")
                    }
                else:
                    y = pd.to_numeric(y, errors="coerce")
                    valid = y.notna()
                    X, y = X.loc[valid], y.loc[valid]
                    models = {
                        "Random Forest": RandomForestRegressor(n_estimators=180, random_state=42),
                        "Decision Tree": DecisionTreeRegressor(random_state=42),
                        "Linear Regression": LinearRegression(),
                        "Ridge Regression": Ridge(),
                        "Extra Trees": ExtraTreesRegressor(n_estimators=180, random_state=42)
                    }
                    stratify = None
                if len(y) < 10:
                    st.error("Not enough valid target rows to train.")
                else:
                    try:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size/100, random_state=42,
                            stratify=stratify if task == "Classification" else None
                        )
                        allowed = {
                            "Random Forest": ["Random Forest"],
                            "Decision Tree": ["Decision Tree"],
                            "Linear / Logistic Regression": ["Logistic Regression"] if task=="Classification" else ["Linear Regression","Ridge Regression"],
                            "Extra Trees": ["Extra Trees"],
                            "Compare models automatically": list(models.keys())
                        }[model_choice]
                        results, fitted = [], {}
                        for name in allowed:
                            pipe = Pipeline([("preprocess", preprocess), ("model", models[name])])
                            pipe.fit(X_train, y_train)
                            pred = pipe.predict(X_test)
                            fitted[name] = (pipe, pred)
                            if task == "Classification":
                                results.append({"Model":name, "Accuracy":accuracy_score(y_test,pred), "Precision (weighted)":precision_score(y_test,pred,average="weighted",zero_division=0), "Recall (weighted)":recall_score(y_test,pred,average="weighted",zero_division=0), "F1 (weighted)":f1_score(y_test,pred,average="weighted",zero_division=0)})
                            else:
                                results.append({"Model":name, "MAE":mean_absolute_error(y_test,pred), "RMSE":mean_squared_error(y_test,pred)**0.5, "R²":r2_score(y_test,pred)})
                        res = pd.DataFrame(results)
                        score_col = "F1 (weighted)" if task=="Classification" else "R²"
                        best_name = res.sort_values(score_col, ascending=False).iloc[0]["Model"]
                        best_pipe, best_pred = fitted[best_name]
                        st.session_state["ml_result"] = {
                            "results":res, "best_name":best_name, "task":task,
                            "target":target, "y_test":y_test, "pred":best_pred,
                            "pipeline":best_pipe, "X_test":X_test, "feature_names":list(X.columns),
                            "train_rows":len(X_train), "test_rows":len(X_test)
                        }
                        st.success(f"Training complete. Best model by {score_col}: {best_name}")
                    except Exception as e:
                        st.error(f"Training failed: {e}")
    if "ml_result" in st.session_state:
        r = st.session_state["ml_result"]
        st.markdown("#### Model comparison")
        st.dataframe(r["results"].style.format(precision=3), use_container_width=True, hide_index=True)
        st.markdown(f"#### Best model: {r['best_name']}")
        st.write(f"Trained on {r['train_rows']} rows and evaluated on {r['test_rows']} unseen test rows. Target: **{r['target']}**.")
        if r["task"] == "Classification":
            labels = sorted(pd.Series(r["y_test"]).astype(str).unique().tolist())
            pred = pd.Series(r["pred"]).astype(str)
            actual = pd.Series(r["y_test"]).astype(str)
            label_order = sorted(set(actual) | set(pred))
            cm = confusion_matrix(actual, pred, labels=label_order)
            fig, ax = plt.subplots(figsize=(max(5,len(label_order)*0.7), max(4,len(label_order)*0.55)))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_order, yticklabels=label_order, ax=ax)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix")
            st.pyplot(fig); plt.close(fig)
            st.text("Classification report\n"+classification_report(actual,pred,zero_division=0))
        else:
            fig, ax = plt.subplots(figsize=(6,5))
            ax.scatter(r["y_test"], r["pred"], alpha=0.7)
            low = min(float(np.min(r["y_test"])), float(np.min(r["pred"])))
            high = max(float(np.max(r["y_test"])), float(np.max(r["pred"])))
            ax.plot([low,high],[low,high], linestyle="--")
            ax.set_xlabel("Actual"); ax.set_ylabel("Predicted"); ax.set_title("Actual vs predicted")
            st.pyplot(fig); plt.close(fig)
        st.markdown("#### Predictions on the test set")
        pred_df = pd.DataFrame({"Actual":np.asarray(r["y_test"]), "Predicted":np.asarray(r["pred"])})
        st.dataframe(pred_df.head(100), use_container_width=True)
        st.download_button("Download test predictions", pred_df.to_csv(index=False).encode("utf-8"), "predictions.csv", "text/csv")

with tab5:
    st.markdown("#### Automated insight summary")
    numeric = df.select_dtypes(include=np.number)
    insight_lines = [
        f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.",
        f"There are {int(df.isna().sum().sum())} missing cells across {int((df.isna().sum()>0).sum())} columns.",
        f"There are {int(df.duplicated().sum())} duplicate rows.",
        f"{len(numeric.columns)} columns are numeric and {df.shape[1]-len(numeric.columns)} are non-numeric."
    ]
    if len(numeric.columns) >= 2:
        corr = numeric.corr(numeric_only=True)
        pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
        if len(pairs):
            strongest = pairs.loc[pairs.abs().idxmax()]
            colpair = pairs.abs().idxmax()
            insight_lines.append(f"Strongest absolute numeric correlation is between {colpair[0]} and {colpair[1]} (r = {strongest:.3f}). Correlation does not prove causation.")
    for line in insight_lines:
        st.markdown(f"- {line}")
    if "ml_result" in st.session_state:
        r = st.session_state["ml_result"]
        st.markdown(f"- Best trained model: **{r['best_name']}**, for **{r['task'].lower()}**, predicting **{r['target']}**.")
        st.markdown("- Model scores are estimates from a holdout split; performance may change on new data.")
    report = {
        "project": "AI-Powered Automated Data Analysis and Machine Learning Insight Generation System",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": {"rows": int(df.shape[0]), "columns": int(df.shape[1]), "missing_cells": int(df.isna().sum().sum()), "duplicate_rows": int(df.duplicated().sum()), "columns": list(df.columns)},
        "automated_insights": insight_lines
    }
    if "ml_result" in st.session_state:
        r=st.session_state["ml_result"]
        report["machine_learning"]={"task":r["task"],"target":r["target"],"best_model":r["best_name"],"metrics":r["results"].to_dict(orient="records"),"train_rows":r["train_rows"],"test_rows":r["test_rows"]}
    st.download_button("Download insight report (JSON)", json.dumps(report, indent=2, default=str).encode("utf-8"), "insight_report.json", "application/json")
    st.caption("This tool generates statistical and model-based insights. Review results in context; it does not guarantee causal explanations or business outcomes.")
