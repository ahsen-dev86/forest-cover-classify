# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# ----------------- Page Setup -----------------
st.set_page_config(page_title="Forest Cover Classification", layout="wide")
st.title("🌲 Forest Cover Type Prediction")
st.markdown("Predict forest cover type using **Random Forest** and **XGBoost**")

# ----------------- Data Loading -----------------
@st.cache_data
def load_data():
    df = pd.read_csv("covtype.data", header=None)
    df[54] = df[54] - 1  # shift labels (1–7 → 0–6)
    return df

df = load_data()
st.write("✅ Data Loaded. Shape:", df.shape)

# Display class distribution
st.subheader("Class Distribution")
st.bar_chart(df[54].value_counts())

# ----------------- Train/Test Split -----------------
X = df.drop(54, axis=1)
y = df[54]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
st.write("Train size:", X_train.shape, " Test size:", X_test.shape)

# ----------------- Random Forest -----------------
st.subheader("🌲 Random Forest Results")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

rf_acc = rf.score(X_test, y_test)
st.metric("Random Forest Accuracy", f"{rf_acc:.4f}")
st.text("Classification Report")
st.text(classification_report(y_test, rf_preds))

# Confusion Matrix RF
fig, ax = plt.subplots(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, rf_preds), annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_title("Random Forest Confusion Matrix")
st.pyplot(fig)

# Feature Importance RF
importances = rf.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)[:15]
fig, ax = plt.subplots(figsize=(6,4))
sns.barplot(x=feat_imp.values, y=feat_imp.index, ax=ax)
ax.set_title("Top 15 Important Features (RF)")
st.pyplot(fig)

# ----------------- XGBoost -----------------
st.subheader("🚀 XGBoost Results")
xgb = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", n_jobs=-1, random_state=42)
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)

xgb_acc = xgb.score(X_test, y_test)
st.metric("XGBoost Accuracy", f"{xgb_acc:.4f}")
st.text("Classification Report")
st.text(classification_report(y_test, xgb_preds))

# Confusion Matrix XGB
fig, ax = plt.subplots(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, xgb_preds), annot=True, fmt="d", cmap="Greens", ax=ax)
ax.set_title("XGBoost Confusion Matrix")
st.pyplot(fig)

# Feature Importance XGB
xgb_importances = xgb.feature_importances_
feat_imp_xgb = pd.Series(xgb_importances, index=X.columns).sort_values(ascending=False)[:15]
fig, ax = plt.subplots(figsize=(6,4))
sns.barplot(x=feat_imp_xgb.values, y=feat_imp_xgb.index, ax=ax)
ax.set_title("Top 15 Important Features (XGBoost)")
st.pyplot(fig)

st.success("✅ All tasks completed!")
