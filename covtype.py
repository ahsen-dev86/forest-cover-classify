# covtype.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")
import joblib

print("✅ Loading data...")
df = pd.read_csv("covtype.data"    , header=None)

print("Shape:", df.shape)

# FIX: shift labels from 1–7 → 0–6 for XGBoost compatibility
df[54] = df[54] - 1
print(df[54].value_counts())

# Features & Target
X = df.drop(54, axis=1)
y = df[54]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\n✅ Data Split Done")
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# ------------------ Random Forest ------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

print("\n🌲 RandomForest Accuracy:", rf.score(X_test, y_test))
print(classification_report(y_test, rf_preds))

# Confusion Matrix RF
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, rf_preds), annot=True, fmt="d", cmap="Blues")
plt.title("RandomForest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("rf_confusion.png")
plt.close()

# Feature Importance RF
importances = rf.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)[:15]
plt.figure(figsize=(8,6))
sns.barplot(x=feat_imp.values, y=feat_imp.index)
plt.title("Top 15 Important Features (RF)")
plt.savefig("rf_feature_importance.png")
plt.close()

# ------------------ XGBoost ------------------
xgb = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", n_jobs=-1, random_state=42)
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)

print("\n🚀 XGBoost Accuracy:", xgb.score(X_test, y_test))
print(classification_report(y_test, xgb_preds))

# Confusion Matrix XGB
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, xgb_preds), annot=True, fmt="d", cmap="Greens")
plt.title("XGBoost Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("xgb_confusion.png")
plt.close()

# Feature Importance XGB
xgb_importances = xgb.feature_importances_
feat_imp_xgb = pd.Series(xgb_importances, index=X.columns).sort_values(ascending=False)[:15]
plt.figure(figsize=(8,6))
sns.barplot(x=feat_imp_xgb.values, y=feat_imp_xgb.index)
plt.title("Top 15 Important Features (XGBoost)")
plt.savefig("xgb_feature_importance.png")
plt.close()

print("\n✅ All tasks completed! Confusion matrices and feature importances saved as PNGs.")
