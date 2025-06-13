#Import all the necessory libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import shap
import matplotlib.pyplot as plt


# Load files into dataframe
df = pd.read_csv('../../provided_data/cleaned_result_dataset.csv')

# === Rebalance: Drop 40% of non-switchers (Q80 == 2) ===
switchers_df = df[df['Q80'] == 1]
non_switchers_df = df[df['Q80'] == 2]
non_switchers_reduced = non_switchers_df.sample(frac=0.6, random_state=42)

# Combine and shuffle
rebalanced_df = pd.concat([switchers_df, non_switchers_reduced]).sample(frac=1, random_state=42).reset_index(drop=True)

# === Define target and features ===
y = (rebalanced_df['Q80'] == 1).astype(int)  # 1 = switchers
X = rebalanced_df.drop(columns=['Q80']).fillna(0)

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === Random Forest ===
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    class_weight='balanced',
    random_state=42
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("\n=== Random Forest ===")
print(classification_report(y_test, rf_pred))
print("ROC-AUC (RF):", roc_auc_score(y_test, rf_pred))

# === SHAP for Random Forest ===
rf_explainer = shap.Explainer(rf_model, X_train)
rf_shap_values = rf_explainer(X_test)
shap.plots.bar(rf_shap_values[:, :, 1], max_display=15)



# === Decision Tree ===
dt_model = DecisionTreeClassifier(
    max_depth=12,
    class_weight='balanced',
    random_state=42
)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

print("\n=== Decision Tree ===")
print(classification_report(y_test, dt_pred))
print("ROC-AUC (DT):", roc_auc_score(y_test, dt_pred))

# === SHAP for Decision Tree ===
dt_explainer = shap.Explainer(dt_model, X_train)
dt_shap_values = dt_explainer(X_test)
shap.plots.bar(dt_shap_values[:, :, 1], max_display=15)
