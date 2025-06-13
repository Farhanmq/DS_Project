import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Step 1: Load data
df = pd.read_excel("Mapped_Result.xlsx")
print("✅ File loaded successfully.")

# Step 2: Define target (binary encoding)
df['target'] = df['Q16 - S15: Planned change n6m '].map({
    1: 1, 2: 1, 3: 0, 4: 0, 5: 0
})
df = df.dropna(subset=['target'])
print(f"✅ Target column mapped. Rows retained: {len(df)}")

# Step 3: Define feature categories
demographics = [
    'Q1 - S01: Gender', 'Q2 - S02: Age', 'Q3 - AG: Age Groups',
    'Q4 - S02a: Region', 'Q6 - S04: Personal Income',
    'Q98 - D01: HH size', 'Q99 - D02: Children in household',
    'Q100 - D02a: Kulturelle und ethische Hintergrund',
    'Q101 - D02b: Kulturelle und ethische Hintergrund im Detail',
    'Q102 - D03: Employment status', 'Q103 - D03a: Professional experience',
    'Q104 - D04: Marital status'
]
insurance_awareness = [col for col in df.columns if col.startswith('Q8.')] + [
    'Q9 - S07: Usage', 'Q10 - S07a: Usage detail', 'Q11 - S07_dummy'
]
switch_behavior = [
    'Q12 - S08: Time of consideration change', 'Q13 - S09: Time of consideration change',
    'Q14 - S10: Number of changes ', 'Q15 - S11: Change from PKV to GKV '
]
switch_motives = [col for col in df.columns if col.startswith('Q24.') or col in [
    'Q26 - R01: Moment of truth', 'Q28 - R02: Moment of truth in detail',
    'Q32 - R03a: Moment of truth – Other Main', 'Q33 - R03b: Moment of truth – Other from List'
]]
insurance_features = [col for col in df.columns if col.startswith('Q40.')]
barriers = [col for col in df.columns if col.startswith('Q55.') or col.startswith('Q56.')]
info_channels = [col for col in df.columns if col.startswith('Q47.') or col == 'Q52 - C02: Purchase channel']

# Step 4: Define all features
feature_groups = {
    "Demographics": demographics,
    "Insurance Awareness and Usage": insurance_awareness,
    "Switch Behavior and Motives": switch_behavior + switch_motives,
    "Insurance Service Features": insurance_features,
    "Barriers to Switching": barriers,
    "Information Channels": info_channels
}
all_features = sum(feature_groups.values(), [])

# Step 5: Clean specific scale-based values
columns_to_clean = {
    'Q12 - S08: Time of consideration change': [1],
    'Q13 - S09: Time of consideration change': [1],
    'Q40.2 - Hervorragendes Angebot zusätzlicher Leistungen (...)': [6],
    'Q40.13 - Hervorragender Ruf der Krankenkasse': [6],
    'Q55.11 - Es ist mir nicht wichtig, bei welcher Krankenkasse ich versichert bin': [1],
    'Q55.12 - Mich hat keine andere Krankenkasse überzeugt': [1],
    'Q55.14 - Ich habe/hatte keine Bedenken': [1],
    'Q56.11 - Es ist mir nicht wichtig, bei welcher Krankenkasse ich versichert bin': [8,9,10,11,12,13],
    'Q56.12 - Mich hat keine andere Krankenkasse überzeugt': [8,9,10,11,12,13],
    'Q56.13 - Bürokratischer Aufwand für den Kassenwechsel zu hoch': [8,9,10,11,12,13]
}

print("\n🧹 Cleaning specific unwanted values...")
for col, vals in columns_to_clean.items():
    if col in df.columns:
        before = df[col].isna().sum()
        df.loc[df[col].isin(vals), col] = pd.NA
        after = df[col].isna().sum()
        print(f" - {col}: set {vals} to NaN | {after - before} values affected.")
df = df.fillna(0)
print("✅ All NaN values replaced with 0.")


# Step 6: Prepare features and target
X = df[all_features].select_dtypes(include="number")
y = df["target"]

# Step 7: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Logistic Regression (with scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train_scaled, y_train)
y_pred_log = logreg.predict(X_test_scaled)

# Step 9: Decision Tree (no scaling)
tree = DecisionTreeClassifier(max_depth=4, random_state=42)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)

# Step 10: Evaluation
print("\n=== Logistic Regression Report ===")
print(classification_report(y_test, y_pred_log))

print("Confusion Matrix (Logistic Regression):")
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_log)).plot()
plt.title("Logistic Regression")
plt.show()

print("\n=== Decision Tree Report ===")
print(classification_report(y_test, y_pred_tree))

print("Confusion Matrix (Decision Tree):")
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_tree)).plot()
plt.title("Decision Tree")
plt.show()

# Optional: Visualize the Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(tree, feature_names=X.columns, class_names=["No Switch", "Switch"], filled=True)
plt.title("Decision Tree Structure")
plt.show()