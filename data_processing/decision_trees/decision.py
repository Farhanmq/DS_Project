# Import Libraries & Load Data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import shap

# Load files
result_df = pd.read_csv('../../provided_data/Result.xlsx')
codebook_df = pd.read_csv('../../provided_data/Codebook.xlsx')

# Create the target variable
result_df['Q80'] = pd.to_numeric(result_df['Q80'], errors='coerce')
result_df['Q84'] = pd.to_numeric(result_df['Q84'], errors='coerce')
result_df['switch_target'] = ((result_df['Q80'] == 1) & (result_df['Q84'].isin([4, 5]))).astype(int)

# Use all Q-prefixed columns (excluding Q80, Q84)
question_columns = [col for col in result_df.columns if col.startswith('Q') and col not in ['Q80', 'Q84']]
X = result_df[question_columns].select_dtypes(include=[np.number]).fillna(0)
y = result_df['switch_target']

# ----------------------------------------------------
# Correlation Matrix
correlation_matrix = X.corr()

# Plot the correlation matrix
plt.figure(figsize=(12, 10))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.title("Correlation Matrix of Q-features")
plt.xticks(ticks=np.arange(len(correlation_matrix.columns)), labels=correlation_matrix.columns, rotation=90)
plt.yticks(ticks=np.arange(len(correlation_matrix.columns)), labels=correlation_matrix.columns)
plt.tight_layout()
plt.show()
# ----------------------------------------------------

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a decision tree
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Evaluate
print("Classification Report:")
print(classification_report(y_test, model.predict(X_test)))

# Feature importances
importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nTop Important Features:")
print(importances[importances["Importance"] > 0].head(15))

# Plot
plt.figure(figsize=(10, 6))
plt.barh(importances['Feature'].head(10)[::-1], importances['Importance'].head(10)[::-1])
plt.xlabel("Importance")
plt.title("Top 10 Important Features for Switching Decision")
plt.tight_layout()
plt.show()
