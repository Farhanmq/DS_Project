import pandas as pd

# === Step 1: Load the result dataset ===
result_df = pd.read_excel('../../provided_data/Result.xlsx')

# === Step 2: Define feature groups ===

# Updated multi-select prefixes (all follow the Qxx.1, Qxx.2 pattern)
multi_prefixes = ['Q83', 'Q85', 'Q87', 'Q90', 'Q93', 'Q19']

# Extract all columns for multi-select questions
multi_cols = [col for col in result_df.columns if any(col.startswith(p + '.') for p in multi_prefixes)]

# Single-select features to keep (manually defined; update if needed)
single_cols = ['Q1', 'Q2', 'Q3', 'Q10', 'Q86', 'Q91', 'Q100', 'Q102', 'Q104']

# Target columns
target_cols = ['Q80', 'Q84']

# === Step 3: Subset the data ===

# Combine selected columns
all_cols = multi_cols + single_cols + target_cols

# Subset the DataFrame
model_data = result_df[all_cols].copy()

# Keep rows where at least one of the target variables is not missing
model_data = model_data[model_data[target_cols].notna().any(axis=1)]

# === Step 4: Save or return cleaned data ===

# Optional: save to CSV
model_data.to_csv("cleaned_result_dataset2.csv", index=False)

# Preview
print(f"Shape of cleaned dataset: {model_data.shape}")
print(model_data.head())
