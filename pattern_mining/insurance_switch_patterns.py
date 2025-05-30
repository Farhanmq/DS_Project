import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

df = pd.read_csv('formatted_data/Kundenmonitor_GKV_2023/Band/Question_1.csv')

def prepare_data(df):
    # Convert categorical columns to binary columns
    binary_df = pd.DataFrame()
    
    # Handle 'Wechsel geplant' (planned switch) column
    binary_df['plans_to_switch'] = (df['Wechsel geplant'] == 'Ja').astype(int)
    binary_df['no_switch_planned'] = (df['Wechsel geplant'] == 'Nein').astype(int)
    
    # Handle age groups
    age_mapping = {
        '16-29 Jahre': '16-29',
        '30-39 Jahre': '30-39',
        '40-49 Jahre': '40-49',
        '50-59 Jahre': '50-59',
        '60-69 Jahre': '60-69',
        '70+ Jahre': '70+'
    }
    
    for col, age_group in age_mapping.items():
        binary_df[f'age_{age_group}'] = (df['Alter'] == col).astype(int)
    
    # Handle gender
    binary_df['is_male'] = (df['Geschlecht'] == 'Männlich').astype(int)
    binary_df['is_female'] = (df['Geschlecht'] == 'Weiblich').astype(int)
    
    # Handle education
    education_levels = df['Bildungsstatus'].unique()
    for edu in education_levels:
        if pd.notna(edu):  # Skip NaN values
            binary_df[f'education_{edu}'] = (df['Bildungsstatus'] == edu).astype(int)
    
    # Handle income groups
    income_levels = df['Haushaltsnettoeinkommen in Euro'].unique()
    for income in income_levels:
        if pd.notna(income):
            binary_df[f'income_{income}'] = (df['Haushaltsnettoeinkommen in Euro'] == income).astype(int)
            
    # Handle occupation
    occupation_levels = df['Berufstätigkeit des Befragten'].unique()
    for occupation in occupation_levels:
        if pd.notna(occupation):
            binary_df[f'occupation_{occupation}'] = (df['Berufstätigkeit des Befragten'] == occupation).astype(int)
    
    return binary_df

binary_df = prepare_data(df)

# Debug
print("\nShape of binary dataframe:", binary_df.shape)
print("\nColumns in binary dataframe:", binary_df.columns.tolist())
print("\nSample of the data:")
print(binary_df.head())
print("\nColumn sums (to see frequency of each feature):")
print(binary_df.sum())

# Find frequent itemsets using Apriori with lower support
frequent_itemsets = apriori(binary_df, min_support=0.01, use_colnames=True)  # Lowered to 1%

# Debug
print("\nNumber of frequent itemsets found:", len(frequent_itemsets))
if len(frequent_itemsets) > 0:
    print("\nSample of frequent itemsets:")
    print(frequent_itemsets.head())

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)  # Lowered confidence threshold

# Sort rules by lift
rules = rules.sort_values('lift', ascending=False)

# Filter rules related to switching behavior
switch_rules = rules[
    rules['antecedents'].apply(lambda x: 'plans_to_switch' in str(x)) |
    rules['consequents'].apply(lambda x: 'plans_to_switch' in str(x))
]

# Print the most interesting rules
print("\nMost interesting patterns about insurance switching behavior:")
print("\nTop 10 rules by lift:")
pd.set_option('display.max_columns', None)
if len(switch_rules) > 0:
    print(switch_rules.head(10).to_string())
    # Save the results
    switch_rules.to_csv('pattern_mining/switch_patterns_results.csv', index=False)
    print("\nResults have been saved to 'pattern_mining/switch_patterns_results.csv'")
else:
    print("No switching patterns found. Try adjusting the support and confidence thresholds.") 