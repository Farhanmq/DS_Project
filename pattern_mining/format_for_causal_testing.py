import pandas as pd
import numpy as np
from pathlib import Path
import ast

def load_pattern_results():
    """
    Load the pattern mining results from the CSV files
    """
    results_dir = Path("pattern_mining/results")
    rules = pd.read_csv(results_dir / "association_rules.csv")
    itemsets = pd.read_csv(results_dir / "frequent_itemsets.csv")
    return rules, itemsets

def parse_pattern(pattern_str):
    """
    Parse a pattern string into a list of (question_num, category) tuples
    """
    # Remove frozenset and split into individual items
    pattern_str = pattern_str.replace("frozenset({", "").replace("})", "")
    items = pattern_str.split("', '")
    items = [item.strip("'") for item in items]
    
    # Parse each item into question number and category
    parsed_items = []
    for item in items:
        if item.startswith("Q"):
            parts = item.split("_", 1)
            if len(parts) == 2:
                question_num = int(parts[0][1:])
                category = parts[1]
                parsed_items.append((question_num, category))
    
    return parsed_items

def load_question_data(question_num):
    """
    Load data for a specific question
    """
    data_path = Path(f"formatted_data/Kundenmonitor_GKV_2023/Band/Question_{question_num}.csv")
    if not data_path.exists():
        return None
    
    df = pd.read_csv(data_path)
    return df

def find_matching_data_points(pattern, question_data_dict):
    """
    Find data points that match a given pattern
    """
    if not question_data_dict:
        return []
        
    # Initialize with all indices from the first dataframe
    first_df = next(iter(question_data_dict.values()))
    matching_indices = set(range(len(first_df)))
    
    for question_num, category in pattern:
        if question_num not in question_data_dict:
            continue
            
        df = question_data_dict[question_num]
        # Find the column that matches the category
        matching_cols = [col for col in df.columns if category in col]
        if not matching_cols:
            continue
            
        # Get indices where the category is present
        for col in matching_cols:
            # Convert column to numeric if possible, otherwise treat as string
            try:
                col_values = pd.to_numeric(df[col], errors='coerce')
                col_indices = set(df[~col_values.isna() & (col_values > 0)].index)
            except:
                # If conversion fails, treat as string and check for non-empty values
                col_indices = set(df[df[col].notna() & (df[col] != '')].index)
            
            matching_indices = matching_indices.intersection(col_indices)
    
    return list(matching_indices)

def format_for_causal_testing():
    """
    Main function to format pattern mining results for causal independence testing
    """
    print("Loading pattern mining results...")
    rules, itemsets = load_pattern_results()
    
    # Create output directory
    output_dir = Path("pattern_mining/causal_testing")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load all question data
    print("Loading question data...")
    question_data_dict = {}
    for question_num in range(1, 200):  # Adjust range based on your data
        df = load_question_data(question_num)
        if df is not None:
            question_data_dict[question_num] = df
    
    # Process each rule
    print("Processing patterns for causal testing...")
    causal_testing_data = []
    
    for _, rule in rules.iterrows():
        try:
            # Parse antecedent and consequent patterns
            antecedent_pattern = parse_pattern(rule['antecedents'])
            consequent_pattern = parse_pattern(rule['consequents'])
            
            if not antecedent_pattern or not consequent_pattern:
                continue
            
            # Find matching data points
            matching_indices = find_matching_data_points(antecedent_pattern, question_data_dict)
            
            if not matching_indices:
                continue
            
            # Create a record for causal testing
            for idx in matching_indices:
                record = {
                    'pattern_id': f"pattern_{len(causal_testing_data)}",
                    'antecedent_pattern': str(antecedent_pattern),
                    'consequent_pattern': str(consequent_pattern),
                    'data_point_index': idx,
                    'support': rule['support'],
                    'confidence': rule['confidence'],
                    'lift': rule['lift']
                }
                causal_testing_data.append(record)
        except Exception as e:
            print(f"Error processing rule: {rule['antecedents']} -> {rule['consequents']}")
            print(f"Error details: {str(e)}")
            continue
    
    # Convert to DataFrame and save
    if causal_testing_data:
        df = pd.DataFrame(causal_testing_data)
        df.to_csv(output_dir / "causal_testing_data.csv", index=False)
        print(f"Saved {len(df)} patterns for causal testing")
    else:
        print("No patterns found for causal testing")

if __name__ == "__main__":
    format_for_causal_testing() 