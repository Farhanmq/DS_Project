import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import LabelEncoder
import os
from pathlib import Path
from tqdm import tqdm
import gc

def load_and_prepare_data(data_dir):
    """
    Load all question CSV files and prepare them for pattern mining
    """
    data_path = Path(data_dir)
    all_data = {}
    
    # Load question table first
    question_table = pd.read_csv(data_path / 'question_table.csv')
    print(f"Loaded question table with shape: {question_table.shape}")
    
    # Get list of all question files first
    question_files = list(data_path.glob('Question_*.csv'))
    question_files = [f for f in question_files if f.name.lower() != 'question_table.csv']
    
    # Load all question files with progress bar
    print("\nLoading and processing question files:")
    for file in tqdm(question_files, desc="Processing files"):
        question_num = int(file.stem.split('_')[1])
        try:
            df = pd.read_csv(file)
            
            # Get the first row which contains the actual column names
            column_names = df.columns.tolist()
            
            # Get the second row which contains the categories
            if len(df) >= 2:
                categories = df.iloc[0].tolist()
                
                # Create a dictionary of category values for this question
                category_values = {}
                for col, cat in zip(column_names[1:], categories[1:]):  # Skip the first empty column
                    if pd.notna(cat) and str(cat).strip():
                        try:
                            value = float(df.iloc[-1][col])
                            if not np.isnan(value):
                                category_values[col] = value
                        except (ValueError, TypeError):
                            continue
                
                all_data[question_num] = category_values
            
            # Clean up memory
            del df
            gc.collect()
            
        except Exception as e:
            print(f"\nError processing {file.name}: {str(e)}")
    
    print(f"\nSuccessfully processed {len(all_data)} question files")
    return all_data, question_table

def prepare_for_pattern_mining(data_dict, min_support_value=10.0):
    """
    Prepare the data for pattern mining by creating a binary matrix of responses
    that meet the minimum support threshold
    """
    print("\nPreparing data for pattern mining...")
    
    # Create a list of all patterns that meet the minimum support threshold
    patterns = []
    pattern_names = []
    
    # Process patterns with progress bar
    items = list(data_dict.items())
    for question_num, category_values in tqdm(items, desc="Processing patterns"):
        # Sort categories by value and take only top 3 most common responses per question
        sorted_categories = sorted(category_values.items(), key=lambda x: x[1], reverse=True)[:3]
        for category, value in sorted_categories:
            if value >= min_support_value:  # Only include patterns that appear in at least 10% of responses
                patterns.append(1)  # Convert to binary: 1 if the pattern exists
                pattern_names.append(f"Q{question_num}_{category}")
    
    # Create a DataFrame with the patterns
    pattern_df = pd.DataFrame([patterns], columns=pattern_names).astype(bool)
    print(f"\nCreated pattern matrix with shape: {pattern_df.shape}")
    
    return pattern_df

def perform_pattern_mining(data, min_support=0.1, min_confidence=0.5):
    """
    Perform pattern mining using the Apriori algorithm
    """
    print("\nStarting pattern mining...")
    print(f"Input data shape: {data.shape}")
    print(f"Using minimum support: {min_support*100}% and minimum confidence: {min_confidence*100}%")
    
    # Generate frequent itemsets
    print("\nGenerating frequent itemsets...")
    frequent_itemsets = apriori(data,
                              min_support=min_support,
                              use_colnames=True,
                              max_len=3)  # Limit to patterns of at most 3 items
    
    print(f"Found {len(frequent_itemsets)} frequent itemsets")
    
    if len(frequent_itemsets) > 0:
        # Generate association rules
        print("\nGenerating association rules...")
        rules = association_rules(frequent_itemsets,
                                metric="confidence",
                                min_threshold=min_confidence)
        
        print(f"Generated {len(rules)} association rules")
        
        # Add question numbers to the rules for better readability
        def extract_question_num(pattern_list):
            return [int(p.split('_')[0].replace('Q', '')) for p in pattern_list]
        
        if len(rules) > 0:
            print("\nProcessing rules...")
            rules['antecedent_questions'] = rules['antecedents'].apply(extract_question_num)
            rules['consequent_questions'] = rules['consequents'].apply(extract_question_num)
    else:
        rules = pd.DataFrame()
        print("No association rules could be generated")
    
    return frequent_itemsets, rules

def analyze_patterns():
    """
    Main function to analyze patterns in the data
    """
    print("Starting pattern analysis...")
    
    try:
        # Load data
        data_dir = "formatted_data/Kundenmonitor_GKV_2023/Band"
        all_data, question_table = load_and_prepare_data(data_dir)
        
        # Prepare data for pattern mining with higher support threshold
        prepared_data = prepare_for_pattern_mining(all_data, min_support_value=10.0)
        
        # Perform pattern mining with higher thresholds
        frequent_itemsets, rules = perform_pattern_mining(
            prepared_data,
            min_support=0.1,  # 10% minimum support
            min_confidence=0.6  # 60% minimum confidence
        )
        
        # Save results
        output_dir = Path("pattern_mining/results")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        if len(frequent_itemsets) > 0:
            # Add support percentage to frequent itemsets
            frequent_itemsets['support_pct'] = frequent_itemsets['support'] * 100
            
            # Sort by support
            frequent_itemsets = frequent_itemsets.sort_values('support', ascending=False)
            
            # Save top 1000 frequent itemsets
            frequent_itemsets.head(1000).to_csv(output_dir / "frequent_itemsets.csv")
            print(f"\nSaved top 1000 frequent itemsets")
            
            if len(rules) > 0:
                # Add percentage columns for better interpretability
                rules['support_pct'] = rules['support'] * 100
                rules['confidence_pct'] = rules['confidence'] * 100
                
                # Sort rules by lift and take top 1000
                rules = rules.sort_values('lift', ascending=False)
                rules.head(1000).to_csv(output_dir / "association_rules.csv")
                print(f"Saved top 1000 association rules")
                
                # Print top 10 rules by lift
                print("\nTop 10 association rules by lift:")
                pd.set_option('display.max_columns', None)
                pd.set_option('display.max_rows', None)
                print(rules[['antecedent_questions', 'consequent_questions', 'support_pct', 'confidence_pct', 'lift']].head(10))
        else:
            print("No patterns found that meet the minimum support and confidence thresholds")
            
    except Exception as e:
        print(f"\nError during pattern analysis: {str(e)}")
        raise

if __name__ == "__main__":
    analyze_patterns() 