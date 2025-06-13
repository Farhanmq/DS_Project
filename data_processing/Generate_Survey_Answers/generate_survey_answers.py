'''
    IMPORTANT - This code will generate multiple files. Each file for a different question.
'''
import os
import pandas as pd

# Find excel file path
dir_name = os.path.dirname(__file__)
excel_path = os.path.join(dir_name,'../../provided_data/230807_Survey.xlsx')
if not os.path.exists(excel_path):
    print("Path not found:", excel_path)
else:
    print("File found at:", excel_path)

# Load the Excel file
xls = pd.ExcelFile(excel_path)

# Load sheets
codebook_df = pd.read_excel(xls, sheet_name='Codebook')
result_df = pd.read_excel(xls, sheet_name='Result')

# Apply forward-fill in-place for those columns
columns_to_ffill = codebook_df.columns[[1, 2]]
for col in columns_to_ffill:
    codebook_df[col] = codebook_df[col].ffill()

try:
    phase_col_index = result_df.columns.get_loc('Phase')
    headers_after_phase = result_df.iloc[:, phase_col_index + 1:].columns.tolist()
except KeyError:
    headers_after_phase = []

for header in headers_after_phase:
    # Step 1: Locate the Q1 metadata and value-label mapping in Codebook
    question_start_idx = codebook_df[codebook_df['Question'] == header].index[0]
    question_end_idx = codebook_df.iloc[question_start_idx + 1:].index[
        codebook_df.iloc[question_start_idx + 1:]['Question'].notna()
    ].min()

    question_block = codebook_df.iloc[question_start_idx:question_end_idx]

    # Extract metadata
    question_question_code = question_block.iloc[0]['Question']
    question_type = question_block.iloc[0]['Type']
    question_name = question_block.iloc[0]['Name']

    # Create a dictionary for value-label mapping
    value_label_map = question_block[['Value', 'Label']].dropna().set_index('Value')['Label'].to_dict()

    # Step 2: Extract Question answers from result sheet
    question_column = question_question_code # Question name like 'Q1', 'Q2' and so on
    question_answers_df = result_df[['Participant', question_column]].copy()

    # Step 3: Map answers to labels
    def map_label(value):
        key = value
            
        if '0,1' in value_label_map:
            return f"{value_label_map.get('0,1', 'NULL')}"
        else:
            return f"{value_label_map.get(key, 'NULL')}"

    question_answers_df['Answer'] = question_answers_df[question_column]
    question_answers_df['Value'] = question_answers_df[question_column].apply(map_label)
    
    # Step 4: Add metadata columns
    question_answers_df['Question'] = question_question_code
    question_answers_df['Type'] = question_type
    question_answers_df['Name'] = question_name

    # Step 5: Reorder columns
    final_df = question_answers_df[
        ['Participant',
        'Question',
        'Type',
        'Name',
        'Answer',
        'Value']
    ]

    # Step 6: Export to Excel
    final_df.to_excel("questions/"+header+"_Survey_Answers.xlsx", index=False)
