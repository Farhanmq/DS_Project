import pandas as pd
import difflib
import openpyxl

# Load the file without headers
df = pd.read_excel("/content/drive/MyDrive/health_insurance_xai_project/data/processed/Band.xlsx", sheet_name = 
                   'Band', header=None)


questions = []
threshold = 0.85  # similarity threshold to consider two questions as duplicates

# Helper function to remove the duplicates made because of each question has multiple tables.
def is_duplicate(new_q, existing_qs, threshold=0.85):
    for q in existing_qs:
        ratio = difflib.SequenceMatcher(None, new_q.lower(), q.lower()).ratio()
        if ratio > threshold:
            return True
    return False

i = 0
while i < len(df) - 1:
    row1 = df.iloc[i].dropna().astype(str).str.strip().tolist()
    row2 = df.iloc[i + 1].dropna().astype(str).str.strip().tolist()

    # Heuristic
    if 0 < len(row1) < 5 and 0 < len(row2) < 5:
        combined_question = ' '.join(row1 + row2)

        if not is_duplicate(combined_question, questions, threshold):
            questions.append(combined_question)

        i += 2  # Skip the next row (already processed as part of question)
    else:
        i += 1

# Save to TXT file
with open("/content/drive/MyDrive/health_insurance_xai_project/data/Questions/Band_2024_questions.txt", "w", encoding="utf-8") as f:
    for q in questions:
        f.write(q + "\n")

print(f"Extracted {len(questions)} unique questions.")
