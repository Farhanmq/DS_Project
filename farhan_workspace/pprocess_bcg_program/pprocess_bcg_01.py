# -*- coding: utf-8 -*-
"""
Created on Fri May 22 14:50:40 2025

@author: LENOVO
"""
import pandas as pd

def split_excel_by_empty_rows(excel_path):
    """
    Load an Excel file and split it into multiple DataFrames
    wherever two consecutive empty rows are found.
    
    Returns a list of DataFrames.
    """
    df = pd.read_excel(excel_path, header=None)

    # Identify empty row indices
    empty_row_indices = df[df.isnull().all(axis=1)].index

    # Find break points (two consecutive empty rows)
    break_points = []
    for i in range(len(empty_row_indices) - 1):
        if empty_row_indices[i + 1] - empty_row_indices[i] == 1:
            break_points.append(empty_row_indices[i])

    # Determine start and end ranges
    sections = []
    start = 0
    for bp in break_points:
        sections.append((start, bp))
        start = bp + 2
    sections.append((start, len(df)))
    
    separated_dataframes = []
    err_counter = -1
    for start, end in sections:
        sub_df = df.iloc[start:end].dropna(axis=1, how='all').reset_index(drop=True)

        # Optional: apply ffill to row 3 if it exists
        if sub_df.shape[0] > 3:
            sub_df.iloc[3, 1:] = sub_df.iloc[3, 1:].ffill()
            
        # Validate: no more than one fully NaN row
        nan_row_count = sub_df[sub_df.isnull().all(axis=1)].shape[0]
        #if nan_row_count > 1:
        #    print("Error in record: " + str(err_counter))
        #    raise ValueError(f"Section from row {start} to {end} contains {nan_row_count} empty rows.")

        separated_dataframes.append(sub_df)
        err_counter = err_counter + 1

    return separated_dataframes

#--- END ---