# -*- coding: utf-8 -*-
"""
Created on Fri May 22 14:36:55 2025

@author: LENOVO
"""
import os
import pandas as pd

from pprocess_bcg_01 import split_excel_by_empty_rows
from pprocess_bcg_02 import split_sections_by_single_nan_row

# Find excel file path
dir_name = os.path.dirname(__file__)
excel_path = os.path.join(dir_name,'../GKV_2024_Band/Question1.xlsx')
if not os.path.exists(excel_path):
    print("Path not found:", excel_path)
else:
    print("File found at:", excel_path)

# define output path
output_path = 'pprocess_bcg_output_' + excel_path.split('/')[2]

# Call the function to split the Excel file
dataframes = split_excel_by_empty_rows(excel_path)

# Call the function to merge tables into one
part1, tables = split_sections_by_single_nan_row(dataframes)

with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    # Write part1 at the top
    part1[0].to_excel(writer, sheet_name='Sheet1', index=False, header=False)

    # Write merged_df below part1, starting at the next row
    start_row = len(part1[0]) + 1  # +2 for one empty row between
    tables.to_excel(writer, sheet_name='Sheet1', startrow=start_row, index=False, header=False)


# Print how many sections were found
print(f"Total sections: {len(dataframes)}")

# Example: Show the first section
print("First section:")
print(dataframes[0])