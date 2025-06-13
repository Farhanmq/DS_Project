# -*- coding: utf-8 -*-
"""
Created on Fri May 22 15:37:25 2025

@author: LENOVO
"""

import pandas as pd

def split_sections_by_single_nan_row(dataframe_list):
    """
    Takes a list of DataFrames, and for each one:
    - Finds the single completely NaN row
    - Splits the DataFrame into two sections at that row
    - Returns a new list of all resulting DataFrames

    Raises an error if:
    - No NaN row is found
    - More than one NaN row is found
    """
    part1_tables = []
    part2_tables = []

    err_counter = -1
    for idx, df in enumerate(dataframe_list):
        err_counter = err_counter + 1
        # Find the index of completely NaN rows
        nan_rows = df[df.isnull().all(axis=1)].index

        if len(nan_rows) == 0:
            raise ValueError(f"No NaN row found in DataFrame {idx + 1}")
            print(err_counter)
        elif len(nan_rows) > 1:
            print(err_counter)
            raise ValueError(f"More than one NaN row found in DataFrame {idx + 1}")

        nan_idx = nan_rows[0]

        # Split into two parts and drop the NaN row
        part1 = df.iloc[:nan_idx].dropna(axis=1, how='all').reset_index(drop=True)
        part2 = df.iloc[nan_idx + 1:].reset_index(drop=True)
        
        # For all except the first part2, drop the first two columns
        if idx > 0:
            part2 = part2.iloc[:, 2:].reset_index(drop=True)

        part1_tables.append(part1)
        part2_tables.append(part2)
        
    # Concatenate all part2s horizontally
    merged_df = pd.concat(part2_tables, axis=1)
    
    return part1_tables, merged_df



#import pandas as pd

#def one_for_all(list_of_dataframes):
    
    
    