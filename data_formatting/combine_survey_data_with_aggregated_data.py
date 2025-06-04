import os
import re

import pandas as pd
from pandas.io.sas.sas_constants import column_name_offset_length


def get_question_columns(folderpath):
    column_set = set()
    for question_file in os.listdir(folderpath):
        dataframe_columns = pd.read_csv(os.path.join(folderpath, question_file)).columns
        # print(list(dataframe_columns))
        column_set = column_set.union(set(list(dataframe_columns)))

    column_set = sorted(column_set)

    clean_column_set = set()

    column_number_answers = dict()
    for column in column_set:
        column_str = str(column)

        column_split = column_str.split(".")
        column_name = column_split[0]
        if column_name not in column_number_answers.keys():
            column_number_answers[column_name] = 1
            clean_column_set.add(column_name)

        elif len(column_split) > 1:
            answer_number = column_split[1]
            column_number_answers[column_name] += 1

    print(sorted(clean_column_set))
    print(column_number_answers)


# def combine_survey_and_aggregated_data():
#     # The question number mapping
#     question_mapping = {
#         "Q1": 106,
#         "Q3": 107,
#         "Q4":
#     }
#
#     column_mapping = {
#         "Q1": 'Geschlecht',
#         "Q3": "Alter",
#         "Q4": "Bundesländer",
#     }


if __name__ == '__main__':
    get_question_columns("../formatted_data/Kundenmonitor_GKV_2023/Band")