import os.path

import numpy as np
import pandas as pd
from numpy.ma.core import shape


# from slugify import slugify


def create_table_per_question(filepath_data, sheet_name, folder_path_questions):
    """
    Creates tables for each question of a sheet  of an excel file and saves them in a folder.
    :param filepath_data: The filepath of the excel file containing the data.
    :param sheet_name: The name of the sheet in the excel file.
    :param folder_path_questions: The folder path containing the resulting question tables.
    """
    table =pd.read_excel(filepath_data, sheet_name=sheet_name, header=None)
    # print(table)

    #get indicies of nan rows
    nan_rows = table.isna().all(axis=1)
    nan_rows = nan_rows.where(cond=(nan_rows==True)).dropna()
    # print(nan_rows)

    #init question_tables dict
    question_tables = dict()

    # read tables
    last_index=-1
    second_last_index=-2
    current_question=""

    for i in range(len(nan_rows)):
        if last_index != nan_rows.index[i] -1 and second_last_index == last_index -1:
            # read question
            question_raw = table.iloc[last_index+1:nan_rows.index[i],:]

            if current_question != question_to_string(question_raw):
                # New question gets added to the dictionary
                current_question = question_to_string(question_raw)
                question_tables[current_question] = []

            # print(question_to_string(question_raw))
            # print(question_raw)


        elif last_index == nan_rows.index[i] -1:
            table.iloc[second_last_index+1,:].ffill(inplace=True)
            # add table to question list
            question_tables[current_question].append(pd.DataFrame(table.iloc[second_last_index+1:last_index, :])
                                                     .reset_index(drop=True))
            # print("table")

        # update indices
        second_last_index = last_index
        last_index = nan_rows.index[i]

    # combine tables for each question
    for question in question_tables.keys():
        for i in range(len(question_tables[question])):
            if i != 0:
                # remove duplicate columns
                question_tables[question][i] = question_tables[question][i].iloc[:, 2:]

        # concat question tables to one big question table
        question_tables[question] = pd.concat(question_tables[question], axis=1)
        question_tables[question].columns = question_tables[question].iloc[0]
        question_tables[question] = question_tables[question].iloc[1:, :]

    # save table for each question
    if not os.path.exists(folder_path_questions):
        os.makedirs(folder_path_questions)

    i = 1
    question_list = []
    for question in question_tables.keys():
        # filename = slugify(question)
        # if len(filename) > 250:
        #     filename = filename[:250]
        filename = "Question_" + str(i)
        question_list.append(filename)
        question_tables[question].to_csv(os.path.join(folder_path_questions, filename + ".csv"), index=False)
        i+=1

    # questions = np.empty(shape=(len(question_list),2))
    # questions[:,0] = question_list
    # questions[:,1] = question_tables.keys()
    # write questions table
    question_table =pd.DataFrame({"Question Nr":question_list,"Question": question_tables.keys() },
                                 columns=["Question Nr","Question"])
    question_table.to_csv(os.path.join(folder_path_questions, "question_table.csv"), index=False)


def question_to_string(question_table: pd.DataFrame) -> str:
    """
    Converts a question table into a question string.
    :param question_table: The table, in which the question is included (over multiple rows).
    :return: The question string.
    """
    question_string = ""
    for i in range(len(question_table)):
        if question_string != "":
            question_string += " | "
        question_string += str(question_table.iloc[i, 0])
    return question_string


def find_faulty_tables(folder_path):
    """
    Prints the filenames of all faulty question tables in a folder.
    :param folder_path: The folder path to the question tables.
    """
    for file in os.listdir(folder_path):
        df = pd.read_csv(os.path.join(folder_path, file))
        if file != "question_table.csv" and df.columns[1] != "Gesamt":
            print("faulty table: " + file)


if __name__ == '__main__':
    create_table_per_question("../provided_data/Kundenmonitor_GKV_2023.xlsx", "Band", "../formatted_data/Kundenmonitor_GKV_2023/Band")
    find_faulty_tables("../formatted_data/Kundenmonitor_GKV_2023/Band")