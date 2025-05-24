import re

import numpy as np
import pandas as pd
import causallearn.search as cl


def define_random_variable_for_question_table(question_table:pd.DataFrame)-> np.random:
    # find labels and probabilities per label of total distribution of question
    row_headers = question_table.iloc[:,0]
    row_headers = row_headers.dropna(inplace=False)

    # find last row with a total number value (n <*>) and Summe
    last_row_total_numbers = -1
    sum_row = -1
    for row_header in row_headers:
        if re.fullmatch("n .*", row_header) is not None:
            # save row index of last row seen with numerical data
            last_row_total_numbers = row_headers[row_headers==row_header].index[0]

        elif row_header == "Summe":
            # save row index of sum row
            sum_row = row_headers[row_headers==row_header].index[0]

    label_names = question_table.iloc[last_row_total_numbers+1:sum_row,0]
    label_probabilities = question_table.iloc[last_row_total_numbers+1:sum_row,1]/100
    label_numbers = list(range(sum_row - last_row_total_numbers -1))
    random_variable_table = pd.DataFrame({"label_names":label_names,
                                          "label_numbers": label_numbers,
                                          "label_probabilities":label_probabilities})
    # print(random_variable_table)
    # define random variable
    return lambda num_samples: np.random.choice(np.array(random_variable_table["label_numbers"]),
                                                num_samples,
                                                p=np.array(label_probabilities))


# def define_random_variables_for_question_tables(question_tables:list[str]):


if __name__ == '__main__':
    question_table = pd.read_csv('../formatted_data/Kundenmonitor_GKV_2023/Band/Question_1.csv')
    r = define_random_variable_for_question_table(question_table)
    print(type(r))
    print(r(100))