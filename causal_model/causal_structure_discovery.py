import re
from typing import Callable

import numpy as np
import pandas as pd
import causallearn.search as cl
# Deprecated
class ObservationalRandomVariable:
    def __init__(self,generator: Callable[[int],np.ndarray], description: pd.DataFrame):
        """
        Represents an observational random variable of a data table.
        :param generator: The generator to generate samples of the random variable.
        :param description: The description of the random variable, including label definitions and their distribution.
        """
        self.generator = generator
        self.description = description

    def sample(self,n) -> np.ndarray:
        """
        Samples n random values of the random variable.
        :param n: The number of samples to generate.
        :return: An array of n samples of the random variable.
        """
        return self.generator(n)

def define_random_variable_for_question_table(question_table:pd.DataFrame)-> ObservationalRandomVariable:
    # find labels and probabilities per label of total distribution of question
    #TODO differentiate first column "Gesamt" or not
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
    random_variable_generator = lambda num_samples: np.random.choice(np.array(random_variable_table["label_numbers"]),
                                                num_samples,
                                                p=np.array(label_probabilities))
    return ObservationalRandomVariable(random_variable_generator, random_variable_table)


def define_random_variables_for_question_tables(question_tables:list[pd.DataFrame])-> list[ObservationalRandomVariable]:
    random_variables = []
    for question_table in question_tables:
        random_variables.append(define_random_variable_for_question_table(question_table))
    return random_variables


if __name__ == '__main__':
    question_table = pd.read_csv('../formatted_data/Kundenmonitor_GKV_2023/Band/Question_1.csv')
    r = define_random_variable_for_question_table(question_table)
    print(type(r))
    print(r(100))