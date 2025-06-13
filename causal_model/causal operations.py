import os

import networkx as nx
import pandas as pd

from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.GraphUtils import GraphUtils
from dowhy import gcm
from pandas import DataFrame


def causal_search_space_reduction(questions: DataFrame, vos: str):
    """
    Reduces the search space by appling causal discovery on the set of questions.
    :param folder_path: The folder path to the questions.
    :param vos: The variable of significant importance.
    :return: The reduced search space.
    """
    # dataframe_elements = []
    # for filename in os.listdir(folder_path):
    #     dataframe_elements.append(pd.read_csv(os.path.join(folder_path, filename))["values"])
    #
    # question_dataframe = pd.concat(dataframe_elements)

    questions_drop = questions.dropna(axis="columns")
    graph, edges = fci(questions_drop.to_numpy())

    # reduce the search space to relevant questions
    # edges_to_vos = []
    # for edge in edges:
    #     if edge.node2 == vos:
    #         edges_to_vos.append(edge)

    # visualization
    pdy = GraphUtils.to_pydot(g)
    pdy.write_png('simple_test.png')

    return edges_to_vos


def causal_pattern_importance_assessment(patterns: list, patterns_to_questions: dict, sov, threshold: float):
    confounder_patterns = []
    for pattern in patterns:
        result = gcm.independence_test(pattern, sov, conditioned_on=patterns_to_questions[pattern])

        # add the patterns to the result, which fail the independence test
        if result < threshold:
            confounder_patterns.append(pattern)

    return confounder_patterns


def causal_pattern_inference(patterns: list, sov, threshold: float):
    graph_data = []
    for i in range(len(patterns)):
        graph_data.append(["P" + str(i), "SOV"])
    causal_graph = nx.DiGraph(graph_data)
    causal_model = gcm.StructuralCausalModel(causal_graph)

    gcm.auto.assign_causal_mechanisms(causal_model,patterns+sov)


if __name__ == '__main__':
    table = pd.read_excel("../provided_data/230807_Survey.xlsx", sheet_name="Result", header=None)
    causal_search_space_reduction(table,"")