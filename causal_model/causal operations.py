import os

import networkx as nx
import numpy as np
import pandas as pd

from causallearn.search.ConstraintBased.FCI import fci
# from independence_tests_with_fallback import fci
from causallearn.utils.GraphUtils import GraphUtils
from dowhy import gcm
from numpy.distutils.misc_util import is_string
from pandas import DataFrame


def causal_search_space_reduction(questions: DataFrame, vos: str, output_file:str):
    """
    Reduces the search space by appling causal discovery on the set of questions.
    :param questions: The questions as dataframe.
    :param vos: The variable of significant importance.
    :return: The reduced search space.
    """
    # dataframe_elements = []
    # for filename in os.listdir(folder_path):
    #     dataframe_elements.append(pd.read_csv(os.path.join(folder_path, filename))["values"])
    #
    # question_dataframe = pd.concat(dataframe_elements)

    questions_drop = questions.fillna(0)
    vos_question = questions[vos].fillna(0)


    #find string columns
    str_columns = []
    for c in questions_drop.columns:
        str_col = questions_drop.loc[1:,c].apply(is_string)
        if str_col.any():
            str_columns.append(c)

    # questions_np = np.delete(questions_drop, str_columns, axis=1)
    questions_np = questions_drop.drop(str_columns, axis=1)
    questions_np_header = list(questions_drop.drop(str_columns, axis=1).columns)
    questions_np = questions_np.to_numpy().astype(int)
    # account for numpy error
    # questions_np = np.delete(questions_np, [75], axis=1).astype(int)

    # questions_np = np.delete(questions_np, [0], axis=0)
    # questions_np = questions_np.to_numpy()
    # questions_np = questions_np[:,~np.any(np.isnan(questions_np), axis=1)]
    # questions_np =  np.ndarray(questions_np)

    # same columns
    # same_columns = []
    # for i in range(questions_np.shape[1]):
    #     for j in range(questions_np.shape[1]):
    #         if (questions_np[:,i] == questions_np[:,j]).all() and i != j and not (j,i) in same_columns:
    #             same_columns.append((i, j))
    #
    # columns_to_delete = [j for (i,j) in same_columns]
    # questions_np = np.delete(questions_np, columns_to_delete, axis=1)

    p_values = []
    for i in range(questions_np.shape[1]):
        p_values.append(gcm.independence_test(questions_np[:,i], vos_question))

    dependent_questions = np.array(p_values) < 0.05

    graph_edges = []
    for i in range(len(dependent_questions)):
        if dependent_questions[i]:
            graph_edges.append((questions_np_header[i],vos))


    # random_noise = np.clip(np.random.rand(*questions_np.shape) -0.5,-0.1,0.1)
    # questions_np = questions_np + random_noise
    #
    # graph, edges = fci(questions_np,independence_test_method="fisherz", verbose= True, show_progress=True, depth=1, max_path_length=1)

    # reduce the search space to relevant questions
    # edges_to_vos = []
    # for edge in edges:
    #     if edge.node2 == vos:
    #         edges_to_vos.append(edge)

    causal_graph = nx.DiGraph(graph_edges)

    # visualization

    # pdy = GraphUtils.to_pydot(graph)
    nx.write_gml(causal_graph, output_file)
    # graph = nx.drawing.nx_pydot.to_pydot(causal_graph)
    # graph.write_png(output_file)
    # causal_graph.

    return causal_graph


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
    table = pd.read_excel("../provided_data/230807_Survey.xlsx", sheet_name="Result")
    causal_search_space_reduction(table,"Q16","./causal_graph.gml")