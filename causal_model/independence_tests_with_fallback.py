import warnings
from math import sqrt, log

import numpy as np
from causallearn.search.ConstraintBased.FCI import reorientAllWith, rule0, removeByPossibleDsep, rulesR1R2cycle, ruleR3, \
    ruleR4B, ruleR5, ruleR6, ruleR7, rule8, rule9, rule10, get_color_edges
from causallearn.utils.cit import CIT_Base
from scipy.stats import norm
from statsmodels.tools.sm_exceptions import ValueWarning

# from __future__ import annotations

import warnings
from queue import Queue
from typing import List, Set, Tuple, Dict, Generator
from numpy import ndarray

from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.Graph import Graph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Node import Node
from causallearn.utils.ChoiceGenerator import ChoiceGenerator
from causallearn.utils.DepthChoiceGenerator import DepthChoiceGenerator
from causallearn.utils.cit import *
from causallearn.utils.FAS import fas
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from itertools import combinations


def fci(dataset: ndarray, independence_test_method: str = fisherz, alpha: float = 0.05, depth: int = -1,
        max_path_length: int = -1, verbose: bool = False, background_knowledge: BackgroundKnowledge | None = None,
        show_progress: bool = True, node_names=None,
        **kwargs) -> Tuple[Graph, List[Edge]]:
    """
    Perform Fast Causal Inference (FCI) algorithm for causal discovery

    Parameters
    ----------
    dataset: data set (numpy ndarray), shape (n_samples, n_features). The input data, where n_samples is the number of
            samples and n_features is the number of features.
    independence_test_method: str, name of the function of the independence test being used
            [fisherz, chisq, gsq, kci]
           - fisherz: Fisher's Z conditional independence test
           - chisq: Chi-squared conditional independence test
           - gsq: G-squared conditional independence test
           - kci: Kernel-based conditional independence test
    alpha: float, desired significance level of independence tests (p_value) in (0,1)
    depth: The depth for the fast adjacency search, or -1 if unlimited
    max_path_length: the maximum length of any discriminating path, or -1 if unlimited.
    verbose: True is verbose output should be printed or logged
    background_knowledge: background knowledge

    Returns
    -------
    graph : a GeneralGraph object, where graph.graph[j,i]=1 and graph.graph[i,j]=-1 indicates  i --> j ,
                    graph.graph[i,j] = graph.graph[j,i] = -1 indicates i --- j,
                    graph.graph[i,j] = graph.graph[j,i] = 1 indicates i <-> j,
                    graph.graph[j,i]=1 and graph.graph[i,j]=2 indicates  i o-> j.
    edges : list
        Contains graph's edges properties.
        If edge.properties have the Property 'nl', then there is no latent confounder. Otherwise,
            there are possibly latent confounders.
        If edge.properties have the Property 'dd', then it is definitely direct. Otherwise,
            it is possibly direct.
        If edge.properties have the Property 'pl', then there are possibly latent confounders. Otherwise,
            there is no latent confounder.
        If edge.properties have the Property 'pd', then it is possibly direct. Otherwise,
            it is definitely direct.
    """

    if dataset.shape[0] < dataset.shape[1]:
        warnings.warn("The number of features is much larger than the sample size!")

    # independence_test_method = CIT(dataset, method=independence_test_method, **kwargs)
    independence_test_method = FisherZ_F(dataset, **kwargs)
    ## ------- check parameters ------------
    if (depth is None) or type(depth) != int:
        raise TypeError("'depth' must be 'int' type!")
    if (background_knowledge is not None) and type(background_knowledge) != BackgroundKnowledge:
        raise TypeError("'background_knowledge' must be 'BackgroundKnowledge' type!")
    if type(max_path_length) != int:
        raise TypeError("'max_path_length' must be 'int' type!")
    ## ------- end check parameters ------------

    nodes = []
    if node_names is None:
        node_names = [f"X{i + 1}" for i in range(dataset.shape[1])]
    for i in range(dataset.shape[1]):
        node = GraphNode(node_names[i])
        node.add_attribute("id", i)
        nodes.append(node)

    # FAS (“Fast Adjacency Search”) is the adjacency search of the PC algorithm, used as a first step for the FCI algorithm.
    graph, sep_sets, test_results = fas(dataset, nodes, independence_test_method=independence_test_method, alpha=alpha,
                                        knowledge=background_knowledge, depth=depth, verbose=verbose,
                                        show_progress=show_progress)

    # pdb.set_trace()
    reorientAllWith(graph, Endpoint.CIRCLE)

    rule0(graph, nodes, sep_sets, background_knowledge, verbose)

    removeByPossibleDsep(graph, independence_test_method, alpha, sep_sets)

    reorientAllWith(graph, Endpoint.CIRCLE)
    rule0(graph, nodes, sep_sets, background_knowledge, verbose)

    change_flag = True
    first_time = True

    while change_flag:
        change_flag = False
        change_flag = rulesR1R2cycle(graph, background_knowledge, change_flag, verbose)
        change_flag = ruleR3(graph, sep_sets, background_knowledge, change_flag, verbose)

        if change_flag or (first_time and background_knowledge is not None and
                           len(background_knowledge.forbidden_rules_specs) > 0 and
                           len(background_knowledge.required_rules_specs) > 0 and
                           len(background_knowledge.tier_map.keys()) > 0):
            change_flag = ruleR4B(graph, max_path_length, dataset, independence_test_method, alpha, sep_sets,
                                  change_flag,
                                  background_knowledge, verbose)

            first_time = False

            if verbose:
                print("Epoch")

        # rule 5
        change_flag = ruleR5(graph, change_flag, verbose)

        # rule 6
        change_flag = ruleR6(graph, change_flag, verbose)

        # rule 7
        change_flag = ruleR7(graph, change_flag, verbose)

        # rule 8
        change_flag = rule8(graph, nodes, change_flag)

        # rule 9
        change_flag = rule9(graph, nodes, change_flag)
        # rule 10
        change_flag = rule10(graph, change_flag)

    graph.set_pag(True)

    edges = get_color_edges(graph)

    return graph, edges


class FisherZ_F(CIT_Base):
    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self.check_cache_method_consistent('fisherz_f', NO_SPECIFIED_PARAMETERS_MSG)
        self.assert_input_data_is_valid()
        self.correlation_matrix = np.corrcoef(data.T)

    def __call__(self, X, Y, condition_set=None):
        '''
        Perform an independence test using Fisher-Z's test.

        Parameters
        ----------
        X, Y and condition_set : column indices of data

        Returns
        -------
        p : the p-value of the test
        '''
        Xs, Ys, condition_set, cache_key = self.get_formatted_XYZ_and_cachekey(X, Y, condition_set)
        if cache_key in self.pvalue_cache: return self.pvalue_cache[cache_key]
        var = Xs + Ys + condition_set
        sub_corr_matrix = self.correlation_matrix[np.ix_(var, var)]
        try:
            inv = np.linalg.inv(sub_corr_matrix)
        except np.linalg.LinAlgError:
            warnings.warn("Data correlation matrix is singular. Cannot run fisherz test. Please check your data.", ValueWarning)
            return 0

            # raise ValueError('Data correlation matrix is singular. Cannot run fisherz test. Please check your data.')
        r = -inv[0, 1] / sqrt(abs(inv[0, 0] * inv[1, 1]))
        if abs(r) >= 1: r = (1. - np.finfo(float).eps) * np.sign(r) # may happen when samplesize is very small or relation is deterministic
        Z = 0.5 * log((1 + r) / (1 - r))
        X = sqrt(self.sample_size - len(condition_set) - 3) * abs(Z)
        p = 2 * (1 - norm.cdf(abs(X)))
        self.pvalue_cache[cache_key] = p
        return p