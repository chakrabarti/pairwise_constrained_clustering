import numpy as np
from typing import *
from scipy.spatial import distance
import sklearn.preprocessing as preprocessing
from spc import ClusteringProblem, kCenterScr, Objective
from enum import IntEnum
import pickle

Point = int


class Metric(IntEnum):
    F1 = 1
    F2 = 2
    F3 = 3


def GenerateClusteringProblem(data: np.array, k: int, similarity_metric: Metric = Metric.F2, m=100, name: str = None, load_scr: bool = False, scr_directory: str = None) -> ClusteringProblem:
    """
    Given a dataset, number of clusters and type of fairness constraints, creates a ClusteringProblem object representing the clustering problem with fairness constraints that we care about solving.
    Args:
        data (np.array): The points in the original metric space 
        k (int): The number of clusters.
        similarity_metric (Metric, optional): The type of fairness constraints we want to impose in the ClusteringProblem. Defaults to Metric.F2.
        m (int, optional):  Only relevant for Metric.F2. The number of neighbors to be considered for each point in the creation of the fairness constraints. Defaults to 100.
        name (str, optional): Only needs to be specified for Metric.F1. Saves the Scr clustering computed using this name so that it and its corresponding Scr radius can be reused. Defaults to None.
        load_scr (bool, optional): Only relevant for Metric.F1. If False, generates the Scr assignments and saves them. If True, loads the Scr assignments from the expected location. Defaults to False.
        scr_directory (str, optional): Only relevant for Metric.F1. If load_scr is set, the Scr assignments are loaded from this directory; otherwise they are saved to this directory. 

    Returns:
        ClusteringProblem: The created ClusteringProblem object on which we can now run our algorithm to produce clusterings.
    """
    scaler = preprocessing.StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)

    # Return Euclidean distance matrix between datapoints
    dist_matrix = distance.cdist(data, data, 'euclidean')

    # Scale distances between 0 and 1 so that if they are directly used as separation bounds they are sensible
    dist_matrix = dist_matrix/np.max(dist_matrix)

    pbs_problem = ClusteringProblem(k,
                                    data, dist_matrix, objective=Objective.KCENTER if similarity_metric == Metric.F1 else Objective.KMEANS)

    if similarity_metric == Metric.F2:
        # make sure not to pick yourself; have to add 1 because distance to self will be included
        new_m = min(pbs_problem.num_points, m+1)
        for i, row in enumerate(dist_matrix):
            rel_indices = np.argpartition(row, new_m-1)[:new_m]
            for j in rel_indices[rel_indices != i]:
                pbs_problem.AddStochasticConstraint(i, j, dist_matrix[i, j])
    if similarity_metric == Metric.F3:
        m = int(pbs_problem.num_points/k)
        # make sure not to pick yourself; have to add 1 because distance to self will be included
        new_m = min(pbs_problem.num_points, m+1)
        for i, row in enumerate(dist_matrix):
            rel_indices = np.argpartition(row, new_m-1)[:new_m]
            normalizing_factor = max(row[rel_indices])
            for j in rel_indices[rel_indices != i]:
                pbs_problem.AddStochasticConstraint(
                    i, j, dist_matrix[i, j]/normalizing_factor)

    if similarity_metric == Metric.F1:
        assert scr_directory is not None
        assert name is not None
        m = pbs_problem.num_points
        # If Scr clustering has already been computed, load this directly instead of recomputing
        if load_scr:
            with open(f"{scr_directory}/{name}_scr_assignments_{m}_{k}.out", "rb") as f:
                centers, Scr_radius, radii, center_assignments = pickle.load(f)
        else:
            centers, Scr_radius, radii, center_assignments = kCenterScr(
                dist_matrix, k)
            # Dump the Scr clustering produced so that it can be loaded next time
            with open(f"{scr_directory}/{name}_scr_assignments_{m}_{k}.out", "wb") as f:
                pickle.dump(
                    (centers, Scr_radius, radii, center_assignments), f)
        for i, row in enumerate(dist_matrix):
            for j in range(len(dist_matrix)):
                if i != j and dist_matrix[i, j]/Scr_radius <= 1:
                    pbs_problem.AddStochasticConstraint(
                        i, j, 16*dist_matrix[i, j]/(Scr_radius))

    return pbs_problem
