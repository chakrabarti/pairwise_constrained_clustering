import numpy as np
from typing import *
from dataclasses import *
import gurobipy as gp
from gurobipy import GRB
from sklearn.cluster import KMeans
from enum import Enum
from scipy.spatial import distance
import math
from operator import itemgetter
from itertools import groupby
from copy import deepcopy
from typing import *
import pickle
Point = int


@dataclass
class DistMatrix:
    dist_matrix: np.array

    # Check that distance matrix is valid; should have non-negative entries, be symmetric, and have 0s along diagonal
    def __post_init__(self):
        assert(self.dist_matrix.shape[0] == self.dist_matrix.shape[1])
        assert(np.all(self.dist_matrix.diagonal() == 0))
        assert(np.all(self.dist_matrix >= 0))
        assert(np.all(self.dist_matrix.T == self.dist_matrix))

    @property
    def num_points(self):
        return self.dist_matrix.shape[0]


class Objective(Enum):
    KMEANS = 1
    KCENTER = 2


class ClusteringProblem:
    def __init__(self, k: int, original_points: np.array, dist_matrix: np.array, objective: Objective = Objective.KMEANS):
        """
        Returns an object to which stochastic constraints can be added to represent the clustering problem we care about.
        Args:
            k (int): The number of clusters.
            original_points (np.array): Points in original metric space to be clustered.
            dist_matrix (np.array): Distance matrix corresponding to distances between each of the points in original_points (this doesn't have to be the Euclidean metric and will be used for k-center objective calculation).
            objective (Objective, optional): The type of objective we care about for the problem. Defaults to Objective.KMEANS.
        """
        self._k = k
        self._original_points = original_points
        self._objective = objective
        self._dist_matrix = DistMatrix(dist_matrix)
        self._stochastic_constraints: Dict[Tuple[Point, Point], float] = dict()

    def AddStochasticConstraint(self, i: Point, j: Point, psi: float):
        """
        If two points i and j should not be separated with probability more than psi, then this function can be used to add this constraint.
        Args:
            i (Point): Index of first point.
            j (Point): Index of second point.
            psi (float): Stochastic separation bound.
        """
        i, j = min(i, j), max(j, i)
        if (i, j) in self._stochastic_constraints and psi < self._stochastic_constraints[(i, j)]:
            self._stochastic_constraints[(i, j)] = psi
        elif (i, j) not in self._stochastic_constraints:
            self._stochastic_constraints[(i, j)] = psi

    @property
    def num_points(self):
        return self._dist_matrix.num_points

    @property
    def dist_matrix(self):
        return self._dist_matrix.dist_matrix

    @property
    def k(self):
        return self._k

    @property
    def original_points(self):
        return self._original_points

    @property
    def objective(self):
        return self._objective

    @property
    def stochastic_constraints(self):
        return self._stochastic_constraints


def BinarySearchKCenter(clustering_problem: ClusteringProblem) -> Tuple[Dict[Tuple[int, Point], float], float, List[int]]:
    """
    Runs binary search with the k-center problem to determine an optimal tau_star for the clustering problem we care about (which has stochastic pairwise constraints).
    Args:
        clustering_problem (ClusteringProblem): Clustering problem we care about.

    Returns:
        Tuple[int, Point], float], float, List[int]: The probabalistic clustering returned by the LP using the optimal tau_star, the optimal tau_star value, and the indices of the centers with respect to the original points.
    """
    k = clustering_problem.k
    # Grab the distances to run binary search with for the optimal radius of the constrained problem
    dist_matrix = clustering_problem.dist_matrix
    distances, _ = ExtractCostsAdjRepr(dist_matrix)

    lo = 0
    hi = len(distances)-1

    # Store the information corresponding to the best (smallest tau_star) feasible LP seen so far
    best_xvals = dict()
    best_tau_star = max(distances)
    best_S = set()
    best_center_indices = []

    while (lo < hi):
        mid = math.floor((hi+lo)/2)
        tau_star = distances[mid]
        S = set()
        U = set(range(clustering_problem.num_points))

        # Greedily choose centers
        while U:
            j = U.pop()
            S.add(j)
            U_list = list(U)
            new_U_list = [
                j_prime for j_prime in U_list if dist_matrix[j, j_prime] > 2*tau_star]
            U = set(new_U_list)

        # If the number of centers is less than the number of centers we are allowed
        if len(S) <= k:

            # Grab the indices of the centers
            center_indices = list(S)

            # Assign points to centers based on which center is closest
            assignment = -1*np.ones(clustering_problem.num_points, dtype=int)
            for point in range(clustering_problem.num_points):
                best_center = np.argmin(
                    dist_matrix[point, center_indices])
                assignment[point] = best_center

            xvals = SPCLP(clustering_problem, center_indices=center_indices,
                          tau_star=tau_star)

            # As noted in SPCLP, xvals is empty if the LP is not feasible
            lp_feasible = bool(xvals)

            # If the LP is feasible, we update hi to mid since this is an upper bound on the optimal tau_star
            # and we update all the other clustering attributes that we care about
            if lp_feasible:
                hi = mid
                best_xvals = xvals
                best_tau_star = tau_star
                best_center_indices = center_indices
                continue

        lo = mid+1  # If we reach here this means that mid is to small to be the optimal tau_star
    return best_xvals,  best_tau_star, best_center_indices


def KTRound(k: int, num_points: int, xvals: Dict[Tuple[int, Point], float]) -> np.array:
    """
    Implementation of the KTRound algorithm. Returns an assignment of points to centers.
    Args:
        k (int): Number of clusters.
        num_points (int): Number of total points.
        xvals (Dict[Tuple[int, Point], float]): Stochastic assignment of points to centers to be used by the routine to generate assignments.

    Returns:
        np.array: each entry corresponds to the label of the center to which it was assigned; these labels are with respect to k
    """
    assignment = -1 * np.ones(num_points, dtype=int)
    unassigned = set(range(num_points))

    while len(unassigned):
        label = np.random.randint(k)
        y = np.random.uniform()
        for v in list(unassigned):
            if y < xvals[(label, v)]:
                assignment[v] = label
                unassigned.remove(v)

    return assignment


def ComputeSPCObjective(assignment: np.array, original_points: np.array = None, centers: np.array = None, dist_matrix: np.array = None, center_indices: List[int] = None, objective: Objective = Objective.KMEANS) -> float:
    """
    Given an assignment of points to centers, the vectors representing the original points in the metric space, and the centers in the case of the k-means objective
    or an assignment of points to centers, a dist_matrix with the distances between the points, and a set of indices corresponding to the the points used as centers in the case of the k-center objective,
    returns a value representing the objective corresponding to the chosen clustering.
    Args:
        assignment (np.array): Assignment of points to centers (the ith entry represents the label of the center the ith point has been assigned to)
        original_points (np.array, optional): Only required if k-means is the objective; used for calculating Euclidean distances (we need the points in the original metric space). Defaults to None.
        centers (np.array, optional): Only required if k-means is the objective; used for calculating Euclidean distances (we need the centers in the original metric space). Defaults to None.
        dist_matrix (np.array, optional): Only required if k-center is the objective; we use scaled (Euclidean) distances for consistency with the Scr clustering radius calculation used for creating the pairwise constraints. Defaults to None.
        center_indices (List[int], optional): Only required if k-center is the objective; we need this to able to use the dist_matrix and compute the max radius among all clusters. Defaults to None.
        objective (Objective, optional): [description]. Defaults to Objective.KMEANS.

    Returns:
        float: the computed objective
    """

    # If we calculate the objective for k-means, we have to use the original points in the metric space that is used for Lloyd's algorithm
    if objective == Objective.KMEANS:
        assert original_points is not None
        assert centers is not None
        obj = 0
        reverse_dict = dict()
        for point, center in enumerate(assignment):
            try:
                reverse_dict[center].append(point)
            except:
                reverse_dict[center] = [point]

        for center in reverse_dict:
            obj += np.linalg.norm(original_points[reverse_dict[center]] -
                                  centers[center])**2
    else:
        # clustering_problem.objective == Objective.KCENTER:
        assert dist_matrix is not None
        obj = 0
        for point, center in enumerate(assignment):
            obj = max(obj, dist_matrix[point, center_indices[center]])
    return obj


def SPCLP(clustering_problem: ClusteringProblem, chosen_centers: np.array = None, center_indices: List[int] = None,  tau_star: float = None) -> Dict[Tuple[int, Point], float]:
    """
    Solves the LP described in Algorithm 1 or Algorithm 2 depending on which objective is being used.

    Args:
        clustering_problem (ClusteringProblem): The specification of problem.
        chosen_centers (np.array, optional): Only needs to be specified if solving k-means problem. Corresponds to the centers in the original metric space. Defaults to None.
        center_indices (List[int], optional): Only needs to be specified if solving k-center porblem. Corresponds to the indices of the points that are chosen as centers. Defaults to None.
        tau_star (float, optional): Only needs to be specified if solving for k-center problem (this is solved for as part of the binary search routine). Defaults to None.

    Returns:
        Dict[Tuple[int, Point], float]: A dictionary corresponding to the solutions to the x variable (to be used in the KT-Round routine to compute clusterings).
    """

    original_points = clustering_problem.original_points
    dist_matrix = clustering_problem.dist_matrix

    if clustering_problem.objective == Objective.KMEANS:
        assert chosen_centers is not None
        num_centers = len(chosen_centers)
    else:
        # clustering_problem.objective == Objective.KCENTER
        assert center_indices is not None
        assert tau_star is not None
        num_centers = len(center_indices)

    with gp.Env("gurobi.log") as env:
        with gp.Model("SPC", env=env) as m:
            all_edges = set()
            for edge in clustering_problem.stochastic_constraints:
                all_edges.add(edge)
            try:
                x = dict()
                for center_idx in range(num_centers):
                    x[center_idx] = list()
                    for point in range(clustering_problem.num_points):
                        x[center_idx].append(m.addVar(ub=1.0,
                                                      name=f"center_{center_idx}_point_{point}"))

                z_edge = dict()
                z_edge_chosen = dict()
                for edge in all_edges:
                    z_edge[edge] = m.addVar(ub=1.0, name=f"z_edge_{edge}")
                    z_edge_chosen[edge] = dict()
                    for center_idx in range(num_centers):
                        z_edge_chosen[edge][center_idx] = m.addVar(
                            ub=1.0, name=f"z_edge_chosen_{edge}_center_{center_idx}")

                if clustering_problem.objective == Objective.KMEANS:
                    obj = gp.LinExpr()
                    for v in range(clustering_problem.num_points):
                        for center_idx, center in enumerate(chosen_centers):
                            obj += x[center_idx][v] * \
                                (np.linalg.norm(
                                    center - original_points[v]) ** 2)
                    m.setObjective(obj, GRB.MINIMIZE)
                else:
                    for center_idx, global_center_idx in enumerate(center_indices):
                        m.addConstr(x[center_idx][global_center_idx] == 1)

                    for center_idx, global_center_idx in enumerate(center_indices):
                        for point in range(clustering_problem.num_points):
                            if dist_matrix[global_center_idx, point] > 3*tau_star:
                                m.addConstr(
                                    x[center_idx][point] == 0)

                for j in range(clustering_problem.num_points):
                    s = gp.LinExpr()
                    for center_idx in range(num_centers):
                        s += x[center_idx][j]
                    m.addLConstr(s, GRB.EQUAL, 1)

                for edge in all_edges:
                    for center_idx in range(num_centers):
                        m.addLConstr(z_edge_chosen[edge][center_idx] >=
                                     x[center_idx][edge[0]] - x[center_idx][edge[1]])
                        m.addLConstr(z_edge_chosen[edge][center_idx] >=
                                     x[center_idx][edge[1]] - x[center_idx][edge[0]])

                for edge in all_edges:
                    s = gp.LinExpr()
                    for center_idx in range(num_centers):
                        s += z_edge_chosen[edge][center_idx]

                    m.addLConstr(z_edge[edge], GRB.EQUAL, 0.5*s)

                m.Params.BarConvTol = 1e-5
                for edge, psi in clustering_problem.stochastic_constraints.items():
                    s = gp.LinExpr()
                    s += z_edge[edge]
                    m.addLConstr(s, GRB.LESS_EQUAL, psi)

                m.optimize()

                xvals = dict()

                # Check to make sure LP is feasible and bounded
                if m.status != 3 and m.status != 4 and m.status != 5:
                    for center_idx in range(num_centers):
                        for v in range(clustering_problem.num_points):
                            xvals[(center_idx, v)] = x[center_idx][v].x

                return xvals  # if empty this means that the LP was infeasible

            except gp.GurobiError as e:
                print('Error code ' + str(e.errno) + ': ' + str(e))

            except AttributeError:
                print('Encountered an attribute error')


def SPCAlgorithm(clustering_problem: ClusteringProblem,  num_trials: int = 5000) -> Tuple[Dict[Tuple[int, Point], float], np.array, np.array, Union[np.array, List[int]]]:
    """
    Runs Algorithm 1 or Algorithm 2 based on the objective we are concerned with.
    Args:
        clustering_problem (ClusteringProblem): The original clustering problem.
        num_trials (int, optional): The number of clusterings we want to generate. Defaults to 5000.

    Returns:
        Tuple[Dict[Tuple[int, Point], float], np.array, np.array, Union[np.array, List[int]]]: The solutions to the x variable in the LP (so that more assignments can be created later if desired), 
        all assignments generated, the associated objectives of these assignments, and a representation of the centers (in the original metric space for k-means, or the indices if for k-center).
    """
    k = clustering_problem.k

    if clustering_problem.objective == Objective.KMEANS:
        chosen_centers, original_obj, _ = KMeansClustering(
            clustering_problem)

    if clustering_problem.objective == Objective.KMEANS:
        xvals = SPCLP(clustering_problem,
                      chosen_centers=chosen_centers)
        centers = chosen_centers
    else:
        # clustering_problem.objective == Objective.KCENTER
        xvals, tau_star, center_indices = BinarySearchKCenter(
            clustering_problem)
        centers = center_indices

    all_assignments = -1 * \
        np.ones((num_trials, clustering_problem.num_points), dtype=int)
    assignment_objs = np.zeros(num_trials)

    for trial in range(num_trials):
        all_assignments[trial] = KTRound(
            len(centers), clustering_problem.num_points, xvals)

        if clustering_problem.objective == Objective.KMEANS:
            assignment_obj = ComputeSPCObjective(
                all_assignments[trial], original_points=clustering_problem.original_points, centers=chosen_centers, objective=clustering_problem.objective)
        else:
            assignment_obj = ComputeSPCObjective(
                all_assignments[trial], dist_matrix=clustering_problem.dist_matrix, center_indices=center_indices,  objective=clustering_problem.objective)

        assignment_objs[trial] = assignment_obj

    return xvals, all_assignments, assignment_objs, centers


def KMeansClustering(clustering_problem: ClusteringProblem) -> Tuple[np.array, float, np.array]:
    """
    Implementation of Lloyd's algorithm.
    Args:
        clustering_problem (ClusteringProblem): The original clustering problem which provides the points to run Lloyd's algorithm on.

    Returns:
        Tuple[np.array, float, np.array]: The points in the metric space corresponding to the points closest to the centers computed by the algorithm, the objective resulting from using these points as centers, and the assignments of the points to these centers.
    """
    k = clustering_problem.k
    original_points = clustering_problem.original_points
    kmeans = KMeans(k, random_state=42).fit(original_points)

    obj = 0
    centers = []

    for label in range(k):
        clustered_locations = np.ndarray.flatten(
            np.where(kmeans.labels_ == label)[0])
        clustered_points = original_points[clustered_locations]

        center = kmeans.cluster_centers_[label]
        distance_to_center = distance.cdist(
            original_points, np.reshape(center, (1, -1)))
        best_center = np.argmin(distance_to_center)
        cluster_distance_to_center = distance.cdist(
            clustered_points, np.reshape(original_points[best_center], (1, -1)))
        centers.append(best_center)
        obj += np.sum(cluster_distance_to_center**2)

    cluster_centers = original_points[centers]
    return cluster_centers, obj, kmeans.labels_


def EmpiricalProbabilities(clustering_problem: ClusteringProblem, all_assignments: np.array) -> Dict[Tuple[Point, Point], float]:
    """
    Calculates the empirical probability of separation between points that have pairwise constraints specified in the problem.
    Args:
        clustering_problem (ClusteringProblem):  The clustering problem we care about.
        all_assignments (np.array):  The assignments created by an algorithm for which we want to measure the empirical probabilities.

    Returns:
        Dict[Tuple[Point, Point], float]: A dictionary for each of the links corresponding to this empirical probablities based on the clusterings in all_assignments.
    """

    stochastic_constraints = clustering_problem.stochastic_constraints
    empirical_probabilities = dict()
    violations = 0
    count = 0
    for link in stochastic_constraints:
        v1, v2 = link
        prob = sum(all_assignments[:, v1] !=
                   all_assignments[:, v2])/len(all_assignments)
        empirical_probabilities[link] = prob
    return empirical_probabilities


def IndependentRounding(distribution: np.array, num_trials: int = 5000) -> np.array:
    """
    Given a distribution over centers for each point to be clustered, generates sample clusterings by sampling independently from each distribution for each point.
    Used for comparisons to (Anderson et al. 2020).

    Args:
        distribution (np.array): Stochastic clustering of points to centers, to be used by the rounding procedure. 
        num_trials (int, optional): Number of clusterings to run. Defaults to 5000.

    Returns:
        np.array: All num_trial assignments created.
    """
    all_assignments = -1 * \
        np.ones((num_trials, distribution.shape[0]), dtype=int)
    for idx, row in enumerate(distribution):
        sampling = np.random.multinomial(1, row/row.sum(), size=num_trials)
        sampling = np.argmax(sampling, axis=1)
        all_assignments[:, idx] = sampling
    return all_assignments

# Implementation in (Brubach et al. 2020) of Scr algorithm


def ExtractCostsAdjRepr(dist_matrix: np.array) -> Tuple[List[float], List[Tuple[int, int, float]]]:
    """
    Extract the desired representation in order to run the Scr algorithm.
    """
    n = np.shape(dist_matrix)[0]

    indices = [(x, y) for x in range(n) for y in range(n)]
    flattened = list(dist_matrix.reshape(dist_matrix.size))
    combined = list(zip(indices, flattened))
    adj_rep = [(x, y, z) for ((x, y), z) in combined]

    costs = sorted(list(set([z for (x, y, z) in adj_rep])))
    # don't want to include "self edges" so we remove the zero cost edge
    costs = costs[1:]

    return costs, adj_rep


def PruneGraph(graph: np.array, prune_cost: float) -> np.array:
    """
    Prune the graph to only include edges with costs lower than prune_cost.
    """
    filtered_graph = [(x, y, z) for (x, y, z) in graph if z <= prune_cost]
    return filtered_graph


def SortByVertex(pruned: np.array) -> Dict[int, List[Tuple[int, float]]]:
    """
    Sorts graph by vertex.
    """
    sortkeyfn = itemgetter(0)
    result = {}
    for key, valuesiter in groupby(pruned, key=sortkeyfn):
        result[key] = list((v[1], v[2]) for v in valuesiter)
    return result


def Scr(graph_dict: Dict[int, List[Tuple[int, float]]]):
    """
    Runs Scr algorithm.
    """
    CovCnt = {}
    for key in graph_dict.keys():
        CovCnt[key] = len(graph_dict[key])
    score = deepcopy(CovCnt)
    V = len(CovCnt)
    D = []
    for i in range(V):
        next_vertex = min(score, key=score.get)
        vertexFound = False
        for (y, dist) in graph_dict[next_vertex]:
            if CovCnt[y] == 1:
                vertexFound = True
                break
        if vertexFound:
            D.append(next_vertex)
            for (y, dist) in graph_dict[next_vertex]:
                CovCnt[y] = 0
        else:
            for (y, dist) in graph_dict[next_vertex]:
                if CovCnt[y] > 0:
                    CovCnt[y] -= 1
                    score[y] += 1
        score[next_vertex] = np.PINF

    return D


def kCenterScr(dist_matrix: np.array, k: int):
    """
    The k-center version of the Scr algorithm.
    """
    costs, adj_rep = ExtractCostsAdjRepr(dist_matrix)

    for cost_idx, cost in enumerate(costs):
        pruned = PruneGraph(adj_rep, cost)
        sorted_V = SortByVertex(pruned)
        D = Scr(sorted_V)
        if len(D) <= k:
            centers = np.array(D)
            relevant_distances = dist_matrix[:, D]
            center_assignments = centers[np.argmin(relevant_distances, axis=1)]
            radii = np.amin(relevant_distances, axis=1)
            max_radius = np.amax(radii)
            clusters = list(set(center_assignments))
            return centers, max_radius, radii, center_assignments
