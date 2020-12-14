import argparse
import os
import sys
from metrics import *
from spc import SPCAlgorithm, EmpiricalProbabilities, KMeansClustering
import pickle
from enum import IntEnum


class DatasetType(IntEnum):
    BANK = 1
    ADULT = 2
    CREDITCARD = 3


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the k-means experiment.')
    parser.add_argument("--metric", dest="metric", type=int, default=2)
    parser.add_argument("--clusters", nargs="+",
                        type=int, default=[4, 6, 8, 10])
    parser.add_argument("--run_analysis", action="store_true")
    parser.add_argument("--probs_precalculated", action="store_true")
    parser.add_argument("--sample_file", dest="sample_file",
                        type=str, default="data/adult.pkl")
    parser.add_argument("--output_directory", dest="output_directory",
                        type=str, default=None)
    parser.add_argument("--size", type=int, default=500)
    parser.add_argument("--eps", type=float, default=0.05)
    args = parser.parse_args()

    if args.metric == 3:
        metric = Metric.F3
    else:
        metric = Metric.F2

    clusters = args.clusters
    sample_file = args.sample_file
    run_analysis = args.run_analysis
    probs_precalculated = args.probs_precalculated
    eps = args.eps
    size = args.size

    with open(sample_file, "rb") as f:
        data, _ = pickle.load(f)
        data = data[:size, :size]

    if "adult" in sample_file:
        dataset_type = DatasetType.ADULT
    elif "bank" in sample_file:
        dataset_type = DatasetType.BANK
    elif "creditcard" in sample_file:
        dataset_type = DatasetType.CREDITCARD
    else:
        dataset_type = DatasetType.ADULT
        print("BAD SAMPLE FILE! Might cause errors.",
              file=sys.stderr, flush=True)

    if args.output_directory:
        output_directory = args.output_directory
    else:
        output_directory = f"{dataset_type.name.lower()}_assignments/metric_f{metric.value}"

    output_filename = f"{sample_file.split('/')[-1].split('.pkl')[0]}_assignment"
    os.makedirs(output_directory, exist_ok=True)

    for k in clusters:
        clustering_problem = GenerateClusteringProblem(
            data, k, similarity_metric=metric, m=100)
        stochastic_constraints = clustering_problem._stochastic_constraints
        if run_analysis:
            with open(f"{output_directory}/{output_filename}_{k}", "rb") as f:
                results = pickle.load(
                    f)
                assert(len(results) == 4)
                xvals, all_assignments, assignment_objs, centers = results

            kmeans_centers, kmeans_obj, kmeans_assignment = KMeansClustering(
                clustering_problem)
            algorithm_cost = np.average(assignment_objs)

            if not probs_precalculated:
                empirical_probabilities = EmpiricalProbabilities(
                    clustering_problem, all_assignments)
                with open(f"{output_directory}/{output_filename}_probabilities_{k}", "wb") as g:
                    pickle.dump(empirical_probabilities, g)
            else:
                with open(f"{output_directory}/{output_filename}_probabilities_{k}", "rb") as g:
                    empirical_probabilities = pickle.load(g)

            violations = 0
            for constraint in stochastic_constraints:
                if empirical_probabilities[constraint] > stochastic_constraints[constraint] + eps:
                    violations += 1

            with open(f"{output_directory}/{output_filename}_probabilities_{k}_analysis.log", "w+") as f:
                f.write(f"Violations with threshold {eps}: {violations}\n")
                f.write(
                    f"Fractional violations: {violations/len(stochastic_constraints)}\n")
                f.write(f"Lloyd's algorithm cost: {kmeans_obj}\n")
                f.write(f"Algorithm cost: {algorithm_cost}\n")
                f.write(f"Cost of fairness: {algorithm_cost/kmeans_obj}\n")
        else:
            result_assn = SPCAlgorithm(
                clustering_problem)
            with open(f"{output_directory}/{output_filename}_{k}", "wb") as f:
                pickle.dump(result_assn, f)
