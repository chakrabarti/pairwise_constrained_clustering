This is the codebase for our paper **Fairness, Semi-Supervised Learning, and More: A General Framework for Clustering with Stochastic Pairwise Constraints**.

Overview
--------
There are two main components to the codebase: the implementation of the clustering algorithms, and the main files to execute experiments using these algorithms.

| Component                | Description                                                                    | Relevant files                                |
| ------------------------ | ------------------------------------------------------------------------------ | --------------------------------------------- |
| Algorithm Implementation | Implementation of Algorithms 1 and 2 from our paper                            | metrics.py, spc.py                            |
| Simulation               | Uses the algorithm implementation to generate clusterings and analyze fairness | k-means_experiment.py, k-center_experiment.py |

Dependencies
------------

* numpy==1.18.4
* scipy==1.4.1
* scikit-learn==0.22.1
* gurobipy==9.0.2


Data Preparation
----------------

We use (Anderson et al. 2020)'s codebase to prepare our data. The sample files we used for our experiments can be found in `data` (`data/adult.pkl, data/bank.pkl, data/creditcard.pkl`). These each contain 500 points (because that is the maximum number of points we use in our cluster computations).


Simulation
-------------------

This repository provides implementations for 2 different algorithms, Algorithms 1 and 2 from our paper used for the problems of k-means-PBS and k-center-PBS-CC respectively.

```bash
python k-means_experiment.py # Runs Algorithm 1 to compute 5000 clusterings for the k-means-PBS problem of data/adult.pkl and puts them in adult_assignments/metric_f2
python k-means_experiment.py --run_analysis # Runs the analysis of the 5000 clusterings of data/adult.pkl and puts these analysis outputs in adults_assignments/metric_f2

python k-center_experiment.py # Runs Algorithm 2 to compute 5000 clusterings for the k-center-PBS-CC problem of data/adult.pkl and puts them in adult_assignments/metric_f1
python k-center_experiment.py --run_analysis # Runs the analysis of the 5000 clusterings of data/adult.pkl and puts these analysis outputs in adults_assignments/metric_f1


```

| Clustering Problem | Experiment File        |
| ------------------ | ---------------------- |
| k-means-PBS        | k-means_experiment.py  |
| k-center-PBS-CC    | k-center_experiment.py |

All the algorithms have almost the same set of tunable parameters. Note that the dataset size used for all the algorithms have to be consistent across all the non OPT algorithms for inference script (next phase) to work. Since OPT-IF and OPT-CF are run on smaller datasets of size 80, ALG-IF and ALG-CF must be run on the same set so that the inference script can draw a comparison.

| Argument            | Functionality                                                                                                                                                                                                                                                                                                                                               | Values                                                             |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| sample_file         | Name of the sample dataset to be used for the experiment (if generating new samples, note that the sample dataset must start with the dataset type so that the main file can figure out the default name to be used for the output_directory)                                                                                                               | `data/bank.pkl`<br> `data/adult.pkl`<br> `data/creditcard.pkl`<br> |
| metric              | The type of similarity metric to be used to generate the fairness constraints (not an option for k-center since only F_1 is used for these experiments)                                                                                                                                                                                                     | 1 <br> 2 <br> 3                                                    |
| clusters            | The different values of k to be used for the clusterings                                                                                                                                                                                                                                                                                                    | List[Integer]                                                      |
| run_analysis        | Whether to generate the assignments (if flag is excluded) or to run fairness violation and objective cost calculations and produce output files (if flag is included)                                                                                                                                                                                       | BooleanFlag (False by default)                                     |
| probs_precalculated | If this flag is included, then the empirical probabilities are assumed to have been calculated already (saving the time of calculating these). Only relevant if run_analysis flag is included                                                                                                                                                               | Boolean Flag (False by default)                                    |
| output_directory    | Specifies where to dump the assignments from the simulations, and where the assignments have been dumped if running analysis of fairness violation and objective calculations. By default will be `<dataset>_assignments/metric_f<metric_type>` where `<dataset>` is either "bank", "adult", or "creditcard" and `<metric_type>` is either 1, 2, or 3       | String                                                             |
| size                | How many points to use from the sample dataset in the supplied sample_file (cannot exceed the number of total points in the dataset from sample_file; the provided sample files have 500 points)                                                                                                                                                            | Integer                                                            |
| eps                 | Specify the threshold allowed when measuring fairness violations on the stochastic pairwise constraints                                                                                                                                                                                                                                                     | Float                                                              |
| load_scr            | Only relevant for k-center_experiment.py. If this flag is included, loads the Scr assignments created from previously created scr_assignments (by default these are saved in ouptput_directory). If the flat is not included, the generated Scr assignments are saved to the output_directory so that they can be used again (when the flag is set to True) | Boolean Flag (False by default)                                    |
