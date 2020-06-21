import pandas as pd
from sklearn import model_selection
from mml_1sme import MML_1SmE

import ml_utils as mlu

output_path = '../out_1sme_v2_testing/'
prefix = 'exp'
suf_dataForML = '.dataForML.h5'
hdf_key = 'dataForML'

test_percent = 0.1
seed = 2020

algorithms = ["RandomForestClassifier", "LogisticRegression", "SGDClassifier"]
metric = "Balanced_Accuracy"
tune_iterations = 10
manager_mml = MML_1SmE()
trained_mean_sufix = '.trainedMean.txt'

means = None

for i in range(5):
    manager_mml.gen_experts(path=output_path, expert_names=algorithms, prefix=prefix + '_train', metric_max=metric, seed=seed, metric_tune=metric, max_tune=tune_iterations)
    meanss = []
    for alg in algorithms:
        trained_mean_file = output_path + alg + trained_mean_sufix
        file = open(trained_mean_file, "r")
        trained_mean = float(file.read())
        file.close()

        meanss.append(trained_mean)

    if means is None or means == meanss:
        means = meanss
    else:
        print('Se hace mal')
        print(means)
        print(meanss)
