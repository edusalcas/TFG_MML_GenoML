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
algs = "RandomForestClassifier LogisticRegression SGDClassifier"

metric = "Balanced_Accuracy"
tune_iterations = 10
manager_mml = MML_1SmE()
trained_mean_sufix = '.trainedMean.txt'

means = None

dataForML_test = pd.read_hdf(output_path + prefix + '_test' + suf_dataForML, key=hdf_key)

for i in range(5):
    manager_mml.gen_meta_expert(dataForML_test, output_path, algorithms, prefix + "_meta", metric, algs, seed, metric, tune_iterations)
    meanss = []
    for alg in algorithms:
        trained_mean_file = output_path + prefix + "_meta" + trained_mean_sufix
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
