#################################################
# IMPORTS
#################################################
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats
import random
import os
import sys
import joblib

from sklearn import model_selection
from sklearn import metrics

#################################################
# Functions
#################################################


def change_data_path(prefix, output_path):
    features = "_approx_feature_importance.txt"
    dataForML = ".dataForML.h5"
    variants_and_alleles = ".variants_and_alleles.tab"
    dataForML_train = "_train.dataForML.h5"
    dataForML_test = "_test.dataForML.h5"

    print(f"Moving files from {prefix} to {output_path}")
    # os.system(f"ls -l {prefix}*")

    # Move data files to directory
    # features
    source = prefix + features
    destination = output_path + features
    command = f"mv {source} {destination}"
    os.system(command)

    # dataForML
    source = prefix + dataForML
    destination = output_path + dataForML
    command = f"mv {source} {destination}"
    os.system(command)

    # variants_and_alleles
    source = prefix + variants_and_alleles
    destination = output_path + variants_and_alleles
    command = f"mv {source} {destination}"
    os.system(command)

    # dataForML_train
    source = prefix + dataForML_train
    destination = output_path + dataForML_train
    command = f"mv {source} {destination}"
    os.system(command)

    # dataForML_test
    source = prefix + dataForML_test
    destination = output_path + dataForML_test
    command = f"mv {source} {destination}"
    os.system(command)

    # os.system(f"ls -l {output_path}*")


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


def generate_lvl2_expert(output_prefix):
    algs = "LogisticRegression RandomForestClassifier SGDClassifier"
    actual_path = path2output + output_prefix + "/"
    # Load test partition
    test_data = pd.read_hdf(actual_path + "test_MML_test.dataForML.h5", key="dataForML")

    # Predict with experts
    meta_set = pd.DataFrame(test_data['ID'])
    for alg in algorithms:
        actual_path_alg = path2output + output_prefix + "/" + alg + "/"
        # Load level 1 expert
        if os.path.isfile(actual_path_alg + "test_MML.tunedModel.joblib"):
            expert = joblib.load(actual_path_alg + "test_MML.tunedModel.joblib")
        else:
            expert = joblib.load(actual_path_alg + "test_MML.trainedModel.joblib")

        meta_set["Pred-" + alg] = expert.predict(test_data.drop(['PHENO', 'ID'], 1))

    # Merge data
    meta_set = meta_set.merge(test_data, on='ID')
    # Save it
    meta_set.to_hdf(actual_path + "test_MML_merged.dataForML.h5", key="dataForML")
    # Train level 2 expert
    command = f'''
    GenoML discrete supervised train \
    --prefix {actual_path + "test_MML_merged"} \
    --metric_max {metric} \
    --alg  {algs} \
    --seed {seed}
    '''

    os.system(command)
    # Tune level 2 expert
    command = f'''
    GenoML discrete supervised tune \
    --prefix {actual_path + "test_MML_merged"} \
    --metric_tune {metric} \
    --max_tune {tune_iterations} \
    --seed {seed}
    '''

    os.system(command)


def prediction_test(test_df):
    print("#" * 50)
    print("Predict test partition")
    Y_test = test_df['PHENO']

    meta_set = pd.DataFrame(test_df['ID'])
    for alg in algorithms:
        actual_path_alg = actual_path + alg + "/"

        # Load level 1 expert
        if os.path.isfile(actual_path_alg + prefix + ".tunedModel.joblib"):
            expert = joblib.load(actual_path_alg + prefix + ".tunedModel.joblib")
        else:
            expert = joblib.load(actual_path_alg + prefix + ".trainedModel.joblib")

        meta_set["Pred-" + alg] = expert.predict(test_df.drop(['PHENO', 'ID'], 1))

    # Merge data
    meta_set = meta_set.merge(test_df, on='ID')

    # Reorder columns
    df_meta = pd.read_hdf(actual_path + "test_MML_merged" + ".dataForML.h5", key='dataForML')
    meta_set = meta_set[list(df_meta.columns)]

    model = None

    if os.path.isfile(actual_path + prefix + ".tunedModel.joblib"):
        model = joblib.load(actual_path + "test_MML_merged" + ".tunedModel.joblib")
    else:
        model = joblib.load(actual_path + "test_MML_merged" + ".trainedModel.joblib")

    test_predictions = model.predict(meta_set.drop(['PHENO', 'ID'], 1))
    balacc = metrics.balanced_accuracy_score(Y_test, test_predictions)
    return balacc


#################################################
# Parameters
#################################################


# Global parameters
n_repeats = 5
path2output = "/home/edusal/pythonNotebooks/Experiments/Notebook_v2/MMLOutput/"
prefix = "test_MML"

# Parameters for munging
genoTrain = "SPANISH"  # Cohorte a estudiar
path2Data = "/home/edusal/data/FINALES/"  # all data in the same folder (geno and covs)
path2Pheno = "/home/edusal/data/MAF0.05/SPAIN/MyPhenotype.genoml.pheno"
features = 50
test_size = 0.3

# Parameters for train and tune
algorithms = ["RandomForestClassifier", "LogisticRegression", "SGDClassifier", "QuadraticDiscriminantAnalysis"]
metric = "Balanced_Accuracy"
tune_iterations = 10

# Parameters for calculate mean
meta_means = []
test_results = []
meta_tuned_mean_file = 'test_MML_merged.tunedMean.txt'
meta_trained_mean_file = 'test_MML_merged.trainedMean.txt'
columns = ['Iteration', 'Algorithm', 'Trained Mean', 'Tuned Mean', 'Test score']
results_df = pd.DataFrame(columns=columns)
columns = ['Mean', 'Lower confidence-interval', 'Upper confidence-interval']
out_df = pd.DataFrame(columns=columns)
columns = ['Iteration', 'Expert', 'Trained Mean', 'Tuned Mean']
experts_res_df = pd.DataFrame(columns=columns)
tuned_mean_sufix = '.tunedMean.txt'
trained_mean_sufix = '.trainedMean.txt'

# Set the random's seed
random.seed(20)

#################################################
# Generate Munged data (if it is not generated) #
#################################################

if not os.path.isfile(path2output + prefix + '.dataForML.h5'):
    command = f'''
    GenoMLMunging \
    --prefix {path2output + prefix} \
    --datatype d \
    --geno {path2Data + genoTrain} \
    --pheno {path2Pheno} \
    --featureSelection {features}
    '''

    os.system(command)
else:
    print("Munging already done. Munged file is: " + path2output + prefix + ".dataForML.h5\n")


genoTrain_df = pd.read_hdf(path2output + prefix + ".dataForML.h5", key="dataForML")

print("####################################")
print("Full data set:")
print(genoTrain_df)

case_df = genoTrain_df[genoTrain_df.PHENO == 0]
control_df = genoTrain_df[genoTrain_df.PHENO == 1]

case_df.index = range(len(case_df.index))
control_df.index = range(len(control_df.index))

values = list(case_df.index)
case_test_index = random.sample(values, 200)
values = list(control_df.index)
control_test_index = random.sample(values, 200)

test_df = case_df.iloc[case_test_index, :].append(control_df.iloc[case_test_index, :])
test_df = test_df.sample(frac=1).reset_index(drop=True)

print("####################################")
print("Test data set:")
print(test_df)

genoTrain_df = genoTrain_df[~genoTrain_df.ID.isin(list(test_df.ID))]
genoTrain_df.index = range(len(genoTrain_df.index))

print("####################################")
print("Train data set:")
print(genoTrain_df)

genoTrain_df.to_hdf(path2output + prefix + ".dataForML.h5", key="dataForML")

#################################################
# Repeat meta learning 'n_repeats' times
#################################################

# Read file
dataForML_file = path2output + prefix + ".dataForML.h5"
dataForML = pd.read_hdf(dataForML_file, key='dataForML')


for n in range(n_repeats):
    output_prefix = prefix + "_" + str(n)

    # Create output directory
    try:
        os.mkdir(path2output + output_prefix)
    except FileExistsError:
        continue

    # Set seed
    seed = random.randint(1, 1000)

    # Divide train-test datasets
    dataForML_train, dataForML_test = model_selection.train_test_split(dataForML, test_size=test_size, random_state=seed)

    # Save all files
    dataForML_train_prefix = "test_MML_train"
    dataForML_test_prefix = "test_MML_test"

    dataForML_train.to_hdf(path2output + dataForML_train_prefix + ".dataForML.h5", key="dataForML")
    dataForML_test.to_hdf(path2output + dataForML_test_prefix + ".dataForML.h5", key="dataForML")

    # Move data
    actual_path = path2output + output_prefix + "/"
    change_data_path(path2output + prefix, actual_path + prefix)

    # Train level 1 experts
    for alg in algorithms:
        # Create directory
        try:
            os.mkdir(actual_path + alg)
        except FileExistsError:
            pass

        actual_path_alg = actual_path + alg + "/"

        # Move data to directory and change name
        change_data_path(actual_path + prefix, actual_path_alg + prefix)

        # Train model
        command = f'''
        GenoML discrete supervised train \
        --prefix {actual_path_alg + prefix} \
        --metric_max {metric} \
        --alg  {alg} \
        --seed {seed}
        '''

        os.system(command)

        # Tune model
        command = f'''
        GenoML discrete supervised tune \
        --prefix {actual_path_alg + prefix} \
        --metric_tune {metric} \
        --max_tune {tune_iterations} \
        --seed {seed}
        '''

        os.system(command)

        change_data_path(actual_path_alg + prefix, actual_path + prefix)

    # Generate level 2 expert
    generate_lvl2_expert(output_prefix)

    test_results.append(prediction_test(test_df))

    # Change output names and directory
    change_data_path(actual_path + prefix, path2output + prefix)


# Get balanced accuracy n_repeats times
for n in range(n_repeats):
    output_prefix = prefix + "_" + str(n)
    # Load meta mean

    rep_prefix = prefix + "_" + str(n)
    actual_path = path2output + rep_prefix + '/'

    tuned_mean_file = actual_path + "test_MML_merged" + tuned_mean_sufix
    trained_mean_file = actual_path + "test_MML_merged" + trained_mean_sufix
    best_algo_file = actual_path + "test_MML_merged" + ".best_algorithm.txt"

    # Load meta mean
    trained_mean = None
    tuned_mean = None
    best_algo = None

    if os.path.isfile(tuned_mean_file):
        file = open(tuned_mean_file, "r")
        tuned_mean = float(file.read()) * 100
        file.close()
    else:
        print("File does not exist: " + tuned_mean_file)

    if os.path.isfile(trained_mean_file):
        file = open(trained_mean_file, "r")
        trained_mean = float(file.read())
        file.close()
    else:
        print("File does not exist: " + trained_mean_file)

    if os.path.isfile(best_algo_file):
        file = open(best_algo_file, "r")
        best_algo = file.read()
        file.close()
    else:
        print("File does not exist: " + best_algo_file)

    if tuned_mean is None:
        mean = trained_mean
    else:
        mean = tuned_mean

    # Add balanced accuracy to balanced accuracy list
    meta_means.append(mean)

    # Save results in the df
    entry = {'Iteration': n,
             'Algorithm': best_algo,
             'Trained Mean': trained_mean,
             'Tuned Mean': tuned_mean,
             'Test score': test_results[n]}
    results_df = results_df.append(entry, ignore_index=True)

    for alg in algorithms:
        actual_path_alg = actual_path + alg + "/"

        tuned_mean_file = actual_path_alg + prefix + tuned_mean_sufix
        trained_mean_file = actual_path_alg + prefix + trained_mean_sufix

        if os.path.isfile(tuned_mean_file):
            file = open(tuned_mean_file, "r")
            tuned_mean = float(file.read()) * 100
            file.close()
        else:
            print("File does not exist: " + tuned_mean_file)

        if os.path.isfile(trained_mean_file):
            file = open(trained_mean_file, "r")
            trained_mean = float(file.read())
            file.close()
        else:
            print("File does not exist: " + trained_mean_file)

        entry = {'Iteration': n,
                 'Expert': alg,
                 'Trained Mean': trained_mean,
                 'Tuned Mean': tuned_mean}
        experts_res_df = experts_res_df.append(entry, ignore_index=True)


print(meta_means)


# Calculate mean and confidence interval
mean, conf_down, conf_up = mean_confidence_interval(meta_means)

entry = {'Mean': mean,
         'Lower confidence-interval': conf_down,
         'Upper confidence-interval': conf_up}
out_df = out_df.append(entry, ignore_index=True)

# Save results
out_file = prefix + '.mean_confidence_interval.txt'
file = open(path2output + out_file, "w+")
file.write(f'{mean} {conf_down} {conf_up}')
file.close()


# Show results
print()
print('#' * 60)
print(f"Results in training phase are:\n\t- Mean: {mean:.4f}\n\t- Conficence interval: {conf_down:.4f} - {conf_up:.4f}")
print('#' * 60)

mean, conf_down, conf_up = mean_confidence_interval(test_results)

print()
print('#' * 60)
print(f"Results in testing phase are:\n\t- Mean: {mean:.4f}\n\t- Conficence interval: {conf_down:.4f} - {conf_up:.4f}")
print('#' * 60)

out_df.to_csv(path2output + prefix + '.results.csv')
results_df.to_csv(path2output + prefix + '.all_means.csv')
experts_res_df.to_csv(path2output + prefix + '.experts.csv')

print(test_results)
