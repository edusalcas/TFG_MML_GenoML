#################################################
# IMPORTS
#################################################
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats
import random
import os
import joblib

import sys

from sklearn import model_selection
from sklearn import metrics

#################################################
# Functions
#################################################


def change_data_path(input_path, output_path, input_prefix, output_prefix):
    print(f"Moving files from {prefix} to {output_path}")

    # features
    source = input_path + input_prefix + features
    destination = output_path + output_prefix + features
    command = f"mv {source} {destination}"
    os.system(command)

    # dataForML
    source = input_path + input_prefix + dataForML_sufix
    destination = output_path + output_prefix + dataForML_sufix
    command = f"mv {source} {destination}"
    os.system(command)

    # variants_and_alleles
    source = input_path + input_prefix + variants_and_alleles
    destination = output_path + output_prefix + variants_and_alleles
    command = f"mv {source} {destination}"
    os.system(command)


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


def fix_columns(df1, df2):
    aux = df2.copy()

    diff1 = set(df1.columns) - set(df2.columns)
    if len(diff1) > 0:
        new_cols = pd.DataFrame(0, index=np.arange(len(aux)), columns=diff1)

        aux.index = range(len(aux.index))
        aux = pd.concat([aux, new_cols], axis=1)

    diff2 = set(df2.columns) - set(df1.columns)
    if len(diff2) > 0:
        aux.drop(diff2, 1, inplace=True)

    return aux


def bindRows(dataFrames, columns=None):
    # If no columns indicated
    if columns is None:
        # Initialize df
        finalDataFrame = dataFrames[next(iter(dataFrames))]
        # For each data frame, merge it
        for key in dataFrames.keys():
            finalDataFrame = finalDataFrame.append(dataFrames[key])

        # Rename index
        finalDataFrame.index = range(len(finalDataFrame.index))
        # Return merged df
        return finalDataFrame
    else:
        # Initialize df
        finalDataFrame = dataFrames[next(iter(dataFrames))][columns]
        # For each data frame, merge it
        for key in dataFrames.keys():
            finalDataFrame = finalDataFrame.append(dataFrames[key][columns])

        # Rename index
        finalDataFrame.index = range(len(finalDataFrame.index))
        # Return merged df
        return finalDataFrame


def commonColumns(dataFrames):
    # Initialize commonColumns
    commonColumns = dataFrames[next(iter(dataFrames))].columns

    # For each df, intersect columns
    for key in dataFrames.keys():
        commonColumns = np.intersect1d(commonColumns, dataFrames[key].columns)

    # Return commonColumns
    return commonColumns


def generate_metaset(actual_path):
    meta_sets = {}
    pred_sets = {}

    # For each cohort, predict
    for genoTrain in genoTrains:
        dataTest = pd.read_hdf(actual_path + genoTrain + dataForML_test_sufix + ".dataForML.h5", key='dataForML')
        print(dataTest)
        meta_set = dataTest[['PHENO', 'ID']].copy()

        pred_sets[genoTrain] = dataTest.drop(columns=['PHENO', 'ID'])
        # Predict with every expert
        for expert_prefix in genoTrains:
            expert_dataTest = pd.read_hdf(actual_path + expert_prefix + dataForML_test_sufix + ".dataForML.h5", key='dataForML')
            if os.path.isfile(actual_path + expert_prefix + tuned_model_sufix):
                expert = joblib.load(actual_path + expert_prefix + tuned_model_sufix)
            else:
                expert = joblib.load(actual_path + expert_prefix + trained_model_sufix)

            # Fix columns for predict
            expert_dataTest = fix_columns(expert_dataTest, dataTest)
            expert_dataTest = expert_dataTest.drop(columns=['PHENO', 'ID'])

            # Predict
            preds = expert.predict(expert_dataTest)
            meta_set["Pred-" + expert_prefix] = preds

        meta_sets[genoTrain] = meta_set

    # Join predictions
    meta_set = bindRows(meta_sets)
    meta_set.index = range(len(meta_set.index))

    # If not private merge geno
    cols = commonColumns(pred_sets)

    # Merge all SNPs in one data frame
    geno_cols = bindRows(pred_sets, cols)
    geno_cols.index = range(len(geno_cols.index))

    # Merge predict data frame with geno data frame
    meta_set = pd.concat([meta_set, geno_cols], axis=1)

    meta_set.index = range(len(meta_set.index))

    print(meta_set)

    # Save meta-set
    meta_set.to_hdf(actual_path + meta_prefix + ".dataForML.h5", key='dataForML')


def prediction_test(df):
    print("Prediction test")
    meta_set = df[['ID', 'PHENO']].copy()
    print("Lets predict with experts")

    for expert_prefix in genoTrains:
        expert_dataTest = pd.read_hdf(actual_path + expert_prefix + dataForML_test_sufix + ".dataForML.h5", key='dataForML')
        if os.path.isfile(actual_path + expert_prefix + tuned_model_sufix):
            expert = joblib.load(actual_path + expert_prefix + tuned_model_sufix)
        else:
            expert = joblib.load(actual_path + expert_prefix + trained_model_sufix)

        # Fix columns for predict
        expert_dataTest = fix_columns(expert_dataTest, df)
        expert_dataTest = expert_dataTest.drop(columns=['PHENO', 'ID'])

        # Predict
        preds = expert.predict(expert_dataTest)
        meta_set["Pred-" + expert_prefix] = preds
        print("Predicted with " + expert_prefix)

    # Data set we used for train meta model
    df_meta = pd.read_hdf(actual_path + meta_prefix + ".dataForML.h5", key='dataForML')

    df = df.drop(columns=['PHENO', 'ID'])
    meta_set = pd.concat([meta_set, df], axis=1)

    print("Common columns: " + str(len(df_meta.columns.intersection(meta_set.columns))))

    # Fix columns for predict with meta model
    meta_set = fix_columns(df_meta, meta_set)

    # Reorder columns
    meta_set = meta_set[list(df_meta.columns)]

    print(df_meta)
    print(meta_set)

    model = None
    if os.path.isfile(actual_path + meta_prefix + tuned_model_sufix):
        model = joblib.load(actual_path + meta_prefix + ".tunedModel.joblib")
    else:
        model = joblib.load(actual_path + meta_prefix + ".trainedModel.joblib")
    Y_test = meta_set['PHENO']
    test_predictions = model.predict(meta_set.drop(columns=['PHENO', 'ID']))
    balacc = metrics.balanced_accuracy_score(Y_test, test_predictions)
    return balacc

#################################################
# Parameters
#################################################


# Global parameters
n_repeats = 5
path2output = "/home/edusal/pythonNotebooks/Experiments/Experimentacion/exp_MML_nSmE_output/"
prefix = "exp_nsme"

# Sufixes
features = "_approx_feature_importance.txt"
dataForML_sufix = ".dataForML.h5"
variants_and_alleles = ".variants_and_alleles.tab"
tuned_mean_sufix = '.tunedMean.txt'
trained_mean_sufix = '.trainedMean.txt'
dataForML_train_sufix = "_train"
dataForML_test_sufix = "_test"
trained_model_sufix = "_train.trainedModel.joblib"
tuned_model_sufix = "_train.tunedModel.joblib"
meta_prefix = "meta_set"

# Parameters for munging
genoTrain = "AMPPD"
genoTrains = ["AMPPD_1", "AMPPD_2", "AMPPD_3"]
path2Data = "/home/edusal/data/FINALES/"
path2Pheno = "/home/edusal/data/MAF0.05/"
pheno_sufix = "MyPhenotype.genoml.pheno"
n_features = 50
test_size = 0.3

# Parameters for train and tune
algorithms = ["RandomForestClassifier", "LogisticRegression", "SGDClassifier", "QuadraticDiscriminantAnalysis"]
algorithms_names = "RandomForestClassifier LogisticRegression SGDClassifier QuadraticDiscriminantAnalysis"
metric = "Balanced_Accuracy"
tune_iterations = 10

# Parameters for results
means = []
columns = ['Iteration', 'Algorithm', 'Trained Mean', 'Tuned Mean', 'Test score']
results_df = pd.DataFrame(columns=columns)
columns = ['Mean', 'Lower confidence-interval', 'Upper confidence-interval']
out_df = pd.DataFrame(columns=columns)
test_results = []
columns = ['Iteration', 'Expert', 'Trained Mean', 'Tuned Mean']
experts_res_df = pd.DataFrame(columns=columns)

#################################################
# Generate Munged data for each cohort (if it is not generated)
#################################################
# for genoTrain in genoTrains:
#     if not os.path.isfile(path2output + genoTrain + '.dataForML.h5'):
#         command = f'''
#         GenoMLMunging \
#         --prefix {path2output + genoTrain} \
#         --datatype d \
#         --geno {path2Data + genoTrain} \
#         --pheno {path2Pheno + genoTrain + '/' + pheno_sufix}
#         '''
#         # --featureSelection {n_features}

#         os.system(command)
#     else:
#         print("Munging already done. Munged file is: " + path2output + genoTrain + ".dataForML.h5\n")


#################################################
# Repeat experimentation process for each alg
#################################################

# Set the random's seed
random.seed(20)

seeds = []
for n in range(n_repeats):
    seeds.append(random.randint(1, 1000))

print(f"Used seeds are: {seeds}")

file = open(path2output + prefix + ".seeds.txt", "w+")
file.write(" ".join(str(seed) for seed in seeds))
file.close()

random.seed(20)

genoTrain_df = pd.read_hdf(path2output + genoTrain + ".dataForML.h5", key="dataForML")
print("#" * 30)
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

test_df = case_df.iloc[case_test_index, :].append(control_df.iloc[control_test_index, :])
test_df = test_df.sample(frac=1).reset_index(drop=True)

print("#" * 30)
print("Test data set:")
print(test_df)

genoTrain_df = genoTrain_df[~genoTrain_df.ID.isin(list(test_df.ID))]
genoTrain_df.index = range(len(genoTrain_df.index))

print("#" * 30)
print("Train data set:")
print(genoTrain_df)


# Divide dataset in 3 subsets
values = list(genoTrain_df.index)
data_len = len(genoTrain_df.index)
genoTrain_df_0 = random.sample(values, int(data_len / 3))
values = [i for i in values if i not in genoTrain_df_0]
genoTrain_df_1 = random.sample(values, int(data_len / 3))
values = [i for i in values if i not in genoTrain_df_1]
genoTrain_df_2 = values

df_0 = genoTrain_df.iloc[genoTrain_df_0, :]
df_1 = genoTrain_df.iloc[genoTrain_df_1, :]
df_2 = genoTrain_df.iloc[genoTrain_df_2, :]

df_0.to_hdf(path2output + genoTrains[0] + ".dataForML.h5", key="dataForML")
df_1.to_hdf(path2output + genoTrains[1] + ".dataForML.h5", key="dataForML")
df_2.to_hdf(path2output + genoTrains[2] + ".dataForML.h5", key="dataForML")

# Check if split has been succesfully done
if (len(set(genoTrain_df_0).intersection(set(genoTrain_df_1), set(genoTrain_df_2))) == 0):
    print("Subsets created")
else:
    print("Subsets have common indexes: ")
    print(set(genoTrain_df_0).intersection(set(genoTrain_df_1), set(genoTrain_df_2)))
    sys.exit()


for n in range(n_repeats):
    actual_prefix = prefix + '_' + str(n)
    actual_path = path2output + actual_prefix + '/'

    seed = random.randint(1, 1000)

    # Create directory
    try:
        os.mkdir(actual_path)
    except FileExistsError:
        continue
    # Move data
    for name in genoTrains:
        change_data_path(path2output, actual_path, name, name)

    # Divide train-test datasets
    for genoTrain in genoTrains:
        # Read and split data
        dataForML_file = actual_path + genoTrain + ".dataForML.h5"
        dataForML = pd.read_hdf(dataForML_file, key='dataForML')
        print(dataForML)
        dataForML_train, dataForML_test = model_selection.train_test_split(dataForML, test_size=test_size, random_state=seed)

        # Save all files
        dataForML_train.to_hdf(actual_path + genoTrain + dataForML_train_sufix + ".dataForML.h5", key="dataForML")
        dataForML_test.to_hdf(actual_path + genoTrain + dataForML_test_sufix + ".dataForML.h5", key="dataForML")

    # Get level one experts
    for genoTrain in genoTrains:
        # Train expert
        command = f'''
        GenoML discrete supervised train \
        --prefix {actual_path + genoTrain + dataForML_train_sufix} \
        --metric_max {metric} \
        --alg {algorithms_names} \
        --seed {seed}
        '''

        os.system(command)

        # Tune expert
        command = f'''
        GenoML discrete supervised tune \
        --prefix {actual_path + genoTrain + dataForML_train_sufix} \
        --metric_tune {metric} \
        --max_tune {tune_iterations} \
        --seed {seed}
        '''

        os.system(command)

    # Generate meta-set
    generate_metaset(actual_path)

    # Generate level 2 expert
    # Train model
    command = f'''
    GenoML discrete supervised train \
    --prefix {actual_path + meta_prefix} \
    --metric_max {metric} \
    --alg  {algorithms_names} \
    --seed {seed}
    '''

    os.system(command)

    # Tune model
    command = f'''
    GenoML discrete supervised tune \
    --prefix {actual_path + meta_prefix} \
    --metric_tune {metric} \
    --max_tune {tune_iterations} \
    --seed {seed}
    '''

    os.system(command)

    test_results.append(prediction_test(test_df))

    # Remove splits so we save memory
    for genoTrain in genoTrains:
        command = "rm " + actual_path + genoTrain + dataForML_train_sufix + ".dataForML.h5"
        os.system(command)
        command = "rm " + actual_path + genoTrain + dataForML_test_sufix + ".dataForML.h5"
        os.system(command)

    # Return data
    for genoTrain in genoTrains:
        change_data_path(actual_path, path2output, genoTrain, genoTrain)


#################################################
# Obtain results
#################################################


# Get means
for n in range(n_repeats):
    actual_prefix = prefix + '_' + str(n)
    actual_path = path2output + actual_prefix + '/'

    tuned_mean_file = actual_path + meta_prefix + tuned_mean_sufix
    trained_mean_file = actual_path + meta_prefix + trained_mean_sufix
    best_algo_file = actual_path + meta_prefix + ".best_algorithm.txt"

    # Load meta mean
    trained_mean = None
    tuned_mean = None
    best_algo = None

    if os.path.isfile(tuned_mean_file):
        file = open(tuned_mean_file, "r")
        tuned_mean = float(file.read()) * 100
        file.close()

    if os.path.isfile(trained_mean_file):
        file = open(trained_mean_file, "r")
        trained_mean = float(file.read())
        file.close()

    if os.path.isfile(best_algo_file):
        file = open(best_algo_file, "r")
        best_algo = file.read()
        file.close()
    else:
        print("File does not exist: " + best_algo_file)

    if tuned_mean is None or tuned_mean == 100.0:
        mean = trained_mean
    else:
        mean = tuned_mean

    # Add balanced accuracy to balanced accuracy list
    means.append(mean)

    # Save results in the df
    # entry = {'Iteration': n,
    #          'Algorithm': best_algo,
    #          'Trained Mean': trained_mean,
    #          'Tuned Mean': tuned_mean,
    #          'Test score': test_results[n]}
    # results_df = results_df.append(entry, ignore_index=True)

    # Obtain results of experts
    for genoTrain in genoTrains:
        entry = {}
        tuned_mean_file = actual_path + genoTrain + dataForML_train_sufix + tuned_mean_sufix
        trained_mean_file = actual_path + genoTrain + dataForML_train_sufix + trained_mean_sufix
        best_algo_file = actual_path + genoTrain + dataForML_train_sufix + ".best_algorithm.txt"

        trained_mean = None
        tuned_mean = None
        best_algo = None

        if os.path.isfile(tuned_mean_file):
            file = open(tuned_mean_file, "r")
            tuned_mean = float(file.read()) * 100
            file.close()

        if os.path.isfile(trained_mean_file):
            file = open(trained_mean_file, "r")
            trained_mean = float(file.read())
            file.close()

        if os.path.isfile(best_algo_file):
            file = open(best_algo_file, "r")
            best_algo = file.read()
            file.close()

        entry = {'Iteration': n,
                 'Expert': genoTrain,
                 'Best_algo': best_algo,
                 'Trained Mean': trained_mean,
                 'Tuned Mean': tuned_mean}
        experts_res_df = experts_res_df.append(entry, ignore_index=True)


print(means)

# Calculate mean and confidence interval
mean, conf_down, conf_up = mean_confidence_interval(means)

# Save results
out_file = meta_prefix + '.mean_confidence_interval.txt'
file = open(actual_path + out_file, "w+")
file.write(f'{mean} {conf_down} {conf_up}')
file.close()

entry = {'Mean': mean,
         'Lower confidence-interval': conf_down,
         'Upper confidence-interval': conf_up}
out_df = out_df.append(entry, ignore_index=True)


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

print(test_results)

out_df.to_csv(path2output + prefix + '.results.csv')
#results_df.to_csv(path2output + prefix + '.all_means.csv')
experts_res_df.to_csv(path2output + prefix + '.experts.csv')
