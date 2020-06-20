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
    source = input_path + input_prefix + dataForML
    destination = output_path + output_prefix + dataForML
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


def prediction_test(test_df):
    Y_test = test_df['PHENO']
    test_df = test_df.drop(columns=['PHENO', 'ID'])

    model = None
    if os.path.isfile(actual_path_rep + rep_prefix + ".tunedModel.joblib"):
        model = joblib.load(actual_path_rep + rep_prefix + ".tunedModel.joblib")
    else:
        model = joblib.load(actual_path_rep + rep_prefix + ".trainedModel.joblib")

    test_predictions = model.predict(test_df)
    balacc = metrics.balanced_accuracy_score(Y_test, test_predictions)
    return balacc

#################################################
# Parameters
#################################################


# Global parameters
n_repeats = 5
path2output = "/home/edusal/pythonNotebooks/Experiments/Experimentacion/exp_output/"
prefix = "exp"

# Sufixes
features = "_approx_feature_importance.txt"
dataForML = ".dataForML.h5"
variants_and_alleles = ".variants_and_alleles.tab"
dataForML_train = "_train.dataForML.h5"
dataForML_test = "_test.dataForML.h5"
tuned_mean_sufix = '.tunedMean.txt'
trained_mean_sufix = '.trainedMean.txt'

# Parameters for munging
genoTrain = "AMPPD"
path2Data = "/home/edusal/data/FINALES/"
path2Pheno = "/home/edusal/data/MAF0.05/SPAIN/MyPhenotype.genoml.pheno"
n_features = 50
test_size = 0.3

# Parameters for train and tune
algorithms = ["RandomForestClassifier"]
metric = "Balanced_Accuracy"
tune_iterations = 10

# Parameters for results
means = []
columns = ['Algorithm', 'Iteration', 'Trained Mean', 'Tuned Mean']
results_df = pd.DataFrame(columns=columns)
columns = ['Algorithm', 'Mean', 'Lower confidence-interval', 'Upper confidence-interval']
out_df = pd.DataFrame(columns=columns)
test_results = []

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
    --featureSelection {n_features}
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
case_test_index = random.sample(values, 50)
values = list(control_df.index)
control_test_index = random.sample(values, 50)

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

for alg in algorithms:

    random.seed(20)

    alg_prefix = prefix + '_' + alg
    actual_path = path2output + alg_prefix + '/'

    # If directory is already created, skip algorithm
    try:
        os.mkdir(actual_path)
    except FileExistsError:
        continue

    # Move data to directory and change name
    change_data_path(path2output, actual_path, prefix, alg_prefix)

    # For each alg, repeat esperiment n_repeats times
    for n in range(n_repeats):

        # Obtain new seed
        seed = random.randint(1, 1000)

        rep_prefix = alg_prefix + "_" + str(n)
        actual_path_rep = actual_path + rep_prefix + '/'

        # Create directory
        try:
            os.mkdir(actual_path_rep)
        except FileExistsError:
            pass

        # Move data to directory and change name
        change_data_path(actual_path, actual_path_rep, alg_prefix, rep_prefix)

        # Train model
        command = f'''
        GenoML discrete supervised train \
        --prefix {actual_path_rep + rep_prefix} \
        --metric_max {metric} \
        --alg  {alg} \
        --seed {seed}
        '''

        os.system(command)

        # Tune model
        command = f'''
        GenoML discrete supervised tune \
        --prefix {actual_path_rep + rep_prefix} \
        --metric_tune {metric} \
        --max_tune {tune_iterations} \
        --seed {seed}
        '''

        os.system(command)

        test_results.append(prediction_test(test_df))

        # Return data to directory and change name
        change_data_path(actual_path_rep, actual_path, rep_prefix, alg_prefix)

    # Return data to directory and change name
    change_data_path(actual_path, path2output, alg_prefix, prefix)

#################################################
# Obtain results
#################################################

for alg in algorithms:

    means = []

    alg_prefix = prefix + '_' + alg
    actual_path = path2output + alg_prefix + '/'

    # Get means
    for n in range(n_repeats):

        rep_prefix = alg_prefix + "_" + str(n)
        actual_path_rep = actual_path + rep_prefix + '/'

        tuned_mean_file = actual_path_rep + rep_prefix + tuned_mean_sufix
        trained_mean_file = actual_path_rep + rep_prefix + trained_mean_sufix

        # Load meta mean
        trained_mean = None
        tuned_mean = None

        if os.path.isfile(tuned_mean_file):
            file = open(tuned_mean_file, "r")
            tuned_mean = float(file.read()) * 100
            file.close()

        if os.path.isfile(trained_mean_file):
            file = open(trained_mean_file, "r")
            trained_mean = float(file.read())
            file.close()

        if tuned_mean is None or tuned_mean == 100.0:
            mean = trained_mean
        else:
            mean = tuned_mean

        # Add balanced accuracy to balanced accuracy list
        means.append(mean)

        # Save results in the df
        entry = {'Algorithm': alg,
                 'Iteration': n,
                 'Trained Mean': trained_mean,
                 'Tuned Mean': tuned_mean}
        results_df = results_df.append(entry, ignore_index=True)

    print(means)

    # Calculate mean and confidence interval
    mean, conf_down, conf_up = mean_confidence_interval(means)

    # Save results
    out_file = alg_prefix + '.mean_confidence_interval.txt'
    file = open(actual_path + out_file, "w+")
    file.write(f'{mean} {conf_down} {conf_up}')
    file.close()

    entry = {'Algorithm': alg,
             'Mean': mean,
             'Lower confidence-interval': conf_down,
             'Upper confidence-interval': conf_up}
    out_df = out_df.append(entry, ignore_index=True)

    # Show results
    print()
    print('#' * 60)
    print(f"Results for {alg} are:\n\t- Mean: {mean:.4f}\n\t- Conficence interval: {conf_down:.4f} - {conf_up:.4f}")
    print('#' * 60)

# Save results
results_df.to_csv(path2output + prefix + 'all_means.csv')
out_df.to_csv(path2output + prefix + '.results.csv')

print(test_results)
