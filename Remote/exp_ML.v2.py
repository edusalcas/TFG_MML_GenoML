#################################################
# IMPORTS
#################################################
import pandas as pd
import random
import os
import joblib

from sklearn import metrics
import genoml_utils as gml
import ml_utils as mlu

#################################################
# Functions
#################################################


def prediction_test(test_df):
    Y_test = test_df['PHENO']
    test_df = test_df.drop(columns=['PHENO', 'ID'])

    model = None
    if os.path.isfile(actual_path_rep + train_prefix + ".tunedModel.joblib"):
        model = joblib.load(actual_path_rep + train_prefix + ".tunedModel.joblib")
    else:
        model = joblib.load(actual_path_rep + train_prefix + ".trainedModel.joblib")

    test_predictions = model.predict(test_df)
    balacc = metrics.balanced_accuracy_score(Y_test, test_predictions)
    return balacc * 100

#################################################
# Parameters
#################################################


# Prefixes
prefix = "exp"
train_prefix = prefix + "_train"
test_prefix = prefix + "_test"
meta_prefix = prefix + "_meta"

# Sufixes
suf_dataForML = ".dataForML.h5"

# Global parameters
n_repeats = 5
path2output = "/home/edusal/pythonNotebooks/Experiments/Experimentacion/out_ml_v2/"
hdf_key = 'dataForML'

# Sufixes
features = "_approx_feature_importance.txt"
dataForML = ".dataForML.h5"
variants_and_alleles = ".variants_and_alleles.tab"
dataForML_train = "_train.dataForML.h5"
dataForML_test = "_test.dataForML.h5"
tuned_mean_sufix = '.tunedMean.txt'
trained_mean_sufix = '.trainedMean.txt'

# Parameters for munging
genoTrain = "AMPPD"  # Cohorte a estudiar
path2geno = "/home/edusal/genoml-python_v1.5/examples/discrete/training"  # all data in the same folder (geno and covs)
path2pheno = "/home/edusal/genoml-python_v1.5/examples/discrete/training_pheno.csv"
features = None
test_size = 0.3
path2cli = "/home/edusal/data/dataForML/clinicodemographic_march10th2020.csv"
columns_from_cli = []  # AGE FEMALE EDUCATION FAMILY_HISTORY UPSIT


# Parameters for train and tune
algorithms = ["RandomForestClassifier", "LogisticRegression", "SGDClassifier", "QuadraticDiscriminantAnalysis"]
metric = "Balanced_Accuracy"
tune_iterations = 10

# Parameters for results
means = []
columns = ['Iteration', 'Algorithm', 'Trained Mean', 'Tuned Mean', 'Test score']
results_df = pd.DataFrame(columns=columns)
columns = ['Algorithm', 'Mean', 'Lower confidence-interval', 'Upper confidence-interval']
out_df = pd.DataFrame(columns=columns)
test_results = {}

#################################################
# Generate Munged data (if it is not generated) #
#################################################

if not os.path.isfile(path2output + prefix + '.dataForML.h5'):
    gml.munge(prefix=path2output + prefix,
              datatype="d",
              geno=path2geno,
              pheno=path2pheno,
              featureSelection=features)
else:
    print("Munging already done. Munged file is: " + path2output + prefix + ".dataForML.h5\n")

# Read data
geno_df = pd.read_hdf(path2output + prefix + suf_dataForML, key=hdf_key)

# Add clinical data info to data
cli_df = pd.read_csv(path2cli)
geno_df = mlu.copy_columns(dst=geno_df, src=cli_df, colums_names=columns_from_cli, on='ID')

random.seed(20)

seed = random.randint(1, 1000)


#################################################
# Repeat experimentation process for each alg
#################################################

# Set the random's seed
random.seed(20)
seed = random.randint(1, 1000)

genoTrain_df, test_df = mlu.split_train_test(geno_df, test_percent=0.2, balanced=True, seed=seed)
# Save train set
genoTrain_df.to_hdf(path2output + train_prefix + ".dataForML.h5", key="dataForML")

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
    actual_path = path2output + alg + '/'

    # If directory is already created, skip algorithm
    try:
        os.mkdir(actual_path)
    except FileExistsError:
        continue

    mlu.move_data(path2output + prefix, actual_path + prefix)

    test_results[alg] = []

    # For each alg, repeat esperiment n_repeats times
    for n in range(n_repeats):

        # Obtain new seed
        seed = random.randint(1, 1000)
        print("Iteration: " + str(n) + " for algorithm " + alg)

        actual_path_rep = actual_path + prefix + "_" + str(n) + '/'

        # Create directory
        try:
            os.mkdir(actual_path_rep)
        except FileExistsError:
            pass

        mlu.move_data(actual_path + prefix, actual_path_rep + prefix)

        # Train model
        gml.train(prefix=actual_path_rep + train_prefix, metric_max=metric, algs=alg, seed=seed)
        # Tune model
        gml.tune(prefix=actual_path_rep + train_prefix, metric_tune=metric, max_tune=tune_iterations, seed=seed)

        # Predict
        balacc = prediction_test(test_df)
        test_results[alg].append(balacc)

        # Return data to directory and change name
        mlu.move_data(actual_path_rep + prefix, actual_path + prefix)

    # Return data to directory and change name
    mlu.move_data(actual_path + prefix, path2output + prefix)

#################################################
# Obtain results
#################################################

for alg in algorithms:

    means = []

    actual_path = path2output + alg + '/'

    # Get means
    for n in range(n_repeats):

        actual_path_rep = actual_path + prefix + "_" + str(n) + '/'

        tuned_mean_file = actual_path_rep + train_prefix + tuned_mean_sufix
        trained_mean_file = actual_path_rep + train_prefix + trained_mean_sufix

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

        if tuned_mean is None or tuned_mean < trained_mean:
            mean = trained_mean
        else:
            mean = tuned_mean

        # Add balanced accuracy to balanced accuracy list
        means.append(mean)

        # Save results in the df
        entry = {}
        entry = {'Iteration': n_repeats,
                 'Algorithm': alg,
                 'Trained Mean': trained_mean,
                 'Tuned Mean': tuned_mean,
                 'Test score': test_results[alg][n]}
        results_df.append(entry, ignore_index=True)

    # Calculate mean and confidence interval
    mean, conf_down, conf_up = mlu.mean_confidence_interval(means)

    # Save results
    out_file = prefix + '.mean_confidence_interval.txt'
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
    print(f"Results for {alg} in training phase are:\n\t- Mean: {mean:.4f}\n\t- Conficence interval: {conf_down:.4f} - {conf_up:.4f}")
    print('#' * 60)

    mean, conf_down, conf_up = mlu.mean_confidence_interval(test_results[alg])

    # Show results
    print()
    print('#' * 60)
    print(f"Results for {alg} in testing phase are:\n\t- Mean: {mean:.4f}\n\t- Conficence interval: {conf_down:.4f} - {conf_up:.4f}")
    print('#' * 60)

# Save results
results_df.to_csv(path2output + prefix + '_all_means.csv')
out_df.to_csv(path2output + prefix + '_result.csv')
print(test_results)
