#################################################
# IMPORTS
#################################################
import pandas as pd
import random
import os

from sklearn import model_selection
from sklearn import metrics
from mml_nsme import MML_nSmE
import genoml_utils as gml
import ml_utils as mlu

#################################################
# Functions
#################################################


def prediction_test(test_df):
    Y_test = test_df['PHENO']

    test_predictions = manager_mml.predict(test_df, actual_path, [genoTrain + train_sufix for genoTrain in genoTrains], meta_prefix)

    balacc = metrics.balanced_accuracy_score(Y_test, test_predictions)

    return balacc


def divide_data(genoTrain_df):
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

    df_0.to_hdf(path2output + genoTrains[0] + suf_dataForML, key=hdf_key)
    df_1.to_hdf(path2output + genoTrains[1] + suf_dataForML, key=hdf_key)
    df_2.to_hdf(path2output + genoTrains[2] + suf_dataForML, key=hdf_key)

#################################################
# Parameters
#################################################


# Prefixes
prefix = "exp_FEM"
train_prefix = prefix + "_train"
test_prefix = prefix + "_test"
meta_prefix = prefix + "_meta"

# Sufixes
suf_dataForML = ".dataForML.h5"
train_sufix = "_train"
test_sufix = "_test"

# Global parameters
n_repeats = 5
path2output = "/home/edusal/pythonNotebooks/Experiments/Experimentacion/out_nsme_v2/"
hdf_key = 'dataForML'

# Parameters for munging
genoTrain = "AMPPD"
genoTrains = ["exp_FEM_1", "exp_FEM_2", "exp_FEM_3"]
path2geno = "/home/edusal/genoml-python_v1.5/examples/discrete/training"  # all data in the same folder (geno and covs)
path2pheno = "/home/edusal/genoml-python_v1.5/examples/discrete/training_pheno.csv"
features = None
test_size = 0.3
path2cli = "/home/edusal/data/dataForML/clinicodemographic_march10th2020.csv"
columns_from_cli = ['FEMALE']  # AGE FEMALE EDUCATION FAMILY_HISTORY UPSIT

# Parameters for train and tune
algorithms = ["RandomForestClassifier", "LogisticRegression", "SGDClassifier", "QuadraticDiscriminantAnalysis"]
algorithms_names = "RandomForestClassifier LogisticRegression SGDClassifier QuadraticDiscriminantAnalysis"
metric = "Balanced_Accuracy"
tune_iterations = 10

# Parameters for result
test_results = []


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

# Train-Test division
genoTrain_df, test_df = mlu.split_train_test(geno_df, test_percent=0.2, balanced=True, seed=seed)
divide_data(genoTrain_df)


random.seed(20)
#################################################
# Repeat meta learning 'n_repeats' times
#################################################

for n in range(n_repeats):
    print("Iteration: " + str(n))
    output_prefix = prefix + "_" + str(n)
    actual_path = path2output + output_prefix + "/"

    # Gen new seed
    seed = random.randint(1, 1000)

    # Create output directory
    try:
        os.mkdir(path2output + output_prefix)
    except FileExistsError:
        continue

    manager_mml = MML_nSmE(actual_path + prefix + "_")

    # Move data
    for expert in genoTrains:
        mlu.move_data(path2output + expert, actual_path + expert)

    # Divide train-test datasets
    for genoTrain in genoTrains:
        # Read and split data
        dataForML_file = actual_path + genoTrain + suf_dataForML
        dataForML = pd.read_hdf(dataForML_file, key=hdf_key)
        dataForML_train, dataForML_test = model_selection.train_test_split(dataForML, test_size=test_size, random_state=seed)

        # Save all files
        dataForML_train.to_hdf(actual_path + genoTrain + train_sufix + suf_dataForML, key=hdf_key)
        dataForML_test.to_hdf(actual_path + genoTrain + test_sufix + suf_dataForML, key=hdf_key)

    # Generate level 1 experts
    manager_mml.gen_experts(path=actual_path, expert_names=[geno + train_sufix for geno in genoTrains], metric_max=metric, seed=seed, metric_tune=metric, max_tune=tune_iterations, algs=algorithms_names)

    # Generate level 2 expert
    df_0 = pd.read_hdf(actual_path + genoTrains[0] + test_sufix + suf_dataForML, key=hdf_key)
    df_1 = pd.read_hdf(actual_path + genoTrains[1] + test_sufix + suf_dataForML, key=hdf_key)
    df_2 = pd.read_hdf(actual_path + genoTrains[2] + test_sufix + suf_dataForML, key=hdf_key)

    manager_mml.gen_meta_expert([df_0, df_1, df_2], actual_path, [genoTrain + train_sufix for genoTrain in genoTrains], meta_prefix, metric, algorithms_names, seed, metric, tune_iterations)

    # Predict test set
    balacc = manager_mml.predict(test_df, actual_path, [genoTrain + train_sufix for genoTrain in genoTrains], meta_prefix, n)
    test_results.append(balacc)

    # Change output names and directory
    for expert in genoTrains:
        mlu.move_data(actual_path + expert, path2output + expert)

for n in range(n_repeats):
    output_prefix = prefix + "_" + str(n)
    actual_path = path2output + output_prefix + "/"

    manager_mml.get_results(experts_names=[genoTrain + train_sufix for genoTrain in genoTrains], path=actual_path, prefix=meta_prefix, test_result=test_results[n], iteration=n)

manager_mml.save_results(path=path2output, meta_prefix=meta_prefix, test_results=test_results)

print(test_results)
