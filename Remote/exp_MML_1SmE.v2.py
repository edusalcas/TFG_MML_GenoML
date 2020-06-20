#################################################
# IMPORTS
#################################################
import pandas as pd
import random
import os
import sys

from sklearn import model_selection
from mml_1sme import MML_1SmE
import genoml_utils as gml
import ml_utils as mlu

#################################################
# Functions
#################################################


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
path2output = "/home/edusal/pythonNotebooks/Experiments/Experimentacion/out_1sme_v2/"
hdf_key = 'dataForML'

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
algorithms_names = "RandomForestClassifier LogisticRegression SGDClassifier QuadraticDiscriminantAnalysis"
metric = "Balanced_Accuracy"
tune_iterations = 10

# Parameters for result
test_results = []

# Set the random's seed
random.seed(20)
seed = random.randint(1, 1000)

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

# Train-Test division
genoTrain_df, test_df = mlu.split_train_test(geno_df, test_percent=0.2, balanced=True, seed=seed)

# sys.exit()
#################################################
# Repeat meta learning 'n_repeats' times
#################################################
random.seed(20)
for n in range(n_repeats):
    output_prefix = prefix + "_" + str(n)
    actual_path = path2output + output_prefix + "/"
    # Set seed
    seed = random.randint(1, 1000)

    # Create output directory
    try:
        os.mkdir(path2output + output_prefix)
    except FileExistsError:
        pass

    manager_mml = MML_1SmE(actual_path + prefix)

    # Divide train-test datasets
    dataForML_train, dataForML_test = model_selection.train_test_split(genoTrain_df, test_size=test_size, random_state=seed)

    dataForML_train.to_hdf(path2output + train_prefix + suf_dataForML, key=hdf_key)

    # Move data
    mlu.move_data(path2output + prefix, actual_path + prefix)

    # Generate level 1 experts
    manager_mml.gen_experts(path=actual_path, expert_names=algorithms, prefix=train_prefix, metric_max=metric, seed=seed, metric_tune=metric, max_tune=tune_iterations)

    # Generate level 2 expert
    manager_mml.gen_meta_expert(data=dataForML_test, path=actual_path, experts_names=algorithms, prefix=meta_prefix, metric_max=metric, algs=algorithms_names, seed=seed, metric_tune=metric, max_tune=tune_iterations)

    # Predict test set
    balacc = manager_mml.predict(test_df, actual_path, algorithms, meta_prefix, n)
    test_results.append(balacc)

    # Change output names and directory
    mlu.move_data(actual_path + prefix, path2output + prefix)


for n in range(n_repeats):
    output_prefix = prefix + "_" + str(n)
    actual_path = path2output + output_prefix + "/"

    manager_mml.get_results(experts_names=algorithms, path=actual_path, prefix=meta_prefix, test_result=test_results[n], iteration=n)

manager_mml.save_results(path=path2output, meta_prefix=meta_prefix, test_results=test_results)

print(test_results)
