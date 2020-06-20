import numpy as np
import pandas as pd
import scipy as sp
import genoml_utils as gml
import scipy.stats
import random
import os
import joblib

# Sufixes
suf_tuned_model = ".tunedModel.joblib"
suf_trained_model = ".trainedModel.joblib"
suf_dataForML = ".dataForML.h5"
features = "_approx_feature_importance.txt"
dataForML = ".dataForML.h5"
variants_and_alleles = ".variants_and_alleles.tab"
dataForML_train = "_train.dataForML.h5"
dataForML_test = "_test.dataForML.h5"


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1 + confidence) / 2., n - 1)

    return m, m - h, m + h


def welchs_test(list_1, list_2):
    return scipy.stats.ttest_ind(list_1, list_2, equal_var=False)


def split_train_test(df, test_percent=0.2, balanced=True, seed=2020):
    train_df, test_df = None, None
    random.seed(seed)
    print("####################################")
    print("Full data set:")
    print(df)

    # If we want the test partition balanced
    if balanced:
        num_test = int(len(df.index) * test_percent)

        out_val = list(set(df.PHENO))

        case_df = df[df.PHENO == out_val[0]]
        control_df = df[df.PHENO == out_val[1]]

        values = list(case_df.index)
        case_test_index = random.sample(values, num_test // 2)
        values = list(control_df.index)
        control_test_index = random.sample(values, num_test // 2)

        test_df = case_df.loc[case_test_index, :].append(control_df.loc[control_test_index, :])
        train_df = df.copy()
        train_df = train_df.loc[[item for item in train_df.index if item not in set(test_df.index)], :]

        test_df.index = range(len(test_df.index))
        train_df.index = range(len(train_df.index))

        print("####################################")
        print("Test data set:")
        print(test_df)

        print("####################################")
        print("Train data set:")
        print(train_df)

    else:
        # TODO create not balanced split
        pass

    return train_df, test_df


def move_data(in_path, out_path):
    print(f"Moving files from {in_path} to {out_path}")

    # Move data files to directory
    # features
    source = in_path + features
    destination = out_path + features
    command = f"mv {source} {destination} 2>/dev/null"
    os.system(command)

    # dataForML
    source = in_path + dataForML
    destination = out_path + dataForML
    command = f"mv {source} {destination} 2>/dev/null"
    os.system(command)

    # variants_and_alleles
    source = in_path + variants_and_alleles
    destination = out_path + variants_and_alleles
    command = f"mv {source} {destination} 2>/dev/null"
    os.system(command)

    # dataForML_train
    source = in_path + dataForML_train
    destination = out_path + dataForML_train
    command = f"mv {source} {destination} 2>/dev/null"
    os.system(command)

    # dataForML_test
    source = in_path + dataForML_test
    destination = out_path + dataForML_test
    command = f"mv {source} {destination} 2>/dev/null"
    os.system(command)


def fix_columns(df, columns):
    aux = df.copy()

    diff1 = set(df.columns) - set(columns)
    if len(diff1) > 0:
        aux.drop(diff1, 1, inplace=True)

    diff2 = set(columns) - set(df.columns)
    if len(diff2) > 0:
        new_cols = pd.DataFrame(0, index=np.arange(len(aux)), columns=diff2)

        aux.index = range(len(aux.index))
        aux = pd.concat([aux, new_cols], axis=1)

    return aux


def common_columns(dataFrames, columns=None):
    if columns is None:
        # Initialize commonColumns
        commonColumns = dataFrames[next(iter(dataFrames))].columns

        # For each df, intersect columns
        for key in dataFrames.keys():
            commonColumns = np.intersect1d(commonColumns, dataFrames[key].columns)

        # Return commonColumns
        return list(commonColumns)
    else:
        return dataFrames[np.intersect1d(columns, dataFrames.columns)]


def bind_rows(dataFrames, columns=None):
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


def copy_columns(dst, src, colums_names, on):
    if colums_names == []:
        return dst
    return dst.join(src.set_index(on)[colums_names], on=on)
