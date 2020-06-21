import os
import time
import pickle
import pandas as pd
import numpy as np
import fromGenoToMLData as fm

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score, confusion_matrix, cohen_kappa_score
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from glmnet import LogitNet
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier


#----------------------------------------#
#--------------genModels_1SmE------------#
#----------------------------------------#


def genModels_1SmE(workPath,
                   handlerMLdata,
                   k):

    X = pd.read_csv(handlerMLdata["train1mldata"], header=0, delim_whitespace=True)
    X = X.loc[X["AGE"] != -9]

    Y = X["PHENO"]
    Y.columns = ["PHENO"]
    Y = Y.to_numpy()
    le = LabelEncoder()
    le.fit(Y)
    Y_le = le.transform(Y)
    print(le.classes_)

    X.drop("PHENO", 1, inplace=True)
    X.dropna(inplace=True)
    print(Y[Y == 0])

    # ID_column = X["ID"]
    X.drop("ID", 1, inplace=True)
    X = X.dropna()
    X_np = X.to_numpy()

    np.random.seed(1234)
    myfolds = KFold(n_splits=k)

    # algsML = c("glmnet", "C5.0Tree", "earth", "svmRadial", "rf", "xgbTree", "xgbLinear", "xgbDART", "gbm")
    algsML = [RandomForestClassifier(), LogisticRegression(solver='lbfgs'), DecisionTreeClassifier()]
    algs_cols = ["Fold", "Algorithm", "AUC_Percent", "Accuracy_Percent", "Balanced_Accuracy_Percent", "Log_Loss", "Sensitivity", "Specificity", "PPV", "NPV", "Runtime_Seconds"]

    # for each fold, do the training with all the algorithms included in the variable algs
    algs_table = pd.DataFrame(columns=algs_cols)

    # to begin tune first begin parallel parameters
    # cluster < - makeCluster(ncores)  # convention to leave 1 core for OS
    # registerDoParallel(cluster)

    folds_cols = ["Fold", "Algorithm", "AUC_Percent", "Accuracy_Percent", "Balanced_Accuracy_Percent", "Log_Loss", "Sensitivity", "Specificity", "PPV", "NPV", "Runtime_Seconds"]
    folds_table = pd.DataFrame(columns=folds_cols)

    models = {}
    finalModels = {}

    for i in range(1, k + 1):
        models[i] = {}

    for alg in algsML:

        name = alg.__class__.__name__

        print()
        print("#" * 30)
        print(name + "\n")

        fold = 1

        for train_index, test_index in myfolds.split(X_np):

            print()
            print("\t" + "#" * 26)
            print("\t" + "Fold " + str(fold) + "\n")
            start_time = time.time()
            np.random.seed(1234)

            X_train, X_test = X_np[train_index], X_np[test_index]
            Y_train, Y_test = Y_le[train_index], Y_le[test_index]

            alg.fit(X_train, Y_train)

            models[fold][name] = alg

            test_predictions = alg.predict_proba(X_test)
            rocauc = roc_auc_score(Y_test, test_predictions[:, 1])
            print("\t\tAUC: {:.4%}".format(rocauc))

            test_predictions = alg.predict(X_test)
            acc = accuracy_score(Y_test, test_predictions)
            print("\t\tAccuracy: {:.4%}".format(acc))

            test_predictions = alg.predict(X_test)
            balacc = balanced_accuracy_score(Y_test, test_predictions)
            print("\t\tBalanced Accuracy: {:.4%}".format(balacc))

            test_predictions = alg.predict(X_test)
            kappa = cohen_kappa_score(Y_test, test_predictions)
            print("\t\tKappa: {:.4%}".format(kappa))

            CM = confusion_matrix(Y_test, test_predictions)
            TN = CM[0][0]
            FN = CM[1][0]
            TP = CM[1][1]
            FP = CM[0][1]
            sensitivity = TP / (TP + FN)
            specificity = TN / (TN + FP)
            PPV = TP / (TP + FP)
            NPV = TN / (TN + FN)

            test_predictions = alg.predict_proba(X_test)
            ll = log_loss(Y_test, test_predictions)
            print("\t\tLog Loss: {:.4}".format(ll))

            end_time = time.time()
            elapsed_time = (end_time - start_time)
            print("\t\tRuntime in seconds: {:.4}".format(elapsed_time))

            folds_entry = pd.DataFrame([[fold, name, rocauc * 100, acc * 100, balacc * 100, ll, sensitivity, specificity, PPV, NPV, elapsed_time]], columns=folds_cols)
            folds_table = folds_table.append(folds_entry)

            fold = fold + 1

    # # shut down multicore
    # stopCluster(cluster)
    # registerDoSEQ()

    folds_table.index = range(len(folds_table.index))

    for alg in algsML:
        name = alg.__class__.__name__
        alg_best_entry = folds_table.iloc[folds_table[folds_table["Algorithm"] == name]['Balanced_Accuracy_Percent'].idxmax()]
        finalModels[name] = models[alg_best_entry["Fold"]][name]
        algs_table = algs_table.append(alg_best_entry)

    algs_table.index = range(len(algs_table.index))

    print("\n\n")
    print(algs_table)
    print("\n")

    finalModels["columns"] = X.columns
    # Save models
    with open(workPath + "/models.pydat", 'wb') as finalModels_file:
        pickle.dump(finalModels, finalModels_file)

    return finalModels


#----------------------------------------#
#--------------genModels_nSmE------------#
#----------------------------------------#


def genModels_nSmE(workPath,
                   handlerMLdata,
                   k):

    X = pd.read_csv(handlerMLdata["train1mldata"], header=0, delim_whitespace=True)
    X = X.loc[X["AGE"] != -9]

    Y = X["PHENO"]
    Y.columns = ["PHENO"]
    Y = Y.to_numpy()
    le = LabelEncoder()
    le.fit(Y)
    Y_le = le.transform(Y)
    print(le.classes_)

    X.drop("PHENO", 1, inplace=True)
    X.dropna(inplace=True)
    print(Y[Y == 0])

    # ID_column = X["ID"]
    X.drop("ID", 1, inplace=True)
    X = X.dropna()
    X_np = X.to_numpy()

    np.random.seed(1234)
    myfolds = KFold(n_splits=k)

    # algsML = c("glmnet", "C5.0Tree", "earth", "svmRadial", "rf", "xgbTree", "xgbLinear", "xgbDART", "gbm")
    algsML = [LogitNet(), RandomForestClassifier(), LogisticRegression(solver='lbfgs'), DecisionTreeClassifier(), GradientBoostingClassifier()]
    algs_cols = ["Fold", "Algorithm", "AUC_Percent", "Accuracy_Percent", "Balanced_Accuracy_Percent", "Log_Loss", "Sensitivity", "Specificity", "PPV", "NPV", "Runtime_Seconds"]

    # for each fold, do the training with all the algorithms included in the variable algs
    algs_table = pd.DataFrame(columns=algs_cols)

    # to begin tune first begin parallel parameters
    # cluster < - makeCluster(ncores)  # convention to leave 1 core for OS
    # registerDoParallel(cluster)

    folds_cols = ["Fold", "Algorithm", "AUC_Percent", "Accuracy_Percent", "Balanced_Accuracy_Percent", "Log_Loss", "Sensitivity", "Specificity", "PPV", "NPV", "Runtime_Seconds"]
    folds_table = pd.DataFrame(columns=folds_cols)

    models = {}
    finalModel = {}

    for i in range(1, k + 1):
        models[i] = {}

    for alg in algsML:

        name = alg.__class__.__name__

        print()
        print("#" * 30)
        print(name + "\n")

        fold = 1

        for train_index, test_index in myfolds.split(X_np):

            print()
            print("\t" + "#" * 26)
            print("\t" + "Fold " + str(fold) + "\n")
            start_time = time.time()
            np.random.seed(1234)

            X_train, X_test = X_np[train_index], X_np[test_index]
            Y_train, Y_test = Y_le[train_index], Y_le[test_index]

            alg.fit(X_train, Y_train)

            models[fold][name] = alg

            test_predictions = alg.predict_proba(X_test)
            rocauc = roc_auc_score(Y_test, test_predictions[:, 1])
            print("\t\tAUC: {:.4%}".format(rocauc))

            test_predictions = alg.predict(X_test)
            acc = accuracy_score(Y_test, test_predictions)
            print("\t\tAccuracy: {:.4%}".format(acc))

            test_predictions = alg.predict(X_test)
            balacc = balanced_accuracy_score(Y_test, test_predictions)
            print("\t\tBalanced Accuracy: {:.4%}".format(balacc))

            test_predictions = alg.predict(X_test)
            kappa = cohen_kappa_score(Y_test, test_predictions)
            print("\t\tKappa: {:.4%}".format(kappa))

            CM = confusion_matrix(Y_test, test_predictions)
            TN = CM[0][0]
            FN = CM[1][0]
            TP = CM[1][1]
            FP = CM[0][1]
            sensitivity = TP / (TP + FN)
            specificity = TN / (TN + FP)
            PPV = TP / (TP + FP)
            NPV = TN / (TN + FN)

            test_predictions = alg.predict_proba(X_test)
            ll = log_loss(Y_test, test_predictions)
            print("\t\tLog Loss: {:.4}".format(ll))

            end_time = time.time()
            elapsed_time = (end_time - start_time)
            print("\t\tRuntime in seconds: {:.4}".format(elapsed_time))

            folds_entry = pd.DataFrame([[fold, name, rocauc * 100, acc * 100, balacc * 100, ll, sensitivity, specificity, PPV, NPV, elapsed_time]], columns=folds_cols)
            folds_table = folds_table.append(folds_entry)

            fold = fold + 1

    folds_table.index = range(len(folds_table.index))

    alg_best_entry = folds_table.iloc[folds_table['Balanced_Accuracy_Percent'].idxmax()]
    finalModel["model"] = models[alg_best_entry["Fold"]][alg_best_entry["Algorithm"]]

    finalModel["columns"] = X.columns
    # Save models
    with open(workPath + "/models.pydat", 'wb') as finalModel_file:
        pickle.dump(finalModel, finalModel_file)

    return finalModel

#----------------------------------------#
#-----------trainAndTestMML_1SmE---------#
#----------------------------------------#


def trainAndTestMML_1SmE(experts_L1,
                         handlerMLdata,
                         workPath="/home/rafa/MetaLearning/metaML/Modelos/"):

    # dataTrain = pd.read_csv(handlerMLdata["train1mldata"], header=0, delim_whitespace=True)
    dataTest = pd.read_csv(handlerMLdata["test1mldata"], header=0, delim_whitespace=True)

    totalData = dataTest

    totalData = totalData.loc[totalData["AGE"] != -9]
    ID = totalData["ID"]
    totalData.drop("ID", 1, inplace=True)

    Y = totalData["PHENO"]
    totalData.drop("PHENO", 1, inplace=True)

    Y.index = range(len(Y.index))
    ID.index = range(len(ID.index))

    metaSet = pd.concat([ID, Y], axis=1)

    # As argument in future version
    algsML = [RandomForestClassifier(), LogisticRegression(solver='lbfgs'), DecisionTreeClassifier()]

    for alg in algsML:
        name = alg.__class__.__name__
        preds = experts_L1[name].predict(totalData)
        metaSet["Pred-" + name] = preds

    # We assume MGSA type (we add age, sex and geno)

    cols = totalData.columns
    cols = [col for col in cols if not col.startswith("PC")]
    all_cols = totalData[cols]

    metaSet.index = range(len(metaSet.index))
    all_cols.index = range(len(all_cols.index))

    metaSet = pd.concat([metaSet, all_cols], axis=1)

    ID = metaSet["ID"]
    metaSet.drop("ID", 1, inplace=True)
    metaSet.drop("PHENO", 1, inplace=True)

    # addition of PCA to our metaSet
    # subset = [pca for pca in pcaSET if pca["ID"] in ID]
    # subset = pd.merge(pcaSET, ID, on="ID")

    # subset.index = range(len(subset.index))
    # metaSet.index = range(len(metaSet.index))

    # We dont use PCA for MML model
    # metaSet = pd.concat([metaSet, subset.iloc[:, 3:12]], axis=1)
    # metaSet.columns.values[(len(metaSet.columns) - 10):len(metaSet.columns)] = ["PCA1", "PCA2", "PCA3", "PCA4", "PCA5", "PCA6", "PCA7", "PCA8", "PCA9", "PCA10"]

    # separate the new dataframe into train and test
    X_trainMML, X_testMML, Y_trainMML, Y_testMML = train_test_split(metaSet, Y, test_size=0.25)
    X_np = X_trainMML.to_numpy()

    Y_np = Y_trainMML.to_numpy()
    le = LabelEncoder()
    le.fit(Y_np)

    Y_le = le.transform(Y_np)
    k = 5  # Number of folds

    np.random.seed(1234)
    myfolds = KFold(n_splits=k)

    algsMML = [LogitNet(), RandomForestClassifier(), LogisticRegression(solver='lbfgs'), DecisionTreeClassifier(), GradientBoostingClassifier()]

    models = {}
    finalModels = {}

    algs_cols = ["Fold", "Algorithm", "AUC_Percent", "Accuracy_Percent", "Balanced_Accuracy_Percent", "Log_Loss", "Sensitivity", "Specificity", "PPV", "NPV", "Runtime_Seconds"]
    # for each fold, do the training with all the algorithms included in the variable algs
    algs_table = pd.DataFrame(columns=algs_cols)

    folds_table = pd.DataFrame(columns=algs_cols)

    for i in range(1, k + 1):
        models[i] = {}

    for alg in algsMML:

        name = alg.__class__.__name__

        print()
        print("#" * 30)
        print(name + "\n")

        fold = 1

        for train_index, test_index in myfolds.split(X_np):

            print()
            print("\t" + "#" * 26)
            print("\t" + "Fold " + str(fold) + "\n")
            start_time = time.time()
            np.random.seed(1234)

            X_train, X_test = X_np[train_index], X_np[test_index]
            Y_train, Y_test = Y_le[train_index], Y_le[test_index]

            alg.fit(X_train, Y_train)

            models[fold][name] = alg

            test_predictions = alg.predict_proba(X_test)
            rocauc = roc_auc_score(Y_test, test_predictions[:, 1])
            print("\t\tAUC: {:.4%}".format(rocauc))

            test_predictions = alg.predict(X_test)
            acc = accuracy_score(Y_test, test_predictions)
            print("\t\tAccuracy: {:.4%}".format(acc))

            test_predictions = alg.predict(X_test)
            balacc = balanced_accuracy_score(Y_test, test_predictions)
            print("\t\tBalanced Accuracy: {:.4%}".format(balacc))

            test_predictions = alg.predict(X_test)
            kappa = cohen_kappa_score(Y_test, test_predictions)
            print("\t\tKappa: {:.4%}".format(kappa))

            CM = confusion_matrix(Y_test, test_predictions)
            TN = CM[0][0]
            FN = CM[1][0]
            TP = CM[1][1]
            FP = CM[0][1]
            sensitivity = TP / (TP + FN)
            specificity = TN / (TN + FP)
            PPV = TP / (TP + FP)
            NPV = TN / (TN + FN)

            test_predictions = alg.predict_proba(X_test)
            ll = log_loss(Y_test, test_predictions)
            print("\t\tLog Loss: {:.4}".format(ll))

            end_time = time.time()
            elapsed_time = (end_time - start_time)
            print("\t\tRuntime in seconds: {:.4}".format(elapsed_time))

            folds_entry = pd.DataFrame([[fold, name, rocauc * 100, acc * 100, balacc * 100, ll, sensitivity, specificity, PPV, NPV, elapsed_time]], columns=algs_cols)
            folds_table = folds_table.append(folds_entry)

            fold = fold + 1
            # saveRDS(model, paste0(workPath, "/alg-",alg,"-type-", type, ".rds"))

    print(folds_table)

    # Save models
    with open(workPath + "/models_MML.pydat", 'wb') as finalModels_file:
        pickle.dump(finalModels, finalModels_file)

    folds_table.index = range(len(folds_table.index))

    for alg in algsMML:
        name = alg.__class__.__name__
        alg_best_entry = folds_table.iloc[folds_table[folds_table["Algorithm"] == name]['Balanced_Accuracy_Percent'].idxmax()]
        finalModels[name] = models[alg_best_entry["Fold"]][name]
        algs_table = algs_table.append(alg_best_entry)

    print("\n\n")
    algs_table.index = range(len(algs_table.index))
    print(algs_table)

    print("\n")

    bestAlg = algs_table.iloc[algs_table['Balanced_Accuracy_Percent'].idxmax()]['Algorithm']

    expert_L2 = {}
    expert_L2["model"] = finalModels[bestAlg]
    expert_L2["columns"] = metaSet.columns
    metaSet["ID"] = ID
    with open(workPath + "/metaSet.pydat", 'wb') as metaSet_file:
        pickle.dump(metaSet, metaSet_file)

    with open(workPath + "/expert_L2.pydat", 'wb') as expert_L2_file:
        pickle.dump(expert_L2, expert_L2_file)

    with open(workPath + "/metaResults.pydat", 'wb') as metaResults_file:
        pickle.dump(folds_table, metaResults_file)

    return expert_L2


# Function to obtain the common columns from an array of dataframes
def commonColumns(dataFrames):
    # Initialize commonColumns
    commonColumns = dataFrames[0].columns

    # For each df, intersect columns
    for i in range(1, len(dataFrames)):
        commonColumns = np.intersect1d(commonColumns, dataFrames[i].columns)

    # Return commonColumns
    return commonColumns


# Function to merge an array of data frames, if columns indicated, merge only the indicated columns
def bindRows(dataFrames, columns=None):
    # If no columns indicated
    if columns is None:
        # Initialize df
        finalDataFrame = dataFrames[0]
        # For each data frame, merge it
        for i in range(1, len(dataFrames)):
            finalDataFrame = finalDataFrame.append(dataFrames[i])

        # Rename index
        finalDataFrame.index = range(len(finalDataFrame.index))
        # Return merged df
        return finalDataFrame
    else:
        # Initialize df
        finalDataFrame = dataFrames[0][columns]
        # For each data frame, merge it
        for i in range(1, len(dataFrames)):
            finalDataFrame = finalDataFrame.append(dataFrames[i][columns])

        # Rename index
        finalDataFrame.index = range(len(finalDataFrame.index))
        # Return merged df
        return finalDataFrame


#----------------------------------------#
#-----------trainAndTestMML_nSmE---------#
#----------------------------------------#


def trainAndTestMML_nSmE(experts_L1,
                         genoTrain,
                         handlersML,
                         workPath="/home/rafa/MetaLearning/metaML/Modelos/",
                         private=True):

    predSets = {}
    metaSets = {}
    i = 1

    for key in handlersML.keys():
        # dataTrain = pd.read_csv(handlerMLdata["train1mldata"], header=0, delim_whitespace=True)
        dataTest = pd.read_csv(handlersML[key]["test1mldata"], header=0, delim_whitespace=True)

        dataRepo = dataTest

        dataRepo = dataRepo.loc[dataRepo["AGE"] != -9]
        ID = dataRepo["ID"]
        dataRepo.drop("ID", 1, inplace=True)

        Y = dataRepo["PHENO"]
        dataRepo.drop("PHENO", 1, inplace=True)

        Y.index = range(len(Y.index))
        ID.index = range(len(ID.index))

        metaSet = pd.concat([ID, Y], axis=1)
        predSets[i] = dataRepo

        j = 1
        for expertKey in experts_L1.keys():
            aux = dataRepo.copy()
            diff1 = set(experts_L1[expertKey]["columns"]) - set(aux.columns)  # each expert has the same variables, so we use the first one, for example

            if len(diff1) > 0:
                new_cols = pd.DataFrame(0, index=np.arange(len(aux)), columns=diff1)

                aux.index = range(len(aux.index))
                aux = pd.concat([aux, new_cols], axis=1)

            diff2 = set(aux.columns) - set(experts_L1[expertKey]["columns"])
            if len(diff2) > 0:
                aux.drop(diff2, 1, inplace=True)

            preds = experts_L1[expertKey]["model"].predict(aux)
            metaSet["Pred-" + str(j)] = preds
            j = j + 1

        metaSets[i] = metaSet

        i = i + 1

    # Merge all predictions in one data frame
    metaSet = bindRows(metaSets)
    metaSet.index = range(len(metaSet.index))

    # If data isnt private, we can merge geno data in the training
    if not private:
        # We assume MGSA type (we add age, sex and geno)
        # First we find common columns between SNPs in each repository
        cols = commonColumns(predSets)
        cols = [col for col in cols if not col.startswith("PC")]

        # Merge all SNPs in one data frame
        geno_cols = bindRows(predSets, cols)
        geno_cols.index = range(len(geno_cols.index))

        # Merge predict data frame with geno data frame
        metaSet = pd.concat([metaSet, geno_cols], axis=1)

    ID = metaSet["ID"]
    metaSet.drop("ID", 1, inplace=True)
    metaSet.drop("PHENO", 1, inplace=True)

    # separate the new dataframe into train and test
    X_trainMML, X_testMML, Y_trainMML, Y_testMML = train_test_split(metaSet, Y, test_size=0.25)
    X_np = X_trainMML.to_numpy()

    Y_np = Y_trainMML.to_numpy()
    le = LabelEncoder()
    le.fit(Y_np)

    Y_le = le.transform(Y_np)
    k = 5  # Number of folds

    np.random.seed(1234)
    myfolds = KFold(n_splits=k)

    # algsMML = c("glmnet", "C5.0Tree", "earth", "svmRadial", "rf", "xgbTree", "xgbLinear", "xgbDART", "gbm")
    algsMML = [LogitNet(), RandomForestClassifier(), LogisticRegression(solver='lbfgs'), DecisionTreeClassifier(), GradientBoostingClassifier()]

    models = {}
    finalModels = {}

    algs_cols = ["Fold", "Algorithm", "AUC_Percent", "Accuracy_Percent", "Balanced_Accuracy_Percent", "Log_Loss", "Sensitivity", "Specificity", "PPV", "NPV", "Runtime_Seconds"]
    # for each fold, do the training with all the algorithms included in the variable algs
    algs_table = pd.DataFrame(columns=algs_cols)

    folds_table = pd.DataFrame(columns=algs_cols)

    for i in range(1, k + 1):
        models[i] = {}

    for alg in algsMML:

        name = alg.__class__.__name__

        print()
        print("#" * 30)
        print(name + "\n")

        fold = 1

        for train_index, test_index in myfolds.split(X_np):

            print()
            print("\t" + "#" * 26)
            print("\t" + "Fold " + str(fold) + "\n")
            start_time = time.time()
            np.random.seed(1234)

            X_train, X_test = X_np[train_index], X_np[test_index]
            Y_train, Y_test = Y_le[train_index], Y_le[test_index]

            alg.fit(X_train, Y_train)

            models[fold][name] = alg

            test_predictions = alg.predict_proba(X_test)
            rocauc = roc_auc_score(Y_test, test_predictions[:, 1])
            print("\t\tAUC: {:.4%}".format(rocauc))

            test_predictions = alg.predict(X_test)
            acc = accuracy_score(Y_test, test_predictions)
            print("\t\tAccuracy: {:.4%}".format(acc))

            test_predictions = alg.predict(X_test)
            balacc = balanced_accuracy_score(Y_test, test_predictions)
            print("\t\tBalanced Accuracy: {:.4%}".format(balacc))

            test_predictions = alg.predict(X_test)
            kappa = cohen_kappa_score(Y_test, test_predictions)
            print("\t\tKappa: {:.4%}".format(kappa))

            CM = confusion_matrix(Y_test, test_predictions)
            TN = CM[0][0]
            FN = CM[1][0]
            TP = CM[1][1]
            FP = CM[0][1]
            sensitivity = TP / (TP + FN)
            specificity = TN / (TN + FP)
            PPV = TP / (TP + FP)
            NPV = TN / (TN + FN)

            test_predictions = alg.predict_proba(X_test)
            ll = log_loss(Y_test, test_predictions)
            print("\t\tLog Loss: {:.4}".format(ll))

            end_time = time.time()
            elapsed_time = (end_time - start_time)
            print("\t\tRuntime in seconds: {:.4}".format(elapsed_time))

            folds_entry = pd.DataFrame([[fold, name, rocauc * 100, acc * 100, balacc * 100, ll, sensitivity, specificity, PPV, NPV, elapsed_time]], columns=algs_cols)
            folds_table = folds_table.append(folds_entry)

            fold = fold + 1
            # saveRDS(model, paste0(workPath, "/alg-",alg,"-type-", type, ".rds"))

    print(folds_table)

    # Save models
    with open(workPath + "/models_MML.pydat", 'wb') as finalModels_file:
        pickle.dump(finalModels, finalModels_file)

    folds_table.index = range(len(folds_table.index))

    for alg in algsMML:
        name = alg.__class__.__name__
        alg_best_entry = folds_table.iloc[folds_table[folds_table["Algorithm"] == name]['Balanced_Accuracy_Percent'].idxmax()]
        finalModels[name] = models[alg_best_entry["Fold"]][name]
        algs_table = algs_table.append(alg_best_entry)

    print("\n\n")
    algs_table.index = range(len(algs_table.index))
    print(algs_table)

    print("\n")

    bestAlg = algs_table.iloc[algs_table['Balanced_Accuracy_Percent'].idxmax()]['Algorithm']

    expert_L2 = {}
    expert_L2["model"] = finalModels[bestAlg]
    expert_L2["columns"] = metaSet.columns
    metaSet["ID"] = ID
    with open(workPath + "/metaSet.pydat", 'wb') as metaSet_file:
        pickle.dump(metaSet, metaSet_file)

    with open(workPath + "/expert_L2.pydat", 'wb') as expert_L2_file:
        pickle.dump(expert_L2, expert_L2_file)

    with open(workPath + "/metaResults.pydat", 'wb') as metaResults_file:
        pickle.dump(folds_table, metaResults_file)

    return expert_L2

#----------------------------------------#
#-------------prepareFinalTest-----------#
#----------------------------------------#


def prepareFinalTest(workPath,
                     path2Geno,
                     path2Covs,
                     predictor,
                     snpsToPull,
                     path2plink="/home/edusal/packages/",
                     addit="NA"):

    command = "cp " + path2Covs + ".cov " + workPath
    print("The command to run: " + command)
    os.system(command)

    handler = fm.getHandlerToGenotypeData(geno=path2Geno,
                                          covs=path2Covs,
                                          id="IID",
                                          fid="FID",
                                          predictor=predictor,
                                          pheno=workPath + "/MyPhenotype")

    geno = os.path.basename(handler["geno"])
    pheno = os.path.basename(handler["pheno"])
    cov = os.path.basename(handler["covs"])
    path2Genotype = os.path.dirname(handler["geno"]) + "/"
    prefix = "g-" + geno + "-p-" + pheno + "-c-" + cov + "-a-" + addit
    fprefix = workPath + "/" + prefix

    handler["snpsToPull"] = snpsToPull  # set the SNPs to pull to the ones selected in the spanish set

    command = path2plink + "plink --bfile " + path2Genotype + geno + " --extract " +\
        handler["snpsToPull"] + " --recode A --out " + fprefix + ".reduced_genos"

    print("Running command " + command + "\n")
    os.system(command)

    # exports SNP list for extraction in validation set
    command = "cut -f 1 " + handler["snpsToPull"] + " > " + fprefix + ".reduced_genos_snpList"

    print("Running command " + command + "\n")
    os.system(command)

    handler["rgenosSnpList"] = fprefix + ".reduced_genos_snpList"

    # we generate de .dataForML file

    mldatahandler = fm.fromSNPs2MLdata(handler=handler,
                                       addit="NA",
                                       path2plink=path2plink,
                                       predictor=predictor)

    with open(workPath + "/mldatahandler.pydat", 'wb') as mldatahandler_file:
        pickle.dump(mldatahandler, mldatahandler_file)

    return mldatahandler

#----------------------------------------#
#--------------finalTest_1SmE------------#
#----------------------------------------#


def finalTest_1SmE(workPath,
                   expert_L2,
                   handlerTest,
                   experts_L1):

    dataFinal = pd.read_csv(handlerTest["train1mldata"], header=0, delim_whitespace=True)

    # same preprocess applied to the training
    ID = dataFinal["ID"]
    dataFinal.drop("ID", 1, inplace=True)

    Y = dataFinal["PHENO"]
    dataFinal.drop("PHENO", 1, inplace=True)

    Y_np = Y.to_numpy()
    le = LabelEncoder()
    le.fit(Y_np)
    Y_le = le.transform(Y_np)

    Y.index = range(len(Y.index))
    ID.index = range(len(ID.index))

    # dataframe which will contain the predictions from each model
    metaSet = pd.concat([ID, Y], axis=1)

    aux = dataFinal

    # remove variables in the test data and not in the data used for training each expert
    # add variables set to zero present in each expert but not in the test data
    diff1 = set(experts_L1["columns"]) - set(aux.columns)  # each expert has the same variables, so we use the first one, for example

    if len(diff1) > 0:
        new_cols = pd.DataFrame(0, index=np.arange(len(aux)), columns=diff1)

        aux.index = range(len(aux.index))
        aux = pd.concat([aux, new_cols], axis=1)

    diff2 = set(aux.columns) - set(experts_L1["columns"])
    if len(diff2) > 0:
        aux.drop(diff2, 1, inplace=True)

    # As argument in future version
    algsML = [LogitNet(), RandomForestClassifier(), LogisticRegression(solver='lbfgs'), DecisionTreeClassifier(), GradientBoostingClassifier()]

    for alg in algsML:
        name = alg.__class__.__name__

        preds = experts_L1[name].predict(aux)
        metaSet["Pred-" + name] = preds

    # We assume its type MGSA and add age, sex and geno
    cols = dataFinal.columns
    cols = [col for col in cols if not col.startswith("PC")]
    all_cols = dataFinal[cols]

    metaSet.index = range(len(metaSet.index))
    all_cols.index = range(len(all_cols.index))

    metaSet = pd.concat([metaSet, all_cols], axis=1)

    # saveRDS(metaSet, paste0(workPath, "/metaSet-ANTES-FINAL-tipo",type,".rds"))

    ID = metaSet["ID"]
    metaSet.drop("ID", 1, inplace=True)
    metaSet.drop("PHENO", 1, inplace=True)
    # remove variables in the metaset and not in the data used for training the MML model
    # add variables set to zero present in the MML model but not in the data
    aux = metaSet

    diff1 = set(expert_L2["columns"]) - set(aux.columns)
    # print("diff1")
    # print(diff1)
    if len(diff1) > 0:
        new_cols = pd.DataFrame(0, index=np.arange(len(aux)), columns=diff1)

        aux.index = range(len(aux.index))
        aux = pd.concat([aux, new_cols], axis=1)

    diff2 = set(aux.columns) - set(expert_L2["columns"])
    if len(diff2) > 0:
        aux.drop(diff2, 1, inplace=True)

    # Put the columns in the right order
    metaSet = aux[expert_L2["columns"].tolist()]

    # print(metaSet)

    with open(workPath + "/metaSet.pydat", 'wb') as metaSet_file:
        pickle.dump(metaSet, metaSet_file)

    # get final results

    test_predictions = expert_L2["model"].predict_proba(metaSet)
    rocauc = roc_auc_score(Y_le, test_predictions[:, 1])
    print("\t\tAUC: {:.4%}".format(rocauc))

    test_predictions = expert_L2["model"].predict(metaSet)
    acc = accuracy_score(Y_le, test_predictions)
    print("\t\tAccuracy: {:.4%}".format(acc))

    test_predictions = expert_L2["model"].predict(metaSet)
    balacc = balanced_accuracy_score(Y_le, test_predictions)
    print("\t\tBalanced Accuracy: {:.4%}".format(balacc))

    test_predictions = expert_L2["model"].predict(metaSet)
    kappa = cohen_kappa_score(Y_le, test_predictions)
    print("\t\tKappa: {:.4%}".format(kappa))

    CM = confusion_matrix(Y_le, test_predictions)

    test_predictions = expert_L2["model"].predict_proba(metaSet)
    ll = log_loss(Y_le, test_predictions)
    print("\t\tLog Loss: {:.4}".format(ll))

    # save final results
    finalResults = {}
    finalResults["PHENO"] = Y
    finalResults["AUC"] = rocauc
    finalResults["Accuracy"] = acc
    finalResults["Balanced Accuracy"] = balacc
    finalResults["Kappa"] = kappa

    finalResults["confMat"] = CM

    return(finalResults)

#----------------------------------------#
#--------------finalTest_nSmE------------#
#----------------------------------------#


def finalTest_nSmE(workPath,
                   expert_L2,
                   handlerTest,
                   experts_L1):

    dataFinal = pd.read_csv(handlerTest["train1mldata"], header=0, delim_whitespace=True)

    # same preprocess applied to the training
    ID = dataFinal["ID"]
    dataFinal.drop("ID", 1, inplace=True)

    Y = dataFinal["PHENO"]
    dataFinal.drop("PHENO", 1, inplace=True)

    Y_np = Y.to_numpy()
    le = LabelEncoder()
    le.fit(Y_np)
    Y_le = le.transform(Y_np)

    Y.index = range(len(Y.index))
    ID.index = range(len(ID.index))

    # dataframe which will contain the predictions from each model
    metaSet = pd.concat([ID, Y], axis=1)

    # remove variables in the test data and not in the data used for training each expert
    # add variables set to zero present in each expert but not in the test data
    j = 1
    for expertKey in experts_L1.keys():
        aux = dataFinal.copy()
        diff1 = set(experts_L1[expertKey]["columns"]) - set(aux.columns)  # each expert has the same variables, so we use the first one, for example

        if len(diff1) > 0:
            new_cols = pd.DataFrame(0, index=np.arange(len(aux)), columns=diff1)

            aux.index = range(len(aux.index))
            aux = pd.concat([aux, new_cols], axis=1)

        diff2 = set(aux.columns) - set(experts_L1[expertKey]["columns"])
        if len(diff2) > 0:
            aux.drop(diff2, 1, inplace=True)

        preds = experts_L1[expertKey]["model"].predict(aux)
        metaSet["Pred-" + str(j)] = preds
        j = j + 1

    # We assume its type MGSA and add age, sex and geno
    cols = dataFinal.columns
    cols = [col for col in cols if not col.startswith("PC")]
    all_cols = dataFinal[cols]

    metaSet.index = range(len(metaSet.index))
    all_cols.index = range(len(all_cols.index))

    metaSet = pd.concat([metaSet, all_cols], axis=1)

    # saveRDS(metaSet, paste0(workPath, "/metaSet-ANTES-FINAL-tipo",type,".rds"))

    ID = metaSet["ID"]
    metaSet.drop("ID", 1, inplace=True)
    metaSet.drop("PHENO", 1, inplace=True)
    # remove variables in the metaset and not in the data used for training the MML model
    # add variables set to zero present in the MML model but not in the data
    aux = metaSet

    diff1 = set(expert_L2["columns"]) - set(aux.columns)
    if len(diff1) > 0:
        new_cols = pd.DataFrame(0, index=np.arange(len(aux)), columns=diff1)

        aux.index = range(len(aux.index))
        aux = pd.concat([aux, new_cols], axis=1)

    diff2 = set(aux.columns) - set(expert_L2["columns"])
    if len(diff2) > 0:
        aux.drop(diff2, 1, inplace=True)

    # Put the columns in the right order
    metaSet = aux[expert_L2["columns"].tolist()]

    # print(metaSet)

    with open(workPath + "/metaSet.pydat", 'wb') as metaSet_file:
        pickle.dump(metaSet, metaSet_file)

    # get final results

    test_predictions = expert_L2["model"].predict_proba(metaSet)
    rocauc = roc_auc_score(Y_le, test_predictions[:, 1])
    print("\t\tAUC: {:.4%}".format(rocauc))

    test_predictions = expert_L2["model"].predict(metaSet)
    acc = accuracy_score(Y_le, test_predictions)
    print("\t\tAccuracy: {:.4%}".format(acc))

    test_predictions = expert_L2["model"].predict(metaSet)
    balacc = balanced_accuracy_score(Y_le, test_predictions)
    print("\t\tBalanced Accuracy: {:.4%}".format(balacc))

    test_predictions = expert_L2["model"].predict(metaSet)
    kappa = cohen_kappa_score(Y_le, test_predictions)
    print("\t\tKappa: {:.4%}".format(kappa))

    CM = confusion_matrix(Y_le, test_predictions)

    test_predictions = expert_L2["model"].predict_proba(metaSet)
    ll = log_loss(Y_le, test_predictions)
    print("\t\tLog Loss: {:.4}".format(ll))

    # save final results
    finalResults = {}
    finalResults["PHENO"] = Y
    finalResults["AUC"] = rocauc
    finalResults["Accuracy"] = acc
    finalResults["Balanced Accuracy"] = balacc
    finalResults["Kappa"] = kappa

    finalResults["confMat"] = CM

    return(finalResults)
