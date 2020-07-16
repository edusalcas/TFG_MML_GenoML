
import pandas as pd
import genoml_utils as gml
import os
import joblib
import ml_utils as mlu
import pickle
from abc import ABC, abstractmethod
from sklearn import metrics

# Sufixes
suf_tuned_model = ".tunedModel.joblib"
suf_trained_model = ".trainedModel.joblib"
suf_dataForML = ".dataForML.h5"
suf_tuned_mean = ".tunedMean.txt"
suf_trained_mean = ".trainedMean.txt"
suf_best_algo = ".best_algorithm.txt"
suf_out = '.mean_confidence_interval.txt'


def save_obj(file_name, data):
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)


def read_obj(file_name):
    with open(file_name, 'rb') as file:
        data = pickle.load(file)

    return data


class MML(ABC):

    def __init__(self, path):
        self.hdf_key = 'dataForML'
        self.experts_columns = {}
        self.meta_expert_columns = []
        self.meta_model = None

        columns = ['Iteration', 'Algorithm', 'Trained Mean', 'Tuned Mean', 'Test score']
        self.results_df = pd.DataFrame(columns=columns)
        columns = ['Iteration', 'Expert', 'Trained Mean', 'Tuned Mean']
        self.experts_res_df = pd.DataFrame(columns=columns)

        self.means = []

        self.experts_columns_file = path + 'experts_columns.pydat'
        self.meta_columns_file = path + 'meta_columns.pydat'

        self.load_columns()

    def save_columns(self):
        experts_columns = self.experts_columns

        save_obj(self.experts_columns_file, experts_columns)
        save_obj(self.meta_columns_file, self.meta_expert_columns)

    def load_columns(self):
        if not os.path.isfile(self.experts_columns_file):
            return

        experts_columns = read_obj(self.experts_columns_file)
        meta_columns = read_obj(self.meta_columns_file)[0]

        self.experts_columns = experts_columns
        self.meta_expert_columns = meta_columns

    @abstractmethod
    def gen_experts(self,
                    path, 
                    expert_names=None, 
                    prefix=None, 
                    metric_max=None, 
                    seed=None, 
                    metric_tune=None, 
                    max_tune=None, 
                    algs=None):
        pass

    @abstractmethod
    def gen_metaset(self, 
                    data, 
                    experts_path, 
                    experts_names):
        pass

    def gen_pred_metaset(self, data, experts_path, experts_names, path, meta_prefix):
        meta_set = pd.DataFrame(data['ID']).copy()

        # Predict with experts
        for name in experts_names:
            # Fix columns to expert columns
            columns = self.experts_columns[name]
            df = mlu.fix_columns(data, columns)
            # Load level 1 expert
            if os.path.isfile(experts_path + suf_tuned_model):
                expert = joblib.load(experts_path + name + suf_tuned_model)
            else:
                expert = joblib.load(experts_path + name + suf_trained_model)

            # Predict and save it
            pred = expert.predict(df.drop(['PHENO', 'ID'], 1))
            meta_set["Pred-" + name] = pred

        # Fix columns to meta expert
        common_columns_df = mlu.common_columns(data, self.meta_expert_columns)
        # Merge data
        meta_set = meta_set.merge(common_columns_df, on='ID')

        # Reorder columns
        meta_set = meta_set[self.meta_expert_columns]
        print("Meta-set: ")
        print(meta_set)

        return meta_set

    def gen_meta_expert(self, data, path, experts_names, prefix, metric_max, algs, seed, metric_tune, max_tune):
        # Generate Meta-Set
        meta_set = self.gen_metaset(data, path, experts_names)
        meta_set.to_hdf(path + prefix + suf_dataForML, key=self.hdf_key)

        # Train and tune meta expert
        gml.train(path + prefix, metric_max, algs, seed)
        gml.tune(path + prefix, metric_tune, max_tune, seed)

        meta_model = None

        if os.path.isfile(path + prefix + suf_tuned_model):
            meta_model = joblib.load(path + prefix + suf_tuned_model)
        else:
            meta_model = joblib.load(path + prefix + suf_trained_model)

        self.meta_model = meta_model

    def predict(self, data, path, experts_names, meta_prefix, iteration=0):
        meta_set = self.gen_pred_metaset(data, path, experts_names, path, meta_prefix)

        meta_model = None

        if os.path.isfile(path + meta_prefix + suf_tuned_model):
            meta_model = joblib.load(path + meta_prefix + suf_tuned_model)
        else:
            meta_model = joblib.load(path + meta_prefix + suf_trained_model)

        Y_test = meta_set['PHENO']
        test_predictions = meta_model.predict(meta_set.drop(['PHENO', 'ID'], 1))
        # TODO Return more stats
        balacc = metrics.balanced_accuracy_score(Y_test, test_predictions) * 100

        return balacc

    def get_results_files(self, path, prefix):
        tuned_mean_file = path + prefix + suf_tuned_mean
        trained_mean_file = path + prefix + suf_trained_mean
        best_algo_file = path + prefix + suf_best_algo

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

        return tuned_mean, trained_mean, best_algo

    def get_results(self, experts_names, path, prefix, test_result, iteration=0):

        # Experts results
        for name in experts_names:
            tuned_mean, trained_mean, best_algo = self.get_results_files(path, name)

            entry = {}
            entry = {'Iteration': iteration,
                     'Expert': name,
                     'Best_algo': best_algo,
                     'Trained Mean': trained_mean,
                     'Tuned Mean': tuned_mean}
            self.experts_res_df = self.experts_res_df.append(entry, ignore_index=True)

        # Meta results
        tuned_mean, trained_mean, best_algo = self.get_results_files(path, prefix)

        entry = {}
        entry = {'Iteration': iteration,
                 'Algorithm': best_algo,
                 'Trained Mean': trained_mean,
                 'Tuned Mean': tuned_mean,
                 'Test score': test_result}
        self.results_df = self.results_df.append(entry, ignore_index=True)

        if tuned_mean is None or tuned_mean < trained_mean:
            mean = trained_mean
        else:
            mean = tuned_mean

        self.means.append(mean)

        return trained_mean

    def save_results(self, path, meta_prefix, test_results):
        # Calculate mean and confidence interval
        mean, conf_down, conf_up = mlu.mean_confidence_interval(self.means)

        # Save results
        out_file = path + meta_prefix + suf_out
        file = open(out_file, "w+")
        file.write(f'{mean} {conf_down} {conf_up}')
        file.close()
        # Show results
        print()
        print('#' * 60)
        print(f"Results in training phase are:\n\t- Mean: {mean:.4f}\n\t- Conficence interval: {conf_down:.4f} - {conf_up:.4f}")
        print('#' * 60)

        # Calculate mean and confidence interval for test results
        mean, conf_down, conf_up = mlu.mean_confidence_interval(test_results)
        # Show results
        print()
        print('#' * 60)
        print(f"Results in testing phase are:\n\t- Mean: {mean:.4f}\n\t- Conficence interval: {conf_down:.4f} - {conf_up:.4f}")
        print('#' * 60)

        # Save result dfs
        self.results_df.to_csv(path + meta_prefix + '.all_means.csv')
        self.experts_res_df.to_csv(path + meta_prefix + '.experts.csv')
