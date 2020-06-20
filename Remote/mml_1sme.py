import pandas as pd
import os
import joblib

from mml import MML
import genoml_utils as gml
import ml_utils as mlu

suf_dataForML = ".dataForML.h5"
suf_tuned_model = ".tunedModel.joblib"
suf_trained_model = ".trainedModel.joblib"

class MML_1SmE(MML):

    def gen_experts(self, path, expert_names=None, prefix=None, metric_max=None, seed=None, metric_tune=None, max_tune=None, algs=None):
        data = pd.read_hdf(path + prefix + suf_dataForML, key=self.hdf_key)
        # For each algorithm
        for alg in expert_names:
            # Change data prefix
            mlu.move_data(path + prefix, path + alg)
            # Train and tune the expert
            gml.train(path + alg, metric_max, alg, seed)
            gml.tune(path + alg, metric_tune, max_tune, seed)
            # Change data prefix to default
            mlu.move_data(path + alg, path + prefix)

            self.experts_columns[alg] = list(data.columns)

    def gen_metaset(self, data, experts_path, experts_names):
        # Predict with experts
        meta_set = pd.DataFrame(data['ID'])
        for name in experts_names:
            # Load level 1 expert
            if os.path.isfile(experts_path + name + suf_tuned_model):
                expert = joblib.load(experts_path + name + suf_tuned_model)
            else:
                expert = joblib.load(experts_path + name + suf_trained_model)

            meta_set["Pred-" + str(name)] = expert.predict(data.drop(['PHENO', 'ID'], 1))

        # Merge data
        meta_set = meta_set.merge(data, on='ID')

        self.meta_expert_columns = list(meta_set.columns)

        self.save_columns()

        return meta_set
