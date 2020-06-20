import pandas as pd
import os
import joblib

from mml import MML
import genoml_utils as gml
import ml_utils as mlu

suf_dataForML = ".dataForML.h5"
suf_tuned_model = ".tunedModel.joblib"
suf_trained_model = ".trainedModel.joblib"

class MML_nSmE(MML):

    def gen_experts(self, path, expert_names=None, prefix=None, metric_max=None, seed=2020, metric_tune=None, max_tune=None, algs=None):
        # For each cohort
        for expert in expert_names:
            data = pd.read_hdf(path + expert + suf_dataForML, key=self.hdf_key)

            print(f"Training expert with data with shape: {data.shape}.")

            # Train and tune the expert
            gml.train(path + expert, metric_max, algs, seed)
            gml.tune(path + expert, metric_tune, max_tune, seed)

            # Save columns
            self.experts_columns[expert] = list(data.columns)

    def gen_metaset(self, data, experts_path, experts_names):
        meta_sets = {}
        pred_sets = {}
        # Predict with experts
        for i, df in enumerate(data):
            meta_set = df[['PHENO', 'ID']].copy()

            pred_sets[i] = df.drop(columns=['PHENO', 'ID'])
            for name in experts_names:
                # Load level 1 expert
                if os.path.isfile(experts_path + name + suf_tuned_model):
                    expert = joblib.load(experts_path + name + suf_tuned_model)
                else:
                    expert = joblib.load(experts_path + name + suf_trained_model)

                # Fix columns for predict
                expert_dataTest = mlu.fix_columns(df, self.experts_columns[name])
                expert_dataTest = expert_dataTest.drop(columns=['PHENO', 'ID'])

                meta_set["Pred-" + str(name)] = expert.predict(df.drop(['PHENO', 'ID'], 1))

            meta_sets[i] = meta_set

        # Join predictions
        meta_set = mlu.bind_rows(meta_sets)
        meta_set.index = range(len(meta_set.index))

        # If not private merge geno
        cols = mlu.common_columns(pred_sets)

        # Merge all SNPs in one data frame
        geno_cols = mlu.bind_rows(pred_sets, cols)
        geno_cols.index = range(len(geno_cols.index))

        # Merge predict data frame with geno data frame
        meta_set = pd.concat([meta_set, geno_cols], axis=1)

        meta_set.index = range(len(meta_set.index))

        self.meta_expert_columns = list(meta_set.columns)

        return meta_set
