# pyright: basic

import time
from pprint import pprint
from typing import List

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import average_precision_score, mean_squared_error
from sklearn.naive_bayes import CategoricalNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from aliases import (
    CO_OCCUR,
    FEATURES,
    IS_CO_COMP,
    MAE,
    MODEL,
    MSE,
    PROTEIN_U,
    PROTEIN_V,
    SCENARIO,
    STRING,
    SUPER_FEATS,
    SWC_FEATS,
    TOPO,
    TOPO_L2,
    WEIGHT,
)
from assertions import assert_df_bounded, assert_no_zero_weight
from model_preprocessor import ModelPreprocessor
from supervised_weighting import SupervisedWeighting
from utils import (
    construct_composite_network,
    get_cyc_comp_pairs,
    get_cyc_train_test_comp_pairs,
)


class CoCompEdgesEvaluator:
    """
    Compare the performance of the following methods
    in terms of predicting co-complex edges
    1. SWC
    2. SCCP_SWC
    3. SCCP
    """

    def __init__(self, n_iters: int):
        self.n_iters = n_iters

        self.model_prep = ModelPreprocessor()
        df_composite = construct_composite_network()

        # normalized protein networks
        self.df_composite = self.model_prep.normalize_features(df_composite, FEATURES)
        self.df_composite_swc = self.model_prep.normalize_features(
            df_composite, SWC_FEATS
        )

        comp_pairs = get_cyc_comp_pairs()
        # True labels
        self.y_true = (
            self.model_prep.label_composite(
                df_composite=self.df_composite,
                df_positive_pairs=comp_pairs,
                label=IS_CO_COMP,
                seed=0,
                mode="all",
                balanced=False,
            )
            .select(IS_CO_COMP)
            .to_numpy()
            .ravel()
        )

    def main(self):
        sccp_swc = SupervisedWeighting(
            RandomForestClassifier(
                n_estimators=2000,
                criterion="entropy",
                max_features="sqrt",
                n_jobs=-1,
                random_state=6789,
            ),
            "SCCP_SWC",
        )

        sccp = SupervisedWeighting(
            RandomForestClassifier(
                n_estimators=2000,
                criterion="entropy",
                max_features="sqrt",
                n_jobs=-1,
                random_state=6789,
            ),
            "SCCP",
        )

        for xval_iter in range(self.n_iters):
            print()
            print(f"------------------- BEGIN: ITER {xval_iter} ---------------------")
            df_train_pairs, df_test_pairs = get_cyc_train_test_comp_pairs(xval_iter)
            df_train_labeled = self.model_prep.label_composite(
                self.df_composite,
                df_train_pairs,
                IS_CO_COMP,
                xval_iter,
                "subset",
                False,
            )

            df_w_swc = pl.read_csv(
                f"../data/swc/raw_weighted/swc scored_edges iter{xval_iter}.txt",
                new_columns=[PROTEIN_U, PROTEIN_V, WEIGHT],
                separator=" ",
            )
            df_w_sccp_swc = sccp_swc.main(
                self.df_composite_swc, df_train_labeled, xval_iter
            )
            df_w_sccp = sccp.main(self.df_composite, df_train_labeled, xval_iter)
            w_networks = [df_w_swc, df_w_sccp_swc, df_w_sccp]

            for model in models:
                weighted.append(
                    model.main(self.df_composite, df_train_labeled, xval_iter)
                )

            print("Performance evaluations...")
            print()
            for idx, w in enumerate(weighted):
                y_pred = w.select(WEIGHT).to_numpy().ravel()
                rmse = mean_squared_error(self.y_true, y_pred) ** (0.5)
                ap = average_precision_score(self.y_true, y_pred)
                ap_list[models[idx].name].append(ap)
                rmse_list[models[idx].name].append(rmse)
                print(f"[{models[idx].name}] RMSE: {rmse}")
                print(f"[{models[idx].name}] AP: {ap}")

            print(f"---------------- END: ITER {xval_iter} ---------------")
            print()

        print()
        print()
        print("Final evaluations")
        avg_ap = {
            model.name: sum(ap_list[model.name]) / self.n_iters for model in models
        }
        avg_rmse = {
            model.name: sum(rmse_list[model.name]) / self.n_iters for model in models
        }
        print(f"AP: {avg_ap}")
        print(f"RMSE: {avg_rmse}")


if __name__ == "__main__":
    start = time.time()
    n_iters = 10

    evaluator = CoCompEdgesEvaluator(n_iters)
    evaluator.main()
    print("===================================")
    print(f"Execution time: {time.time() - start}")
