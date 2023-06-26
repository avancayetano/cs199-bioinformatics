# pyright: basic

from typing import List, TypedDict

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from sklearn.metrics import auc, brier_score_loss, log_loss, precision_recall_curve

from aliases import (
    BRIER_SCORE,
    F1_SCORE,
    FEATURES,
    IS_CO_COMP,
    LOG_LOSS,
    METHOD,
    PR_AUC,
    PRECISION,
    PROTEIN_U,
    PROTEIN_V,
    RECALL,
    SUPER_FEATS,
    WEIGHT,
    XVAL_ITER,
)
from model_preprocessor import ModelPreprocessor
from utils import (
    construct_composite_network,
    get_cyc_comp_pairs,
    get_cyc_train_test_comp_pairs,
    get_weighted_filename,
)


class CompEdgesEvaluator:
    def __init__(self, dip: bool):
        self.sv_methods = ["SWC", "XGW"]
        self.feat_methods = FEATURES + [method["name"] for method in SUPER_FEATS]
        self.methods = self.sv_methods + self.feat_methods + ["unweighted"]
        self.n_iters = 10

        self.dip = dip

        model_prep = ModelPreprocessor()
        df_composite = construct_composite_network(dip=self.dip)
        comp_pairs = get_cyc_comp_pairs()
        self.df_labeled = model_prep.label_composite(
            df_composite, comp_pairs, IS_CO_COMP, -1, "all", False
        )

    def main(self):
        evals = []
        print(f"Evaluating on these ({len(self.methods)}) methods: {self.methods}")
        print()
        for xval_iter in range(self.n_iters):
            print(f"Evaluating cross-val iteration: {xval_iter}")
            df_train_pairs, _ = get_cyc_train_test_comp_pairs(xval_iter)
            df_composite_test = self.df_labeled.join(
                df_train_pairs, on=[PROTEIN_U, PROTEIN_V], how="anti"
            )
            for method in self.methods:
                path = get_weighted_filename(
                    method.lower(),
                    method in self.sv_methods,
                    self.dip,
                    xval_iter,
                )

                df_w = pl.read_csv(
                    path,
                    has_header=False,
                    separator="\t",
                    new_columns=[PROTEIN_U, PROTEIN_V, WEIGHT],
                )

                df_pred = df_composite_test.join(
                    df_w, on=[PROTEIN_U, PROTEIN_V], how="inner"
                )

                y_true = df_pred.select(IS_CO_COMP).to_numpy().ravel()
                y_pred = df_pred.select(WEIGHT).to_numpy().ravel()

                brier_score = brier_score_loss(y_true, y_pred, pos_label=1)
                log_loss_metric = log_loss(y_true, y_pred)

                precision, recall, thresholds = precision_recall_curve(
                    y_true, y_pred, pos_label=1
                )
                pr_auc = auc(recall, precision)

                evals.append(
                    {
                        METHOD: method,
                        XVAL_ITER: xval_iter,
                        LOG_LOSS: log_loss_metric,
                        BRIER_SCORE: brier_score,
                        PR_AUC: pr_auc,
                    }
                )

            print()

        df_evals = (
            pl.DataFrame(evals)
            .groupby(METHOD)
            .mean()
            .select(pl.exclude(XVAL_ITER))
            .sort([LOG_LOSS, BRIER_SCORE, PR_AUC], descending=[False, False, True])
        )
        print()
        print("Average of all evaluations on all the cross-val iterations")
        print(df_evals)


if __name__ == "__main__":
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_rows(20)

    print(
        "------------------------ Evaluating the composite network --------------------"
    )
    evaluator = CompEdgesEvaluator(dip=False)
    evaluator.main()
    print()

    print(
        "------------------------ Evaluating the DIP composite network --------------------"
    )
    evaluator = CompEdgesEvaluator(dip=True)
    evaluator.main()
    print()
