# pyright: basic

from typing import List, TypedDict

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from sklearn.metrics import (
    PrecisionRecallDisplay,
    auc,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
)

from aliases import (
    BRIER_SCORE,
    F1_SCORE,
    FEATURES,
    IS_CO_COMP,
    LOG_LOSS,
    METHOD,
    METRIC,
    PR_AUC,
    PRECISION,
    PROTEIN_U,
    PROTEIN_V,
    RECALL,
    SUPER_FEATS,
    VALUE,
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

        sns.set_palette("deep")

    def main(self, re_eval: bool = True):
        prefix = "dip_" if self.dip else ""

        if re_eval:
            evals = []
            print(f"Evaluating on these ({len(self.methods)}) methods: {self.methods}")
            print()
            df_prec_recall = pl.DataFrame()
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
                    if xval_iter == 0:
                        df_pr = (
                            pl.from_numpy(precision, schema=[PRECISION])
                            .hstack(pl.from_numpy(recall, schema=[RECALL]))
                            .with_columns(
                                [
                                    pl.lit(method).alias(METHOD),
                                    pl.lit(xval_iter).alias(XVAL_ITER),
                                ]
                            )
                        )
                        df_prec_recall = pl.concat(
                            [df_prec_recall, df_pr], how="vertical"
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

            df_evals.write_csv(f"../data/evals/{prefix}comp_evals.csv", has_header=True)
            df_prec_recall.write_csv(
                f"../data/evals/{prefix}prec_recall_comp_evals.csv", has_header=True
            )
            print(df_prec_recall)

        # plots
        network = "DIP COMPOSITE NETWORK" if self.dip else "ORIGINAL COMPOSITE NETWORK"
        n_methods = 10
        df_evals = pl.read_csv(f"../data/evals/{prefix}comp_evals.csv", has_header=True)
        df_prec_recall = pl.read_csv(
            f"../data/evals/{prefix}prec_recall_comp_evals.csv", has_header=True
        )

        df_loss = (
            df_evals.head(n_methods)
            .select([METHOD, LOG_LOSS, BRIER_SCORE])
            .melt(id_vars=METHOD, variable_name=METRIC, value_name=VALUE)
        )

        print(df_evals)

        plt.figure()
        ax = sns.barplot(data=df_loss.to_pandas(), x=METHOD, y=VALUE, hue=METRIC)
        ax.set_title(
            f"{network}\nClassification of co-complex edges\nTop {n_methods} weighting methods in terms of log loss and Brier score loss."
        )
        plt.xticks(rotation=15)

        df_auc = (
            df_evals.sort(PR_AUC, descending=True)
            .head(n_methods)
            .select([METHOD, PR_AUC])
            .melt(id_vars=METHOD, variable_name=METRIC, value_name=VALUE)
        )
        plt.figure()
        ax = sns.barplot(data=df_auc.to_pandas(), x=METHOD, y=VALUE, hue=METRIC)
        ax.set_title(
            f"{network}\nClassification of co-complex edges\nTop {n_methods} weighting methods in terms of Precision-Recall AUC."
        )
        plt.xticks(rotation=15)

        ###################
        plt.figure()
        ax = sns.barplot(
            data=df_loss.filter(
                (pl.col(METHOD) == "XGW") | (pl.col(METHOD) == "SWC")
            ).to_pandas(),
            x=METHOD,
            y=VALUE,
            hue=METRIC,
        )
        ax.set_title(
            f"{network}\nClassification of co-complex edges\nXGW vs SWC in terms of log loss and Brier score loss."
        )
        plt.xticks(rotation=15)

        df_auc = (
            df_evals.sort(PR_AUC, descending=True)
            .head(n_methods)
            .select([METHOD, PR_AUC])
            .melt(id_vars=METHOD, variable_name=METRIC, value_name=VALUE)
        )
        plt.figure()
        ax = sns.barplot(
            data=df_auc.filter(
                (pl.col(METHOD) == "XGW") | (pl.col(METHOD) == "SWC")
            ).to_pandas(),
            x=METHOD,
            y=VALUE,
            hue=METRIC,
        )
        ax.set_title(
            f"{network}\nClassification of co-complex edges\nXGW vs SWC in terms of Precision-Recall AUC."
        )
        plt.xticks(rotation=15)

        # plt.figure()
        # df_display = df_prec_recall.join(df_auc, on=METHOD, how="inner")
        # sns.lineplot(
        #     data=df_display,
        #     x=RECALL,
        #     y=PRECISION,
        #     hue=METHOD,
        #     errorbar=None,
        # )
        # plt.title(f"Precision-Recall curve on")


if __name__ == "__main__":
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_rows(20)

    print(
        "------------------------ Evaluating the composite network --------------------"
    )
    evaluator = CompEdgesEvaluator(dip=False)
    evaluator.main(re_eval=False)
    print()

    print(
        "------------------------ Evaluating the DIP composite network --------------------"
    )
    evaluator = CompEdgesEvaluator(dip=True)
    evaluator.main(re_eval=False)
    print()

    plt.show()
