# pyright: basic

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from sklearn.metrics import auc, brier_score_loss, log_loss, precision_recall_curve

from aliases import (
    BRIER_SCORE,
    FEATURES,
    IS_CO_COMP,
    LOG_LOSS,
    METHOD,
    METRIC,
    PR_AUC,
    PROTEIN_U,
    PROTEIN_V,
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
        """
        Evaluates co-complex edge classification of each weighting method.

        Args:
            dip (bool): Whether to evaluate DIP composite network or not.
        """

        self.sv_methods = ["SWC", "XGW"]
        self.feat_methods = FEATURES + [method["name"] for method in SUPER_FEATS]
        self.methods = self.sv_methods + self.feat_methods + ["UNWEIGHTED"]
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
        """
        Main method.

        Args:
            re_eval (bool, optional): Whether to re-evaluate or not. Defaults to True.
        """
        prefix = "dip_" if self.dip else ""

        if re_eval:
            evals = []
            print(f"Evaluating on these ({len(self.methods)}) methods: {self.methods}")
            print()
            for xval_iter in range(self.n_iters):
                print(f"Evaluating cross-val iteration: {xval_iter}")
                df_train_pairs, _ = get_cyc_train_test_comp_pairs(xval_iter, self.dip)
                df_composite_test = self.df_labeled.join(
                    df_train_pairs, on=[PROTEIN_U, PROTEIN_V], how="anti"
                )

                for method in self.methods:
                    path = get_weighted_filename(
                        method,
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
                        df_w, on=[PROTEIN_U, PROTEIN_V], how="left"
                    ).fill_null(0.0)

                    y_true = df_pred.select(IS_CO_COMP).to_numpy().ravel()
                    y_pred = df_pred.select(WEIGHT).to_numpy().ravel()

                    brier_score = brier_score_loss(y_true, y_pred)
                    log_loss_metric = log_loss(y_true, y_pred)

                    precision, recall, _ = precision_recall_curve(y_true, y_pred)
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

            df_evals = pl.DataFrame(evals)
            df_evals.write_csv(f"../data/evals/{prefix}comp_evals.csv", has_header=True)

        # plots
        network = "DIP COMPOSITE NETWORK" if self.dip else "ORIGINAL COMPOSITE NETWORK"
        n_methods = 12  # Plot only the top 12 methods
        df_evals = pl.read_csv(f"../data/evals/{prefix}comp_evals.csv", has_header=True)

        df_evals_avg = (
            df_evals.groupby(METHOD)
            .mean()
            .select(pl.exclude(XVAL_ITER))
            .sort([LOG_LOSS, BRIER_SCORE, PR_AUC], descending=[False, False, True])
        )

        print("Average of all evaluations on all the cross-val iterations")
        print(df_evals_avg)

        df_loss_top = (
            df_evals_avg.sort([LOG_LOSS, BRIER_SCORE], descending=[False, False])
            .head(n_methods)
            .select([METHOD, LOG_LOSS, BRIER_SCORE])
            .melt(id_vars=METHOD, variable_name=METRIC, value_name=VALUE)
        )

        # Plot in terms of log loss and Brier score loss
        plt.figure()
        ax = sns.barplot(data=df_loss_top.to_pandas(), x=METHOD, y=VALUE, hue=METRIC)
        ax.set_title(
            f"{network}\nClassification of co-complex edges\nTop {n_methods} weighting methods in terms of log loss and Brier score loss."
        )
        plt.xticks(rotation=15)

        df_auc_top = (
            df_evals_avg.sort(PR_AUC, descending=True)
            .head(n_methods)
            .select([METHOD, PR_AUC])
            .melt(id_vars=METHOD, variable_name=METRIC, value_name=VALUE)
        )

        # Plot in terms of PR AUC
        plt.figure()
        ax = sns.barplot(data=df_auc_top.to_pandas(), x=METHOD, y=VALUE, hue=METRIC)
        ax.set_title(
            f"{network}\nClassification of co-complex edges\nTop {n_methods} weighting methods in terms of Precision-Recall AUC."
        )
        plt.xticks(rotation=15)


if __name__ == "__main__":
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_rows(21)

    print(
        "------------------------ Evaluating the composite network --------------------"
    )
    evaluator = CompEdgesEvaluator(dip=False)
    evaluator.main(re_eval=True)
    print()

    print(
        "------------------------ Evaluating the DIP composite network --------------------"
    )
    evaluator = CompEdgesEvaluator(dip=True)
    evaluator.main(re_eval=True)
    print()

    plt.show()
