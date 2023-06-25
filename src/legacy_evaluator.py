# pyright: basic

from typing import Dict, List, Set, Tuple, Union

import polars as pl
from sklearn.metrics import (
    PrecisionRecallDisplay,
    auc,
    mean_absolute_error,
    mean_squared_error,
)

from aliases import (
    FEATURES,
    IS_CO_COMP,
    MAE,
    MODEL,
    MSE,
    PROTEIN_U,
    PROTEIN_V,
    SCENARIO,
    SUPER_FEATS,
    WEIGHT,
    XVAL_ITER,
)
from assertions import assert_prots_sorted
from utils import get_clusters_list, get_complexes_list

METHOD = "METHOD"
N_CLUSTERS = "N_CLUSTERS"
PREC_NUM = "PREC_NUM"
RECALL_NUM = "RECALL_NUM"
PREC = "PREC"
RECALL = "RECALL"
F_SCORE = "F_SCORE"
MATCH_THRESH = "MATCH_THRESH"
DENS_THRESH = "DENS_THRESH"
THRESH = "THRESH"
N_MATCHES = "N_MATCHES"


class Evaluator:
    """
    Evaluates:
    - [X] - Co-complex pair classification
    - [ ] - Cluster classification
    """

    # -----------------------------------------------------------------------------------
    # Below are the performance evaluation methods for co-complex classifier

    def evaluate_co_comp_classifier(
        self,
        name: str,
        df_w_composite: pl.DataFrame,
        df_test_labeled: pl.DataFrame,
        df_all_labeled: pl.DataFrame,
        label: str,
    ) -> pl.DataFrame:
        """
        Evaluate the results in terms of:
        1. Predicting co-comp pairs in test co-comp pairs
        2. Predicting co-comp pairs in (train + test) co-comp pairs
        3. Predicting co-comp and non-co-comp pairs in (test) (co-comp + non-co-comp) pairs
        4. Predicting co-comp and non-co-comp pairs in (train + test) (co-comp + non-co-comp) pairs

        Scenario codes:
        1. TEST_CO
        2. ALL_CO
        3. TEST_CO-NONCO
        4. ALL_CO-NONCO

        Args:
            df_w_composite (pl.DataFrame): _description_
            df_train_labeled (pl.DataFrame): _description_
            df_test_labeled (pl.DataFrame): _description_
            xval_iter (int): _description_
        """

        df_w_test_labeled = df_w_composite.join(
            df_test_labeled, on=[PROTEIN_U, PROTEIN_V], how="inner"
        )
        df_w_all_labeled = df_w_composite.join(
            df_all_labeled, on=[PROTEIN_U, PROTEIN_V], how="inner"
        )
        scenarios_df = {
            "TEST_CO": df_w_test_labeled.filter(pl.col(IS_CO_COMP) == 1),
            "ALL_CO": df_w_all_labeled.filter(pl.col(IS_CO_COMP) == 1),
            "TEST_CO-NONCO": df_w_test_labeled,
            "ALL_CO-NONCO": df_w_all_labeled,
        }
        eval_summary = {
            MODEL: [],
            SCENARIO: [],
            MAE: [],
            MSE: [],
        }
        for s in scenarios_df:
            eval_info = self.get_co_comp_classifier_metrics(scenarios_df[s], label)
            eval_summary[MODEL].append(name)
            eval_summary[SCENARIO].append(s)
            eval_summary[MAE].append(eval_info[MAE])
            eval_summary[MSE].append(eval_info[MSE])

        df_eval_summary = pl.DataFrame(eval_summary)
        return df_eval_summary

    def get_co_comp_classifier_metrics(self, df_pred_label: pl.DataFrame, label: str):
        y_pred = df_pred_label.select(WEIGHT).to_series().to_numpy()
        y_true = df_pred_label.select(label).to_series().to_numpy()
        return {
            MAE: mean_absolute_error(y_true, y_pred),
            MSE: mean_squared_error(y_true, y_pred),
        }
