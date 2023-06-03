# pyright: basic

from typing import List, Set, Tuple

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
)
from utils import get_clusters_list, get_complexes_list


class Evaluator:
    """
    Evaluates:
    - [X] - Co-complex pair classification
    - [ ] - Cluster classification
    """

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
        scenarios = {
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
        for s in scenarios:
            eval_info = self.get_co_comp_classifier_eval(scenarios[s], label)
            eval_summary[MODEL].append(name)
            eval_summary[SCENARIO].append(s)
            eval_summary[MAE].append(eval_info[MAE])
            eval_summary[MSE].append(eval_info[MSE])

        df_eval_summary = pl.DataFrame(eval_summary)
        return df_eval_summary

    def get_co_comp_classifier_eval(self, df_pred_label: pl.DataFrame, label: str):
        y_pred = df_pred_label.select(WEIGHT).to_series().to_numpy()
        y_true = df_pred_label.select(label).to_series().to_numpy()
        return {
            MAE: mean_absolute_error(y_true, y_pred),
            MSE: mean_squared_error(y_true, y_pred),
        }

    def evaluate_complex_prediction(self, n_iters: int):
        """
        Terminologies:
        - cluster: predicted cluster
        - complex: reference (aka real) complex
        - subgraph: either cluster or complex
        """
        # ref_complexes = None
        # print("Evaluating the null model (unweighted PPIN)")
        # unw_clusters = get_clusters_list("../data/clusters/out.unweighted.csv.I40")
        print("Evaluating clusters from all edges")

        model_clusters = ["cnb", "rf", "mlp", "swc"]
        feat_clusters = [
            f.lower() for f in FEATURES + list(map(lambda sf: sf["name"], SUPER_FEATS))
        ]
        for xval_iter in range(n_iters):
            pass

        # TODO
        # print("Evaluating clusters from top 20k edges")

    def is_match(self, cluster: Set[str], complex: Set[str], thresh: float):
        jaccard_idx = len(cluster.intersection(complex)) / len(cluster.union(complex))
        if jaccard_idx >= thresh:
            return True
        return False

    def there_is_match(
        self, subgraph: Set[str], subgraphs_set: List[Set[str]], thresh: float
    ):
        for s in subgraphs_set:
            if self.is_match(subgraph, s, thresh):
                return True
        return False

    def get_complex_prediction_metrics(
        self,
        clusters: List[Set[str]],
        test_complexes: List[Set[str]],
        thresh: float,
    ):
        # Computing precision
        prec_numerator = len(
            list(
                filter(
                    lambda cluster: self.there_is_match(
                        cluster, test_complexes, thresh
                    ),
                    clusters,
                )
            )
        )
        prec_denominator = len(clusters)
        prec = prec_numerator / prec_denominator

        # Computing recall
        recall_numerator = len(
            list(
                filter(
                    lambda complex: self.there_is_match(complex, clusters, thresh),
                    test_complexes,
                )
            )
        )
        recall_denominator = len(test_complexes)
        recall = recall_numerator / recall_denominator

        f_score = (2 * prec * recall) / (prec + recall)
        n_matches = recall_numerator

        return (n_matches, prec, recall, f_score)
