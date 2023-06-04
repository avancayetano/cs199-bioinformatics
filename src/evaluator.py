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
from utils import get_clusters_list, get_complexes_list

METHOD = "METHOD"
N_MATCHES = "N_MATCHES"
N_CLUSTERS = "N_CLUSTERS"
PREC = "PREC"
RECALL = "RECALL"
F_SCORE = "F_SCORE"
THRESH = "THRESH"


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

    def evaluate_complex_prediction(self, n_iters: int, inflation: int = 40):
        """
        Terminologies:
        - cluster: predicted cluster
        - complex: reference (aka real) complex
        - subgraph: either cluster or complex
        """

        print("Evaluating protein complex prediction")
        feat_methods = [
            f.lower() for f in FEATURES + list(map(lambda sf: sf["name"], SUPER_FEATS))
        ]
        sv_methods = ["cnb", "rf", "mlp", "swc"]

        unw_clusters = get_clusters_list(
            f"../data/clusters/out.unweighted.csv.I{inflation}"
        )

        evals_all_edges: List[Dict[str, Union[str, float]]] = []
        evals_20k_edges: List[Dict[str, Union[str, float]]] = []

        for xval_iter in range(n_iters):
            print()
            print(f"Iteration: {xval_iter}")
            # train_complexes = get_complexes_list(xval_iter, "train")
            test_complexes = get_complexes_list(xval_iter, "test")
            print(f"Number of test complexes: {len(test_complexes)}")

            unw_metrics_050 = self.get_complex_prediction_metrics(
                "unw", unw_clusters, test_complexes, 0.50
            )
            unw_metrics_075 = self.get_complex_prediction_metrics(
                "unw", unw_clusters, test_complexes, 0.75
            )
            evals_all_edges.extend([unw_metrics_050, unw_metrics_075])
            evals_20k_edges.extend([unw_metrics_050, unw_metrics_075])

            iter_evals_all = self.evaluate_clusters(
                feat_methods,
                sv_methods,
                test_complexes,
                xval_iter,
                "all_edges",
                inflation,
            )
            iter_evals_20k = self.evaluate_clusters(
                feat_methods,
                sv_methods,
                test_complexes,
                xval_iter,
                "20k_edges",
                inflation,
            )
            evals_all_edges.extend(iter_evals_all)
            evals_20k_edges.extend(iter_evals_20k)
            print()

        print("Evaluations for clusters using all edges")
        df_evals_all_edges = (
            pl.DataFrame(evals_all_edges)
            .groupby([METHOD, THRESH], maintain_order=True)
            .mean()
        )
        print("On match threshold >= 0.5")
        print(df_evals_all_edges.filter(pl.col(THRESH) == 0.5))

        print("On match threshold >= 0.75")
        print(df_evals_all_edges.filter(pl.col(THRESH) == 0.75))
        print()

        print("Evaluations for clusters using 20k edges")
        df_evals_20k_edges = (
            pl.DataFrame(evals_20k_edges)
            .groupby([METHOD, THRESH], maintain_order=True)
            .mean()
        )
        print("On match threshold >= 0.5")
        print(df_evals_20k_edges.filter(pl.col(THRESH) == 0.5))

        print("On match threshold >= 0.75")
        print(df_evals_20k_edges.filter(pl.col(THRESH) == 0.75))
        print()

    def evaluate_clusters(
        self,
        feat_methods: List[str],
        sv_methods: List[str],
        test_complexes: List[Set[str]],
        xval_iter: int,
        edges: str,
        inflation: int = 40,
    ):
        evals: List[Dict[str, Union[str, float]]] = []
        suffix = "_20k" if edges == "20k_edges" else ""
        for feat in feat_methods:
            feat_clusters = get_clusters_list(
                f"../data/clusters/{edges}/features/out.{feat}{suffix}.csv.I{inflation}"
            )
            feat_metrics_050 = self.get_complex_prediction_metrics(
                feat, feat_clusters, test_complexes, 0.50
            )
            feat_metrics_075 = self.get_complex_prediction_metrics(
                feat, feat_clusters, test_complexes, 0.75
            )
            evals.extend([feat_metrics_050, feat_metrics_075])
            print(f"Done evaluating {feat} clusters on {edges}")

        for sv in sv_methods:
            sv_clusters = get_clusters_list(
                f"../data/clusters/{edges}/cross_val/out.{sv}{suffix}_iter{xval_iter}.csv.I{inflation}"
            )
            sv_metrics_050 = self.get_complex_prediction_metrics(
                sv, sv_clusters, test_complexes, 0.50
            )
            sv_metrics_075 = self.get_complex_prediction_metrics(
                sv, sv_clusters, test_complexes, 0.75
            )
            evals.extend([sv_metrics_050, sv_metrics_075])
            print(f"Done evaluating {sv} clusters on {edges}")

        return evals

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
        method: str,
        clusters: List[Set[str]],
        test_complexes: List[Set[str]],
        thresh: float,
    ):
        # filter out small clusters TODO: filter via density, remove duplicates
        clusters = list(filter(lambda cluster: len(cluster) >= 2, clusters))

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

        if prec + recall > 0:
            f_score = (2 * prec * recall) / (prec + recall)
        else:
            f_score = 0

        n_matches = recall_numerator

        return {
            METHOD: method,
            THRESH: thresh,
            N_MATCHES: n_matches,
            N_CLUSTERS: len(clusters),
            PREC: prec,
            RECALL: recall,
            F_SCORE: f_score,
        }


if __name__ == "__main__":
    pl.Config.set_tbl_rows(30)
    evaluator = Evaluator()
    # evaluator.evaluate_complex_prediction(10, 20)
    # evaluator.evaluate_complex_prediction(10, 30)
    # evaluator.evaluate_complex_prediction(10, 40)
    # evaluator.evaluate_complex_prediction(10, 50)
