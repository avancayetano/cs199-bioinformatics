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

    # -----------------------------------------------------------------------------------
    # Below are the performance evaluation methods for complex prediction

    def evaluate_complex_prediction(
        self,
        edges: List[str],
        feat_methods: List[str],
        sv_methods: List[str],
        inflation: float,
        n_iters: int,
    ):
        """
        Terminologies:
        - cluster: predicted cluster
        - complex: reference (aka real) complex
        - subgraph: either cluster or complex
        """

        print("Evaluating protein complex prediction")
        feat_methods = [
            f for f in FEATURES + list(map(lambda sf: sf["name"], SUPER_FEATS))
        ]
        sv_methods = ["CNB", "RF", "MLP", "SWC"]

        unw_clusters = get_clusters_list(
            f"../data/clusters/out.unweighted.csv.I{inflation}0"
        )

        train_complexes = {}  # ID: xval_iter
        test_complexes = {}  # ID: xval_iter
        clusters = {}  # ID: edges, cross-val or feature, method, xval_iter (?)
        weighted = {}  # ID: edges, cross-val or feature, method, xval_iter (?)

        df_prec_recall_curve = self.prec_recall_curve()

    def prec_recall_curve(self, n_iters: int, unw_clusters: List[Set[str]]):
        evals_all_edges: List[Dict[str, Union[str, float]]] = []
        evals_20k_edges: List[Dict[str, Union[str, float]]] = []

        n_dens = 10
        dens_thresholds = [i / n_dens for i in range(n_dens)]

        for dens_thresh in dens_thresholds:
            for xval_iter in range(n_iters):
                print()
                print(f"dens_thresh={dens_thresh} | iteration: {xval_iter}")
                train_complexes = get_complexes_list(xval_iter, "train")
                test_complexes = get_complexes_list(xval_iter, "test")
                print(f"Number of test complexes: {len(test_complexes)}")

                unw_metrics_050 = self.get_complex_prediction_metrics(
                    method="UNW", unw_clusters, test_complexes, 0.50
                )
                unw_metrics_075 = self.get_complex_prediction_metrics(
                    "UNW", unw_clusters, test_complexes, 0.75
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
            .groupby([METHOD, MATCH_THRESH], maintain_order=True)
            .mean()
        )
        print("On match threshold >= 0.5")
        print(df_evals_all_edges.filter(pl.col(MATCH_THRESH) == 0.5))

        print("On match threshold >= 0.75")
        print(df_evals_all_edges.filter(pl.col(MATCH_THRESH) == 0.75))
        print()

        print("Evaluations for clusters using 20k edges")
        df_evals_20k_edges = (
            pl.DataFrame(evals_20k_edges)
            .groupby([METHOD, MATCH_THRESH], maintain_order=True)
            .mean()
        )
        print("On match threshold >= 0.5")
        print(df_evals_20k_edges.filter(pl.col(MATCH_THRESH) == 0.5))

        print("On match threshold >= 0.75")
        print(df_evals_20k_edges.filter(pl.col(MATCH_THRESH) == 0.75))
        print()

    def evaluate_clusters(
        self,
        dens_thresh: float,
        feat_methods: List[str],
        sv_methods: List[str],
        train_complexes: List[Set[str]],
        test_complexes: List[Set[str]],
        xval_iter: int,
        edges: str,
        inflation: int,
    ):
        evals: List[Dict[str, Union[str, float]]] = []
        suffix = "_20k" if edges == "20k_edges" else ""
        for feat in feat_methods:
            feat_clusters = get_clusters_list(
                f"../data/clusters/{edges}/features/out.{feat.lower()}{suffix}.csv.I{inflation}0"
            )

            df_w = pl.read_csv(
                f"../data/weighted/{edges}/features/{feat.lower()}{suffix}.csv",
                new_columns=[PROTEIN_U, PROTEIN_V, WEIGHT],
            )
            feat_metrics_050 = self.get_complex_prediction_metrics(
                method=feat,
                df_w=df_w,
                clusters=feat_clusters,
                train_complexes=train_complexes,
                test_complexes=test_complexes,
                match_thresh=0.50,
                dens_thresh=dens_thresh,
            )
            feat_metrics_075 = self.get_complex_prediction_metrics(
                method=feat,
                df_w=df_w,
                clusters=feat_clusters,
                train_complexes=train_complexes,
                test_complexes=test_complexes,
                match_thresh=0.75,
                dens_thresh=dens_thresh,
            )
            evals.extend([feat_metrics_050, feat_metrics_075])
            print(
                f"Done evaluating {feat} clusters on {edges}. dens_thresh={dens_thresh}"
            )

        for sv in sv_methods:
            sv_clusters = get_clusters_list(
                f"../data/clusters/{edges}/cross_val/out.{sv.lower()}{suffix}_iter{xval_iter}.csv.I{inflation}0"
            )
            df_w = pl.read_csv(
                f"../data/weighted/{edges}/cross_val/{sv.lower()}{suffix}_iter{xval_iter}.csv",
                new_columns=[PROTEIN_U, PROTEIN_V, WEIGHT],
            )

            sv_metrics_050 = self.get_complex_prediction_metrics(
                method=sv,
                df_w=df_w,
                clusters=sv_clusters,
                train_complexes=train_complexes,
                test_complexes=test_complexes,
                match_thresh=0.50,
                dens_thresh=dens_thresh,
            )
            sv_metrics_075 = self.get_complex_prediction_metrics(
                method=sv,
                df_w=df_w,
                clusters=sv_clusters,
                train_complexes=train_complexes,
                test_complexes=test_complexes,
                match_thresh=0.75,
                dens_thresh=dens_thresh,
            )
            evals.extend([sv_metrics_050, sv_metrics_075])
            print(
                f"Done evaluating {sv} clusters on {edges}. dens_thresh={dens_thresh}"
            )

        return evals

    def is_match(
        self, cluster: Set[str], complex: Set[str], match_thresh: float
    ) -> bool:
        jaccard_idx = len(cluster.intersection(complex)) / len(cluster.union(complex))
        if jaccard_idx >= match_thresh:
            return True
        return False

    def there_is_match(
        self, subgraph: Set[str], subgraphs_set: List[Set[str]], match_thresh: float
    ) -> bool:
        for s in subgraphs_set:
            if self.is_match(subgraph, s, match_thresh):
                return True
        return False

    def cluster_density(self, df_w: pl.DataFrame, cluster: Set[str]) -> float:
        sorted_cluster = list(sorted(cluster))
        df_pairs = pl.DataFrame(
            [
                [u, v]
                for i, u in enumerate(sorted_cluster)
                for v in sorted_cluster[i + 1 :]
            ],
            schema=[PROTEIN_U, PROTEIN_V],
        )
        assert_prots_sorted(df_pairs)
        weight = (
            df_w.join(df_pairs, on=[PROTEIN_U, PROTEIN_V], how="inner")
            .select(WEIGHT)
            .to_series()
            .sum()
        )
        n = len(cluster)
        density = 2 * weight / (n * (n - 1))
        return density

    def get_complex_prediction_metrics(
        self,
        method: str,
        df_w: pl.DataFrame,
        clusters: List[Set[str]],
        train_complexes: List[Set[str]],
        test_complexes: List[Set[str]],
        match_thresh: float,
        dens_thresh: float,
    ):
        # Get only the reliable clusters
        clusters = list(
            filter(
                lambda cluster: len(cluster) >= 2
                and self.cluster_density(df_w, cluster) >= dens_thresh,
                clusters,
            )
        )

        # Computing precision
        prec_numerator = len(
            list(
                filter(
                    lambda cluster: self.there_is_match(
                        cluster, test_complexes, match_thresh
                    ),
                    clusters,
                )
            )
        )

        prec_denominator = len(
            list(
                filter(
                    lambda cluster: (
                        not self.there_is_match(cluster, train_complexes, match_thresh)
                    )
                    or (self.there_is_match(cluster, test_complexes, match_thresh)),
                    clusters,
                )
            )
        )
        prec = prec_numerator / prec_denominator

        # Computing recall
        recall_numerator = len(
            list(
                filter(
                    lambda complex: self.there_is_match(
                        complex, clusters, match_thresh
                    ),
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

        n_clusters = len(clusters)

        return {
            METHOD: method,
            DENS_THRESH: dens_thresh,
            MATCH_THRESH: match_thresh,
            N_CLUSTERS: n_clusters,
            PREC_NUM: prec_numerator,
            RECALL_NUM: recall_numerator,
            PREC: prec,
            RECALL: recall,
            F_SCORE: f_score,
        }


if __name__ == "__main__":
    pl.Config.set_tbl_rows(30)
    evaluator = Evaluator()
    # inflations = [2, 3, 4, 5]
    inflations = [4]
    for I in inflations:
        try:
            complex_pred_evals = evaluator.evaluate_complex_prediction(10, I)
        except:
            continue
