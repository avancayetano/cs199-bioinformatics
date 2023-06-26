import time
from typing import Any, Dict, List, Literal, Set, TypedDict, Union

import numpy as np
import polars as pl

from aliases import (
    COMP_PROTEINS,
    DENSITY,
    FEATURES,
    PROTEIN_U,
    PROTEIN_V,
    SCENARIO,
    SUPER_FEATS,
    WEIGHT,
)
from utils import (
    get_clusters_filename,
    get_clusters_list,
    get_complexes_list,
    get_weighted_filename,
)

ScoredCluster = TypedDict(
    "ScoredCluster", {"COMP_PROTEINS": Set[str], "DENSITY": float}
)

INFLATION = "INFLATION"
N_EDGES = "N_EDGES"
METHOD = "METHOD"
XVAL_ITER = "XVAL_ITER"

N_CLUSTERS = "N_CLUSTERS"
PREC = "PREC"
RECALL = "RECALL"
F_SCORE = "F_SCORE"
MATCH_THRESH = "MATCH_THRESH"
DENS_THRESH = "DENS_THRESH"

AUC = "AUC"
AVG_AUC = "AVG_AUC"


class ClusterEvaluator:
    def __init__(self, dip: bool):
        self.sv_methods = ["SWC", "XGW"]
        self.feat_methods = FEATURES + [method["name"] for method in SUPER_FEATS]
        self.methods = self.sv_methods + self.feat_methods + ["unweighted"]
        self.n_iters = 10
        self.dip = dip
        self.inflations = [2, 3, 4, 5]
        self.n_edges: List[Union[Literal["all_edges"], Literal["20k_edges"]]] = [
            "all_edges",
            "20k_edges",
        ]
        self.data: Dict[str, Any] = {}

    def get_thresholds(self, df_clusters: pl.DataFrame) -> List[float]:
        """
        Thresholds are all the possible cutpoints (average between two points)
        """
        thresholds = (
            df_clusters.select(DENSITY)
            .to_series()
            .sort()
            .unique()
            .rolling_mean(2)
            .drop_nulls()
            .to_list()
        )
        return thresholds

    def get_scored_clusters_from_df(self, df_clusters: pl.DataFrame) -> List[Any]:
        dict_clusters: Dict[str, float] = dict(df_clusters.iter_rows())
        clusters: List[Any] = list(
            map(
                lambda c: {COMP_PROTEINS: set(c.split(",")), DENSITY: dict_clusters[c]},
                dict_clusters,
            )
        )
        return clusters

    def cache_data(self):
        for I in self.inflations:
            for e in self.n_edges:
                for method in self.sv_methods:
                    for xval_iter in range(self.n_iters):
                        clusters_path = get_clusters_filename(
                            e, method, True, I, self.dip, xval_iter, True
                        )

                        df_clusters = pl.read_csv(
                            clusters_path,
                            has_header=False,
                            separator="\t",
                            new_columns=[COMP_PROTEINS, DENSITY],
                        )
                        scored_clusters = self.get_scored_clusters_from_df(df_clusters)
                        thresholds = self.get_thresholds(df_clusters)
                        self.data[clusters_path] = {
                            "scored_clusters": scored_clusters,
                            "thresholds": thresholds,
                        }
                        print(f"Done caching {clusters_path}")

                for method in self.feat_methods:
                    clusters_path = get_clusters_filename(
                        e, method, False, I, self.dip, scored=True
                    )

                    df_clusters = pl.read_csv(
                        clusters_path,
                        has_header=False,
                        separator="\t",
                        new_columns=[COMP_PROTEINS, DENSITY],
                    )
                    thresholds = self.get_thresholds(df_clusters)
                    self.data[clusters_path] = {
                        "df_clusters": df_clusters,
                        "thresholds": thresholds,
                    }
                    print(f"Done caching {clusters_path}")

        self.train_complexes: List[List[Set[str]]] = [
            get_complexes_list(xval_iter, "train") for xval_iter in range(self.n_iters)
        ]
        self.test_complexes: List[List[Set[str]]] = [
            get_complexes_list(xval_iter, "test") for xval_iter in range(self.n_iters)
        ]

    def is_match(
        self, subgraph1: Set[str], subgraph2: Set[str], match_thresh: float
    ) -> bool:
        jaccard_idx = len(subgraph1.intersection(subgraph2)) / len(
            subgraph1.union(subgraph2)
        )
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

    def get_complex_prediction_metrics(
        self,
        inflation: int,
        n_edges: Union[Literal["20k_edges"], Literal["all_edges"]],
        method: str,
        xval_iter: int,
        dens_thresh: float,
        match_thresh: float,
        scored_clusters: List[ScoredCluster],
    ) -> Dict[str, Union[str, float, int]]:
        train_complexes = self.train_complexes[xval_iter]
        test_complexes = self.test_complexes[xval_iter]

        # Get only the reliable clusters
        clusters = list(
            map(
                lambda scored_cluster: scored_cluster[COMP_PROTEINS],
                filter(
                    lambda scored_cluster: scored_cluster[DENSITY] >= dens_thresh,
                    scored_clusters,
                ),
            )
        )

        # Computing the precision
        prec_numerator = len(
            list(
                filter(
                    lambda cluster: (
                        not self.there_is_match(cluster, train_complexes, match_thresh)
                    )
                    and (self.there_is_match(cluster, test_complexes, match_thresh)),
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

        prec = (
            prec_numerator / prec_denominator
            if prec_denominator + prec_numerator > 0
            else 0
        )

        # Computing the recall
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
            print("WARNING! - Zero denominator (precision and recall)")
            f_score = 0

        n_clusters = len(clusters)

        return {
            INFLATION: inflation,
            N_EDGES: n_edges,
            METHOD: method.upper(),
            XVAL_ITER: xval_iter,
            DENS_THRESH: dens_thresh,
            MATCH_THRESH: match_thresh,
            N_CLUSTERS: n_clusters,
            PREC: prec,
            RECALL: recall,
            F_SCORE: f_score,
        }

    def evaluate_clusters(
        self,
        inflation: int,
        edges: Union[Literal["20k_edges"], Literal["all_edges"]],
        xval_iter: int,
        method: str,
    ) -> List[Dict[str, Union[str, float, int]]]:
        clusters_path = get_clusters_filename(
            edges,
            method,
            method in self.sv_methods,
            inflation,
            self.dip,
            xval_iter,
            True,
        )
        d_thresholds: List[float] = self.data[clusters_path]["thresholds"]
        evals: List[Dict[str, Union[str, float, int]]] = []
        for d_thresh in d_thresholds:
            scored_clusters: List[ScoredCluster] = self.data[clusters_path][
                "scored_clusters"
            ]
            metrics_050 = self.get_complex_prediction_metrics(
                inflation=inflation,
                n_edges=edges,
                method=method,
                xval_iter=xval_iter,
                dens_thresh=d_thresh,
                match_thresh=0.5,
                scored_clusters=scored_clusters,
            )
            metrics_075 = self.get_complex_prediction_metrics(
                inflation=inflation,
                n_edges=edges,
                method=method,
                xval_iter=xval_iter,
                dens_thresh=d_thresh,
                match_thresh=0.75,
                scored_clusters=scored_clusters,
            )
            print(f"Done evaluating {d_thresh} {clusters_path}")

            evals.extend([metrics_050, metrics_075])

        print(f"---- Done evaluating {clusters_path} -----------------")

        return evals

    def evaluate_complex_prediction(self):
        """
        Terminologies:
        - cluster: predicted cluster
        - complex: reference (aka real) complex
        - subgraph: either cluster or complex
        """

        print("Evaluating protein complex prediction")

        evals: List[Dict[str, str | float | int]] = []

        for inflation in self.inflations:
            for edges in self.n_edges:
                for xval_iter in range(self.n_iters):
                    for method in self.methods:
                        evals_edges = self.evaluate_clusters(
                            inflation, edges, xval_iter, method
                        )
                        evals.extend(evals_edges)

        df_evals = pl.DataFrame(evals)
        df_evals.write_csv("../data/evals/cluster_evals.csv", has_header=True)

    def main(self):
        self.cache_data()
        self.evaluate_complex_prediction()


if __name__ == "__main__":
    start = time.time()
    evaluator = ClusterEvaluator(dip=False)
    evaluator.main()

    print(f"EXECUTION TIME: {time.time() - start}")
