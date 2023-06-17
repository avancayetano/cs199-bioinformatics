# pyright: basic

import time
from typing import Dict, List, Literal, NotRequired, Set, Tuple, TypedDict, Union

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from sklearn.metrics import PrecisionRecallDisplay, auc

from aliases import FEATURES, PROTEIN_U, PROTEIN_V, SCENARIO, SUPER_FEATS, WEIGHT
from assertions import assert_prots_sorted
from utils import (
    get_cluster_filename,
    get_clusters_list,
    get_complexes_list,
    get_weighted_filename,
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
SCENARIO = SCENARIO

Subgraphs = List[Set[str]]
ScoredCluster = TypedDict(
    "ScoredCluster", {"COMP_PROTEINS": Set[str], "DENSITY": float}
)
ScoredClusters = List[ScoredCluster]

FeatClusters = TypedDict(
    "FeatClusters",
    {
        "20k_edges": Dict[str, ScoredClusters],
        "all_edges": Dict[str, ScoredClusters],
    },
)

SvClusters = TypedDict(
    "SvClusters",
    {
        "20k_edges": Dict[str, Dict[int, ScoredClusters]],
        "all_edges": Dict[str, Dict[int, ScoredClusters]],
    },
)


# the keys are MCL inflation parameter settings
AllFeatClusters = Dict[int, FeatClusters]
AllSvClusters = Dict[int, SvClusters]
AllUnwClusters = Dict[int, ScoredClusters]


FeatWeighted = Dict[str, pl.DataFrame]
SvWeighted = Dict[str, Dict[int, pl.DataFrame]]

Edges = List[Union[Literal["20k_edges"], Literal["all_edges"]]]


class ClustersEvaluator:
    def __init__(
        self,
        inflations: List[int],
        edges: Edges,
        feat_methods: List[str],
        sv_methods: List[str],
        n_dens: int,
        n_iters: int,
    ):
        self.inflations = inflations
        self.n_dens = n_dens
        self.n_iters = n_iters

        self.edges = edges
        self.feat_methods = feat_methods
        self.sv_methods = sv_methods

        self.methods = ["unweighted"] + self.feat_methods + self.sv_methods

        # to track the progress
        self.idx = 0
        self.total = (
            len(self.inflations)
            * (n_dens + 1)
            * n_iters
            * len(self.edges)
            * len(self.methods)
            * 2
        )

    def cluster_density(self, df_w: pl.DataFrame, cluster: Set[str]) -> float:
        if len(cluster) <= 1:
            return 0
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

    def cache_eval_data(self):
        print("Caching necessary eval data (might take a while)...")
        train_complexes: List[Subgraphs] = [
            get_complexes_list(xval_iter, "train") for xval_iter in range(self.n_iters)
        ]
        test_complexes: List[Subgraphs] = [
            get_complexes_list(xval_iter, "test") for xval_iter in range(self.n_iters)
        ]

        feat_clusters: AllFeatClusters = {}
        feat_weighted: FeatWeighted = {}

        sv_clusters: AllSvClusters = {}
        sv_weighted: SvWeighted = {}

        unw_clusters: AllUnwClusters = {}
        df_unweighted = pl.read_csv(
            get_weighted_filename("unweighted", False),
            new_columns=[PROTEIN_U, PROTEIN_V, WEIGHT],
            separator="\t",
            has_header=False,
        )

        for m in self.sv_methods:
            sv_weighted[m] = {}
            for j in range(self.n_iters):
                sv_weighted[m][j] = pl.read_csv(
                    get_weighted_filename(m, True, f"_iter{j}"),
                    new_columns=[PROTEIN_U, PROTEIN_V, WEIGHT],
                    separator="\t",
                    has_header=False,
                )

        for m in self.feat_methods:
            feat_weighted[m] = pl.read_csv(
                get_weighted_filename(m, False),
                new_columns=[PROTEIN_U, PROTEIN_V, WEIGHT],
                separator="\t",
                has_header=False,
            )

        cache_idx = 0
        max_cache_idx = len(self.inflations) * len(self.edges) * (
            len(self.sv_methods) * self.n_iters + len(self.feat_methods)
        ) + len(self.inflations)

        for I in self.inflations:
            feat_clusters[I] = {}  # type: ignore
            sv_clusters[I] = {}  # type: ignore
            unw_clusters[I] = {}  # type: ignore

            for e in self.edges:
                feat_clusters[I][e] = {}
                sv_clusters[I][e] = {}

                for m in self.sv_methods:
                    sv_clusters[I][e][m] = {}
                    for j in range(self.n_iters):
                        clusters = get_clusters_list(
                            get_cluster_filename(e, m, True, I, f"_iter{j}")
                        )
                        sv_clusters[I][e][m][j] = list(
                            map(
                                lambda cluster: {
                                    "COMP_PROTEINS": cluster,
                                    "DENSITY": self.cluster_density(
                                        sv_weighted[m][j], cluster
                                    ),
                                },
                                clusters,
                            )
                        )
                        cache_idx += 1
                        print(f"[{cache_idx}/{max_cache_idx}] Done caching")

                for m in self.feat_methods:
                    clusters = get_clusters_list(get_cluster_filename(e, m, False, I))
                    feat_clusters[I][e][m] = list(
                        map(
                            lambda cluster: {
                                "COMP_PROTEINS": cluster,
                                "DENSITY": self.cluster_density(
                                    feat_weighted[m], cluster
                                ),
                            },
                            clusters,
                        )
                    )
                    cache_idx += 1
                    print(f"[{cache_idx}/{max_cache_idx}] Done caching")

            clusters = get_clusters_list(
                get_cluster_filename("", "unweighted", False, I)
            )
            unw_clusters[I] = list(
                map(
                    lambda cluster: {
                        "COMP_PROTEINS": cluster,
                        "DENSITY": self.cluster_density(df_unweighted, cluster),
                    },
                    clusters,
                )
            )
            cache_idx += 1
            print(f"[{cache_idx}/{max_cache_idx}] Done caching")

        self.train_complexes = train_complexes
        self.test_complexes = test_complexes

        self.feat_clusters = feat_clusters
        self.feat_weighted = feat_weighted

        self.sv_clusters = sv_clusters
        self.sv_weighted = sv_weighted

        self.unw_clusters = unw_clusters
        self.df_unweighted = df_unweighted

        print("Done caching eval data")

    def evaluate_complex_prediction(self):
        """
        Terminologies:
        - cluster: predicted cluster
        - complex: reference (aka real) complex
        - subgraph: either cluster or complex
        """

        print("Evaluating protein complex prediction")
        dens_thresholds = [0.0] + [(i + 1) / self.n_dens for i in range(self.n_dens)]

        evals: List[Dict[str, str | float | int]] = []

        for inflation in self.inflations:
            for dens_thresh in dens_thresholds:
                for xval_iter in range(self.n_iters):
                    for edges in self.edges:
                        evals_edges = self.evaluate_clusters(
                            inflation,
                            edges,
                            dens_thresh,
                            xval_iter,
                        )
                        evals.extend(evals_edges)

        df_evals = pl.DataFrame(evals)
        df_evals.write_csv("../data/evals/cluster_evals.csv", has_header=True)

    def evaluate_clusters(
        self,
        inflation: int,
        n_edges: Union[Literal["20k_edges"], Literal["all_edges"]],
        dens_thresh: float,
        xval_iter: int,
    ) -> List[Dict[str, Union[str, float, int]]]:
        evals: List[Dict[str, Union[str, float, int]]] = []

        for method in self.methods:
            metrics_050 = self.get_complex_prediction_metrics(
                inflation=inflation,
                n_edges=n_edges,
                method=method,
                xval_iter=xval_iter,
                dens_thresh=dens_thresh,
                match_thresh=0.5,
            )
            metrics_075 = self.get_complex_prediction_metrics(
                inflation=inflation,
                n_edges=n_edges,
                method=method,
                xval_iter=xval_iter,
                dens_thresh=dens_thresh,
                match_thresh=0.75,
            )
            evals.extend([metrics_050, metrics_075])
            self.idx += 2
            print(
                f"[{self.idx}/{self.total}] Done evaluating {method} clusters on {n_edges}. dens_thresh={dens_thresh}. xval_iter={xval_iter}"
            )

        return evals

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
    ) -> Dict[str, Union[str, float, int]]:
        if method in self.sv_methods:
            scored_clusters = self.sv_clusters[inflation][n_edges][method][xval_iter]
        elif method in self.feat_methods:
            scored_clusters = self.feat_clusters[inflation][n_edges][method]
        else:  # for method == unweighted
            scored_clusters = self.unw_clusters[inflation]

        train_complexes = self.train_complexes[xval_iter]
        test_complexes = self.test_complexes[xval_iter]

        # Get only the reliable clusters
        clusters = list(
            map(
                lambda scored_cluster: scored_cluster["COMP_PROTEINS"],
                filter(
                    lambda scored_cluster: len(scored_cluster["COMP_PROTEINS"]) >= 3
                    and scored_cluster["DENSITY"] >= dens_thresh,
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

    def prec_recall_curves(self):
        """
        Four precision-recall curves (average of all inflation settings)
        - all_edges, match_thresh = 0.5
        - all_edges, match_thresh = 0.75
        - 20k_edges, match_thresh = 0.5
        - 20k_edges, match_thresh = 0.75
        """
        df_cluster_evals = pl.read_csv("../data/evals/cluster_evals.csv")

        # df_all_050 = self.get_prec_recall_curve(df_cluster_evals, "all_edges", 0.5)
        # df_all_075 = self.get_prec_recall_curve(df_cluster_evals, "all_edges", 0.75)
        df_20k_050 = self.get_prec_recall_curve(df_cluster_evals, "20k_edges", 0.5)
        df_20k_075 = self.get_prec_recall_curve(df_cluster_evals, "20k_edges", 0.75)

        # df_all_050_auc = self.get_prec_recall_auc(df_all_050, "all_050")
        # df_all_075_auc = self.get_prec_recall_auc(df_all_075, "all_075")
        df_20k_050_auc = self.get_prec_recall_auc(df_20k_050, "20k_050")
        df_20k_075_auc = self.get_prec_recall_auc(df_20k_075, "20k_075")

        # Print AUC summary of the four scenarios
        df_auc_summary = (
            pl.concat(
                # [df_all_050_auc, df_all_075_auc, df_20k_050_auc, df_20k_075_auc],
                [df_20k_050_auc, df_20k_075_auc],
                how="vertical",
            )
            .pivot(
                values=AUC, index=METHOD, columns=SCENARIO, aggregate_function="first"
            )
            .with_columns((pl.sum(pl.all().exclude(METHOD)) / 4).alias(AVG_AUC))
        )
        print(df_auc_summary.sort(AVG_AUC, descending=True))

        n_methods = 7
        df_top_methods = (
            df_auc_summary.sort(AVG_AUC, descending=True).select(METHOD).head(n_methods)
        )
        # Plot the four curves
        # self.plot_prec_recall_curve(
        #     df_all_050, df_top_methods, f"all edges, match_thresh=0.5"
        # )
        # self.plot_prec_recall_curve(
        #     df_all_075, df_top_methods, f"all edges, match_thresh=0.75"
        # )
        self.plot_prec_recall_curve(
            df_20k_050, df_top_methods, f"20k edges, match_thresh=0.5"
        )
        self.plot_prec_recall_curve(
            df_20k_075, df_top_methods, f"20k edges, match_thresh=0.75"
        )
        plt.show()

    def plot_prec_recall_curve(
        self, df: pl.DataFrame, df_top_methods: pl.DataFrame, scenario: str
    ):
        plt.figure()
        df_display = df.join(df_top_methods, on=METHOD, how="inner")
        sns.lineplot(
            data=df_display,
            x=RECALL,
            y=PREC,
            hue=METHOD,
            errorbar=None,
            markers=True,
            marker="o",
        )
        plt.title(f"Precision-Recall curve on {scenario}")

    def get_prec_recall_curve(
        self, df_cluster_evals: pl.DataFrame, n_edges: str, match_thresh: float
    ) -> pl.DataFrame:
        df_prec_recall = (
            df_cluster_evals.lazy()
            .filter(
                (pl.col(N_EDGES) == n_edges) & (pl.col(MATCH_THRESH) == match_thresh)
            )
            .groupby([INFLATION, METHOD, DENS_THRESH])
            .mean()
            .groupby([METHOD, DENS_THRESH])
            .mean()
            .sort([METHOD, DENS_THRESH])
            .collect()
        )

        return df_prec_recall

    def get_prec_recall_auc(
        self, df_prec_recall: pl.DataFrame, scenario: str
    ) -> pl.DataFrame:
        df_auc = (
            df_prec_recall.lazy()
            .groupby(METHOD, maintain_order=True)
            .agg(
                pl.struct([PREC, RECALL])
                .apply(
                    lambda prec_recall: auc(
                        prec_recall.struct.field(RECALL),
                        prec_recall.struct.field(PREC),
                    )
                )
                .alias(AUC)
            )
            .with_columns(pl.lit(scenario).alias(SCENARIO))
            .collect()
        )

        return df_auc

    def main(self, re_eval: bool = True):
        if re_eval:
            self.cache_eval_data()
            self.evaluate_complex_prediction()

        self.prec_recall_curves()


if __name__ == "__main__":
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_rows(21)
    sns.set_palette("deep")
    start = time.time()
    inflations = [4]
    edges: Edges = ["20k_edges"]
    feat_methods = [
        f.lower() for f in FEATURES + list(map(lambda sf: sf["name"], SUPER_FEATS))
    ]
    sv_methods = ["cnb", "rf", "mlp", "swc", "rf_swc", "rf_mlp"]
    n_dens = 5
    n_iters = 1

    cluster_eval = ClustersEvaluator(
        inflations=inflations,
        edges=edges,
        feat_methods=feat_methods,
        sv_methods=sv_methods,
        n_dens=n_dens,
        n_iters=n_iters,
    )

    cluster_eval.main(re_eval=False)
    print(f"Execution Time: {time.time() - start}")
