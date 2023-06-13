# pyright: basic

from typing import Dict, List, Literal, NotRequired, Set, TypedDict, Union

import polars as pl
from sklearn.metrics import auc

from aliases import FEATURES, PROTEIN_U, PROTEIN_V, SUPER_FEATS, WEIGHT
from assertions import assert_prots_sorted
from utils import get_clusters_list, get_complexes_list

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

Subgraphs = List[Set[str]]

FeatClusters = TypedDict(
    "FeatClusters",
    {
        "20k_edges": Dict[str, Subgraphs],
        "all_edges": Dict[str, Subgraphs],
    },
)

SvClusters = TypedDict(
    "SvClusters",
    {
        "20k_edges": Dict[str, Dict[int, Subgraphs]],
        "all_edges": Dict[str, Dict[int, Subgraphs]],
    },
)


# the keys are MCL inflation parameter settings
AllFeatClusters = Dict[int, FeatClusters]
AllSvClusters = Dict[int, SvClusters]
AllUnwClusters = Dict[int, Subgraphs]


FeatWeighted = TypedDict(
    "FeatWeighted",
    {
        "20k_edges": Dict[str, pl.DataFrame],
        "all_edges": Dict[str, pl.DataFrame],
    },
)

SvWeighted = TypedDict(
    "SvWeighted",
    {
        "20k_edges": Dict[str, Dict[int, pl.DataFrame]],
        "all_edges": Dict[str, Dict[int, pl.DataFrame]],
    },
)


class ClusterEvaluator:
    def __init__(
        self,
        inflations: List[int],
        edges: List[str],
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

        self.cache_eval_data()

        # to track the progress
        self.idx = 0
        self.total = (
            len(self.inflations) * (n_dens + 1) * n_iters * 2 * len(self.methods) * 2
        )

    def get_cluster_filename(
        self,
        n_edges: str,
        method: str,
        inflation: int,
        iter: str = "",
    ):
        if method == "unweighted":
            return f"../data/clusters/out.{method}.csv.I{inflation}0"

        suffix = "_20k" if n_edges == "20k_edges" else ""
        if method in self.sv_methods:
            return f"../data/clusters/{n_edges}/cross_val/out.{method}{suffix}{iter}.csv.I{inflation}0"
        return f"../data/clusters/{n_edges}/features/out.{method}{suffix}.csv.I{inflation}0"

    def get_weighted_filename(
        self,
        n_edges: str,
        method: str,
        iter: str = "",
    ):
        if method == "unweighted":
            return f"../data/weighted/{method}.csv"

        suffix = "_20k" if n_edges == "20k_edges" else ""
        if method in self.sv_methods:
            return f"../data/weighted/{n_edges}/cross_val/{method}{suffix}{iter}.csv"
        return f"../data/weighted/{n_edges}/features/{method}{suffix}.csv"

    def cache_eval_data(self):
        train_complexes: List[Subgraphs] = [
            get_complexes_list(xval_iter, "train") for xval_iter in range(self.n_iters)
        ]
        test_complexes: List[Subgraphs] = [
            get_complexes_list(xval_iter, "test") for xval_iter in range(self.n_iters)
        ]

        feat_clusters: AllFeatClusters = {}
        feat_weighted: FeatWeighted = {}  # type: ignore

        sv_clusters: AllSvClusters = {}
        sv_weighted: SvWeighted = {}  # type: ignore

        unw_clusters: AllUnwClusters = {}
        df_unweighted = pl.read_csv(
            self.get_weighted_filename("", "unweighted"),
            new_columns=[PROTEIN_U, PROTEIN_V, WEIGHT],
            separator="\t",
            has_header=False,
        )

        for I in self.inflations:
            feat_clusters[I] = {}  # type: ignore
            sv_clusters[I] = {}  # type: ignore
            unw_clusters[I] = {}  # type: ignore

            for e in self.edges:
                feat_clusters[I][e] = {}
                feat_weighted[e] = {}

                sv_clusters[I][e] = {}
                sv_weighted[e] = {}

                for m in self.sv_methods:
                    sv_clusters[I][e][m] = {}
                    sv_weighted[e][m] = {}
                    for j in range(self.n_iters):
                        sv_clusters[I][e][m][j] = get_clusters_list(
                            self.get_cluster_filename(e, m, I, f"_iter{j}")
                        )
                        sv_weighted[e][m][j] = pl.read_csv(
                            self.get_weighted_filename(e, m, f"_iter{j}"),
                            new_columns=[PROTEIN_U, PROTEIN_V, WEIGHT],
                            separator="\t",
                            has_header=False,
                        )

                for m in self.feat_methods:
                    feat_clusters[I][e][m] = get_clusters_list(
                        self.get_cluster_filename(e, m, I)
                    )
                    feat_weighted[e][m] = pl.read_csv(
                        self.get_weighted_filename(e, m),
                        new_columns=[PROTEIN_U, PROTEIN_V, WEIGHT],
                        separator="\t",
                        has_header=False,
                    )

            unw_clusters[I] = get_clusters_list(
                self.get_cluster_filename("", "unweighted", I)
            )

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
                    evals_all_edges = self.evaluate_clusters(
                        inflation,
                        "all_edges",
                        dens_thresh,
                        xval_iter,
                    )
                    evals_20k_edges = self.evaluate_clusters(
                        inflation,
                        "20k_edges",
                        dens_thresh,
                        xval_iter,
                    )
                    evals.extend(evals_all_edges)
                    evals.extend(evals_20k_edges)

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
                f"[{self.idx}/{self.total}]Done evaluating {method} clusters on {n_edges}. dens_thresh={dens_thresh}. xval_iter={xval_iter}"
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
        inflation: int,
        n_edges: Union[Literal["20k_edges"], Literal["all_edges"]],
        method: str,
        xval_iter: int,
        dens_thresh: float,
        match_thresh: float,
    ) -> Dict[str, Union[str, float, int]]:
        if method in self.sv_methods:
            clusters = self.sv_clusters[inflation][n_edges][method][xval_iter]
            df_weighted = self.sv_weighted[n_edges][method][xval_iter]
        elif method in self.feat_methods:
            clusters = self.feat_clusters[inflation][n_edges][method]
            df_weighted = self.feat_weighted[n_edges][method]
        else:  # for method == unweighted
            clusters = self.unw_clusters[inflation]
            df_weighted = self.df_unweighted

        train_complexes = self.train_complexes[xval_iter]
        test_complexes = self.test_complexes[xval_iter]

        # Get only the reliable clusters
        clusters = list(
            filter(
                lambda cluster: len(cluster) >= 2
                and self.cluster_density(df_weighted, cluster) >= dens_thresh,
                clusters,
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
        Four precision-recall curves per inflation setting
        - all_edges, match_thresh = 0.5
        - all_edges, match_thresh = 0.75
        - 20k_edges, match_thresh = 0.5
        - 20k_edges, match_thresh = 0.75
        """
        df_cluster_evals = pl.read_csv("../data/evals/cluster_evals.csv")

        curves = {}
        auc_vals = {}

        for inflation in self.inflations:
            curves[inflation] = {}

            df_all_050 = (
                df_cluster_evals.lazy()
                .filter(
                    (pl.col(INFLATION) == inflation)
                    & (pl.col(N_EDGES) == "all_edges")
                    & (pl.col(MATCH_THRESH) == 0.5)
                )
                .select(pl.exclude([INFLATION, N_EDGES, MATCH_THRESH, XVAL_ITER]))
                .groupby([METHOD, DENS_THRESH])
                .mean()
                .sort([METHOD, DENS_THRESH])
                .collect()
            )

            df_all_050_auc = (
                df_all_050.groupby(METHOD, maintain_order=True)
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
                .sort(AUC, descending=True)
            )

            print(df_all_050_auc)

    def main(self, re_eval: bool = True):
        if re_eval:
            self.evaluate_complex_prediction()

        self.prec_recall_curves()


if __name__ == "__main__":
    pl.Config.set_tbl_cols(30)

    inflations = [4]
    edges = ["20k_edges", "all_edges"]
    feat_methods = [
        f.lower() for f in FEATURES + list(map(lambda sf: sf["name"], SUPER_FEATS))
    ]
    sv_methods = ["cnb", "rf", "mlp", "swc"]
    n_dens = 10
    n_iters = 2

    cluster_eval = ClusterEvaluator(
        inflations=inflations,
        edges=edges,
        feat_methods=feat_methods,
        sv_methods=sv_methods,
        n_dens=n_dens,
        n_iters=n_iters,
    )
    cluster_eval.main(re_eval=False)
