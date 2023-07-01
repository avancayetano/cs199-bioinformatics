# pyright: basic

import time
from typing import Dict, List, Literal, Set, TypedDict, Union

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from sklearn.metrics import auc

from aliases import (
    AVG_PR_AUC,
    DENS_THRESH,
    F1_SCORE,
    FEATURES,
    INFLATION,
    MATCH_THRESH,
    METHOD,
    N_CLUSTERS,
    N_EDGES,
    PR_AUC,
    PREC,
    PROTEIN_U,
    PROTEIN_V,
    RECALL,
    SCENARIO,
    SUPER_FEATS,
    WEIGHT,
    XVAL_ITER,
)
from assertions import assert_prots_sorted
from utils import (
    get_clusters_filename,
    get_clusters_list,
    get_complexes_list,
    get_weighted_filename,
)

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
    def __init__(self, dip: bool):
        """
        Evaluates the quality of clusters, i.e. the number of clusters that match
        a complex, precision, recall, etc.

        Args:
            dip (bool): Whether to evaluate clusters from DIP or the original network.
        """

        self.inflations = [2, 3, 4, 5]
        self.edges: Edges = ["all_edges", "20k_edges"]
        self.feat_methods = [
            f.lower() for f in FEATURES + list(map(lambda sf: sf["name"], SUPER_FEATS))
        ]
        self.sv_methods = ["swc", "xgw"]
        self.n_dens = 100
        self.n_iters = 10

        self.methods = ["unweighted"] + self.feat_methods + self.sv_methods

        # to track the progress
        self.idx = 0
        self.total = (
            len(self.inflations)
            * (self.n_dens + 1)
            * self.n_iters
            * len(self.edges)
            * len(self.methods)
            * 2
        )
        self.dip = dip
        sns.set_palette("deep")

    def cluster_density(self, df_w: pl.DataFrame, cluster: Set[str]) -> float:
        """
        Gets the cluster density.

        Args:
            df_w (pl.DataFrame): Weighted network.
            cluster (Set[str]): Cluster.

        Returns:
            float: Cluster density.
        """
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
        """
        Caches all the necessary data.
        """
        print("Caching necessary eval data (might take a while)...")
        train_complexes: List[Subgraphs] = [
            get_complexes_list(xval_iter, "train", self.dip)
            for xval_iter in range(self.n_iters)
        ]
        test_complexes: List[Subgraphs] = [
            get_complexes_list(xval_iter, "test", self.dip)
            for xval_iter in range(self.n_iters)
        ]

        feat_clusters: AllFeatClusters = {}
        feat_weighted: FeatWeighted = {}

        sv_clusters: AllSvClusters = {}
        sv_weighted: SvWeighted = {}

        unw_clusters: AllUnwClusters = {}
        df_unweighted = pl.read_csv(
            get_weighted_filename("unweighted", False, dip=self.dip),
            new_columns=[PROTEIN_U, PROTEIN_V, WEIGHT],
            separator="\t",
            has_header=False,
        )

        for m in self.sv_methods:
            sv_weighted[m] = {}
            for j in range(self.n_iters):
                sv_weighted[m][j] = pl.read_csv(
                    get_weighted_filename(m, True, self.dip, j),
                    new_columns=[PROTEIN_U, PROTEIN_V, WEIGHT],
                    separator="\t",
                    has_header=False,
                )

        for m in self.feat_methods:
            feat_weighted[m] = pl.read_csv(
                get_weighted_filename(m, False, dip=self.dip),
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
                            get_clusters_filename(e, m, True, I, self.dip, j)
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
                    clusters = get_clusters_list(
                        get_clusters_filename(e, m, False, I, dip=self.dip)
                    )
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
                get_clusters_filename("", "unweighted", False, I, dip=self.dip)
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
        NOTE: Performance evaluation is saved in data/evals/*_cluster_evals.csv.

        Also, some terminologies.
        - cluster: predicted cluster.
        - complex: reference (aka real) complex.
        - subgraph: either cluster or complex.
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
        prefix = "dip_" if self.dip else ""
        df_evals.write_csv(f"../data/evals/{prefix}cluster_evals.csv", has_header=True)

    def evaluate_clusters(
        self,
        inflation: int,
        n_edges: Union[Literal["20k_edges"], Literal["all_edges"]],
        dens_thresh: float,
        xval_iter: int,
    ) -> List[Dict[str, Union[str, float, int]]]:
        """
        Evaluates clusters.

        Args:
            inflation (int): MCL inflation parameter.
            n_edges (Union[Literal['20k_edges'], Literal['all_edges']]): Either all or 20k edges.
            dens_thresh (float): Cluster density threshold.
            xval_iter (int): Cross-val iteration.

        Returns:
            List[Dict[str, Union[str, float, int]]]: Evaluations.
        """
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
                f"[{self.idx}/{self.total}] Done evaluating {method} clusters on {n_edges}. dens_thresh={dens_thresh}. xval_iter={xval_iter}. inflation={inflation}"
            )

        return evals

    def is_match(
        self, subgraph1: Set[str], subgraph2: Set[str], match_thresh: float
    ) -> bool:
        """
        Checks if two subgraphs match each other via Jaccard Index.

        Args:
            subgraph1 (Set[str]): First subgraph.
            subgraph2 (Set[str]): Second subgraph.
            match_thresh (float): Match threshold.

        Returns:
            bool: Whether they match or not.
        """
        jaccard_idx = len(subgraph1.intersection(subgraph2)) / len(
            subgraph1.union(subgraph2)
        )
        if jaccard_idx >= match_thresh:
            return True
        return False

    def there_is_match(
        self, subgraph: Set[str], subgraphs_set: List[Set[str]], match_thresh: float
    ) -> bool:
        """
        Checks if a subgraph matches one subgraph in a subgraph list.

        Args:
            subgraph (Set[str]): Subgraph.
            subgraphs_set (List[Set[str]]): Subgraph list.
            match_thresh (float): Match threshold.

        Returns:
            bool: There is a match or not.
        """
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
        """
        Calculates performance evaluation metrics.

        Args:
            inflation (int): MCL inflation parameter.
            n_edges (Union[Literal['20k_edges'], Literal['all_edges']]): Either 20k or all edges.
            method (str): Weighting method.
            xval_iter (int): Cross-val iteration.
            dens_thresh (float): Cluster density threshold.
            match_thresh (float): Match threshold.

        Returns:
            Dict[str, Union[str, float, int]]: Performance evaluations.
        """
        if method in self.sv_methods:
            scored_clusters = self.sv_clusters[inflation][n_edges][method][xval_iter]
        elif method in self.feat_methods:
            scored_clusters = self.feat_clusters[inflation][n_edges][method]
        else:  # for method == unweighted
            scored_clusters = self.unw_clusters[inflation]

        train_complexes = self.train_complexes[xval_iter]
        test_complexes = self.test_complexes[xval_iter]

        # Get only clusters whose density >= dens_thresh.
        clusters = list(
            map(
                lambda scored_cluster: scored_cluster["COMP_PROTEINS"],
                filter(
                    lambda scored_cluster: scored_cluster["DENSITY"] >= dens_thresh,
                    scored_clusters,
                ),
            )
        )

        # Computing the precision
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
            f1_score = (2 * prec * recall) / (prec + recall)
        else:
            print("WARNING! Zero precision and recall. F-score set to 0.")
            f1_score = 0

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
            F1_SCORE: f1_score,
        }

    def prec_recall_curves(self):
        """
        Creates four precision-recall curves per inflation setting.
        - all_edges, match_thresh = 0.5
        - all_edges, match_thresh = 0.75
        - 20k_edges, match_thresh = 0.5
        - 20k_edges, match_thresh = 0.75
        """
        prefix = "dip_" if self.dip else ""
        df_cluster_evals = pl.read_csv(f"../data/evals/{prefix}cluster_evals.csv")
        df_evals = pl.DataFrame()
        for inflation in self.inflations:
            df_all_050 = self.get_prec_recall_curve(
                df_cluster_evals, "all_edges", 0.5, inflation
            )
            df_all_075 = self.get_prec_recall_curve(
                df_cluster_evals, "all_edges", 0.75, inflation
            )
            df_20k_050 = self.get_prec_recall_curve(
                df_cluster_evals, "20k_edges", 0.5, inflation
            )
            df_20k_075 = self.get_prec_recall_curve(
                df_cluster_evals, "20k_edges", 0.75, inflation
            )

            df_all_050_auc = self.get_prec_recall_auc(df_all_050, "all_050")
            df_all_075_auc = self.get_prec_recall_auc(df_all_075, "all_075")
            df_20k_050_auc = self.get_prec_recall_auc(df_20k_050, "20k_050")
            df_20k_075_auc = self.get_prec_recall_auc(df_20k_075, "20k_075")

            # Print PR_AUC summary of the four scenarios
            df_auc_summary = (
                pl.concat(
                    [
                        df_20k_050_auc,
                        df_all_050_auc,
                        df_20k_075_auc,
                        df_all_075_auc,
                    ],
                    how="vertical",
                )
                .pivot(
                    values=PR_AUC,
                    index=METHOD,
                    columns=SCENARIO,
                    aggregate_function="first",
                )
                .with_columns(
                    [
                        (pl.sum(pl.all().exclude(METHOD)) / 4).alias(AVG_PR_AUC),
                        pl.lit(inflation).alias(INFLATION),
                    ]
                )
            )
            df_evals = pl.concat([df_evals, df_auc_summary], how="vertical")

        n_methods = 8
        df_evals_summary = (
            df_evals.groupby(METHOD)
            .mean()
            .sort(AVG_PR_AUC, descending=True)
            .head(n_methods)
            .select(pl.exclude([AVG_PR_AUC, INFLATION]))
            .melt(
                id_vars=METHOD,
                variable_name=SCENARIO,
                value_name=PR_AUC,
            )
        )
        print(df_evals.groupby(METHOD).mean().sort(AVG_PR_AUC, descending=True))
        plt.figure()
        ax = sns.barplot(
            data=df_evals_summary.to_pandas(), x=METHOD, y=PR_AUC, hue=SCENARIO
        )
        network = "DIP COMPOSITE NETWORK" if self.dip else "ORIGINAL COMPOSITE NETWORK"
        ax.set_title(
            f"{network}\nProtein Complex Detection\nTop {n_methods} weighting methods in terms of Precision-Recall AUC"
        )
        plt.xticks(rotation=15)

        # # Plot the four curves
        # TODO: plot only inflation=4 precision-recall curves.
        # self.plot_prec_recall_curve(
        #     df_all_050, df_top_methods, f"all edges, match_thresh=0.5"
        # )
        # self.plot_prec_recall_curve(
        #     df_all_075, df_top_methods, f"all edges, match_thresh=0.75"
        # )
        # self.plot_prec_recall_curve(
        #     df_20k_050, df_top_methods, f"20k edges, match_thresh=0.5"
        # )
        # self.plot_prec_recall_curve(
        #     df_20k_075, df_top_methods, f"20k edges, match_thresh=0.75"
        # )

    def plot_prec_recall_curve(
        self, df: pl.DataFrame, df_top_methods: pl.DataFrame, scenario: str
    ):
        """
        Creates plot of precision-recall curve.

        Args:
            df (pl.DataFrame): Performance evaluation dataframe.
            df_top_methods (pl.DataFrame): Top methods.
            scenario (str): Scenario.
        """
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
        self,
        df_cluster_evals: pl.DataFrame,
        n_edges: str,
        match_thresh: float,
        inflation: int,
    ) -> pl.DataFrame:
        """
        Gets the precision-recall curve given n_edges, match_thresh, and inflation.
        NOTE: Averages the precision and recall values on all the 10 cross-val iterations.
        TODO: The above method may not be the best method. Maybe, try getting the AUC per
            cross-val iteration, then average it after.

        Args:
            df_cluster_evals (pl.DataFrame): Cluster evaluations.
            n_edges (str): Either 20k or all_edges.
            match_thresh (float): Match threshold.
            inflation (int): MCL inflation parameter.

        Returns:
            pl.DataFrame: Precision-recall given the supplied settings.
        """
        df_prec_recall = (
            df_cluster_evals.lazy()
            .filter(
                (pl.col(N_EDGES) == n_edges)
                & (pl.col(MATCH_THRESH) == match_thresh)
                & (pl.col(INFLATION) == inflation)
            )
            .groupby([INFLATION, METHOD, DENS_THRESH])  # TODO: This may be unnecessary.
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
        """
        Gets the precision-recall AUC.

        Args:
            df_prec_recall (pl.DataFrame): Precision-recall dataframe.
            scenario (str): Scenario.

        Returns:
            pl.DataFrame: Dataframe containing each method and its PR_AUC.
        """
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
                .alias(PR_AUC)
            )
            .with_columns(pl.lit(scenario).alias(SCENARIO))
            .collect()
        )

        return df_auc

    def main(self, re_eval: bool = True):
        """
        Main function.

        Args:
            re_eval (bool, optional): Whether to re-evaluate the clusters or not.
                If not, read the saved performance evaluations instead. Defaults to True.
        """
        if re_eval:
            self.cache_eval_data()
            self.evaluate_complex_prediction()

        self.prec_recall_curves()


if __name__ == "__main__":
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_rows(21)
    start = time.time()

    print()
    print(
        "==================== CLUSTER EVALUATION ON COMPOSITE NETWORK =================="
    )
    cluster_eval = ClustersEvaluator(dip=False)
    cluster_eval.main(re_eval=False)
    print("==============================================================")
    print()

    print()
    print(
        "==================== CLUSTER EVALUATION ON DIP COMPOSITE NETWORK =================="
    )
    cluster_eval = ClustersEvaluator(dip=True)
    cluster_eval.main(re_eval=False)
    print("==============================================================")
    print()

    plt.show()

    print(f"Execution Time: {time.time() - start}")
