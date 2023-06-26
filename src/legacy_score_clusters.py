import time
from typing import List

import polars as pl

from aliases import FEATURES, PROTEIN_U, PROTEIN_V, SUPER_FEATS, WEIGHT
from assertions import assert_prots_sorted
from utils import get_clusters_filename, get_clusters_list, get_weighted_filename


class ScoreClusters:
    def __init__(self, dip: bool):
        self.sv_methods = ["SWC", "XGW"]
        self.feat_methods = FEATURES + [method["name"] for method in SUPER_FEATS]
        self.methods = self.sv_methods + self.feat_methods + ["unweighted"]
        self.n_iters = 10
        self.dip = dip
        self.inflations = [2, 3, 4, 5]
        self.n_edges = ["all_edges", "20k_edges"]

    def cluster_density(self, df_w: pl.DataFrame, cluster: List[str]) -> float:
        df_pairs = pl.DataFrame(
            [[u, v] for i, u in enumerate(cluster) for v in cluster[i + 1 :]],
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

    def score_clusters(
        self, clusters_list: List[List[str]], df_w: pl.DataFrame, clusters_path: str
    ):
        output = ""
        for cluster in clusters_list:
            if len(cluster) >= 3:
                density = self.cluster_density(df_w, cluster)
                output_row = f"{','.join(cluster)}\t{density}"
                output += output_row
                output += "\n"

        path = clusters_path.replace("clusters", "scored_clusters")
        with open(path, "w") as file:
            file.write(output)

        print(f"Done scoring {clusters_path}!")

    def main(self):
        for I in self.inflations:
            weighted_path = get_weighted_filename("unweighted", False, self.dip)
            clusters_path = get_clusters_filename("", "unweighted", False, I, self.dip)

            df_w = pl.read_csv(
                weighted_path,
                new_columns=[PROTEIN_U, PROTEIN_V, WEIGHT],
                separator="\t",
            )
            clusters_list = get_clusters_list(clusters_path)
            self.score_clusters(clusters_list, df_w, clusters_path)

            for e in self.n_edges:
                for method in self.sv_methods:
                    for xval_iter in range(self.n_iters):
                        weighted_path = get_weighted_filename(
                            method, True, self.dip, xval_iter
                        )
                        clusters_path = get_clusters_filename(
                            e, method, True, I, self.dip, xval_iter
                        )

                        df_w = pl.read_csv(
                            weighted_path,
                            new_columns=[PROTEIN_U, PROTEIN_V, WEIGHT],
                            separator="\t",
                        )
                        clusters_list = get_clusters_list(clusters_path)

                        self.score_clusters(clusters_list, df_w, clusters_path)

                for method in self.feat_methods:
                    weighted_path = get_weighted_filename(method, False, self.dip)
                    clusters_path = get_clusters_filename(e, method, False, I, self.dip)
                    df_w = pl.read_csv(
                        weighted_path,
                        new_columns=[PROTEIN_U, PROTEIN_V, WEIGHT],
                        separator="\t",
                    )
                    clusters_list = get_clusters_list(clusters_path)
                    self.score_clusters(clusters_list, df_w, clusters_path)


if __name__ == "__main__":
    start = time.time()
    print("===================== SCORING CLUSTERS ==========================")
    score_clusters = ScoreClusters(dip=False)
    score_clusters.main()
    print()

    print("===================== SCORING DIP CLUSTERS ==========================")
    score_clusters = ScoreClusters(dip=True)
    score_clusters.main()
    print()

    print(f">>> EXECUTION TIME: {time.time() - start}")
