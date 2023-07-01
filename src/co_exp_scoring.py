import time
from typing import List

import polars as pl

from aliases import CO_EXP, PROTEIN, PROTEIN_U, PROTEIN_V
from assertions import assert_gene_exp_arranged, assert_same_edges

GE_MEAN = "GE_MEAN"
GE_SD = "GE_SD"
GE_THRESH = "GE_THRESH"
SCORE = CO_EXP


class CoExpScoring:
    """
    Scoring of PPIs based on their gene co-expression correlation using
    Pearson correlation coefficient.

    Uses the GSE3431 gene expression data by Tu et al (2005).
    """

    def read_gene_expression(self, path: str) -> pl.DataFrame:
        """
        Reads the gene expression file (GSE3431).

        Args:
            path (str): Path to the gene expression file.

        Returns:
            pl.DataFrame: Dataframe of the gene expression file.
        """
        df_gene_exp = (
            pl.scan_csv(path, has_header=True, separator="\t", skip_rows_after_header=1)
            .select(pl.exclude(["NAME", "GWEIGHT"]))
            .rename({"YORF": PROTEIN})
        ).collect()

        return df_gene_exp

    def get_gene_exp_prots(self, df_gene_exp: pl.DataFrame) -> pl.Series:
        """
        Gets all the unique proteins in the gene expression data.

        Args:
            df_gene_exp (pl.DataFrame): Gene expression dataframe.

        Returns:
            pl.Series: Series of unique proteins in gene expression data.
        """
        srs_gene_exp_prots = (
            df_gene_exp.lazy().select(pl.col(PROTEIN)).unique().collect().to_series()
        )
        return srs_gene_exp_prots

    def filter_edges(
        self, df_edges: pl.DataFrame, srs_gene_exp_prots: pl.Series
    ) -> pl.DataFrame:
        """
        Filters out edges which have a protein that does not have gene expression data.

        Args:
            df_edges (pl.DataFrame): Edges of the composite network.
            srs_gene_exp_prots (pl.Series): Series of unique proteins in gene expression data.

        Returns:
            pl.DataFrame: Filtered version of df_edges.
        """
        df_filtered = (
            df_edges.lazy()
            .filter(
                pl.col(PROTEIN_U).is_in(srs_gene_exp_prots)
                & pl.col(PROTEIN_V).is_in(srs_gene_exp_prots)
            )
            .collect()
        )

        return df_filtered

    def melt_edges_gene_exp(
        self,
        df_filtered: pl.DataFrame,
        df_gene_exp: pl.DataFrame,
        time_points: List[str],
    ) -> pl.DataFrame:
        """
        Combines the edges of the protein network and the gene expression data.

        Args:
            df_filtered (pl.DataFrame): Filtered version of df_edges.
            df_gene_exp (pl.DataFrame): Gene expression data.
            time_points (List[str]): List of time points.

        Returns:
            pl.DataFrame: Melted dataframe containing the edges of the protein network
                and gene expression data.
        """
        df_edges_gene_exp = (
            df_filtered.lazy()
            .join(df_gene_exp.lazy(), left_on=PROTEIN_U, right_on=PROTEIN, how="left")
            .rename({t: f"{t}_{PROTEIN_U}" for t in time_points})
            .join(df_gene_exp.lazy(), left_on=PROTEIN_V, right_on=PROTEIN, how="left")
            .rename({t: f"{t}_{PROTEIN_V}" for t in time_points})
            .collect()
        )

        df_melted = pl.concat(
            [
                df_edges_gene_exp.select(
                    [PROTEIN_U] + [f"{t}_{PROTEIN_U}" for t in time_points]
                ).melt(
                    id_vars=PROTEIN_U,
                    variable_name=f"T_{PROTEIN_U}",
                    value_name=f"GE_{PROTEIN_U}",
                ),
                df_edges_gene_exp.select(
                    [PROTEIN_V] + [f"{t}_{PROTEIN_V}" for t in time_points]
                ).melt(
                    id_vars=PROTEIN_V,
                    variable_name=f"T_{PROTEIN_V}",
                    value_name=f"GE_{PROTEIN_V}",
                ),
            ],
            how="horizontal",
        )
        assert_same_edges(df_filtered, df_melted)
        assert_gene_exp_arranged(df_melted)

        return df_melted

    def remove_negative_corr(self) -> pl.Expr:
        """
        An Expr that removes protein pairs with negative gene expression correlation.

        Returns:
            pl.Expr: Expr that does the aforementioned.
        """
        return (
            pl.when(pl.col(SCORE) < 0.0)
            .then(pl.lit(0.0))
            .otherwise(pl.col(SCORE))
            .alias(SCORE)
        )

    def score(
        self, df_filtered: pl.DataFrame, df_gene_exp: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Finds the correlation of the gene expression of each protein pair.

        Args:
            df_filtered (pl.DataFrame): Filtered df_edges.
            df_gene_exp (pl.DataFrame): Gene expression data.

        Returns:
            pl.DataFrame: Weighted df_filtered.
        """

        time_points = [f"T{i}" for i in range(1, 37)]  # 36 time points

        df_melted = self.melt_edges_gene_exp(
            df_filtered,
            df_gene_exp,
            time_points,
        ).select(pl.exclude([f"T_{PROTEIN_U}", f"T_{PROTEIN_V}"]))

        df_w_edges = (
            df_melted.lazy()
            .groupby([PROTEIN_U, PROTEIN_V], maintain_order=True)
            .agg(pl.corr(f"GE_{PROTEIN_U}", f"GE_{PROTEIN_V}").alias(SCORE))
            .with_columns(self.remove_negative_corr())
            .collect()
        )

        return df_w_edges

    def main(self, df_edges: pl.DataFrame) -> pl.DataFrame:
        """
        CoExpScoring main method.

        Args:
            df_edges (pl.DataFrame): Edges of the composite protein network.
            df_gene_exp (pl.DataFrame): Gene expression data.

        Returns:
            pl.DataFrame: Weighted protein network.
        """

        # Gene expression data
        df_gene_exp = self.read_gene_expression(
            "../data/databases/GSE3431_setA_family.pcl"
        )

        srs_gene_exp_prots = self.get_gene_exp_prots(df_gene_exp)
        df_filtered = self.filter_edges(df_edges, srs_gene_exp_prots)

        df_w_edges = self.score(df_filtered, df_gene_exp)

        df_w_edges = (
            df_edges.join(df_w_edges, on=[PROTEIN_U, PROTEIN_V], how="left")
            .fill_null(0.0)
            .select([PROTEIN_U, PROTEIN_V, SCORE])
        )

        return df_w_edges


if __name__ == "__main__":
    start_time = time.time()

    df_edges = pl.read_csv(
        "../data/preprocessed/swc_edges.csv",
        has_header=False,
        new_columns=[PROTEIN_U, PROTEIN_V],
    )

    co_exp_scoring = CoExpScoring()
    df_w_edges = co_exp_scoring.main(df_edges)

    print(f">>> [{SCORE}] Scored Edges")
    print(df_w_edges)

    print(df_w_edges.describe())

    df_w_edges.write_csv("../data/scores/co_exp_scores.csv", has_header=True)

    # --------------------------------------------------------------------------
    # for the DIP composite network
    df_dip_edges = pl.read_csv(
        "../data/preprocessed/dip_edges.csv",
        has_header=False,
        new_columns=[PROTEIN_U, PROTEIN_V],
    )

    co_exp_scoring = CoExpScoring()
    df_dip_w_edges = co_exp_scoring.main(df_dip_edges)

    print(f">>> [{SCORE}] DIP Scored Edges")
    print(df_dip_w_edges)

    print(df_dip_w_edges.describe())

    df_dip_w_edges.write_csv("../data/scores/dip_co_exp_scores.csv", has_header=True)

    print(f">>> [{SCORE}] Execution Time")
    print(time.time() - start_time)
