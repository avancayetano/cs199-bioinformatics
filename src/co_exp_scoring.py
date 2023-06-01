import time
from typing import List

import polars as pl

from aliases import CO_EXP, PROTEIN, PROTEIN_U, PROTEIN_V

GE_MEAN = "GE_MEAN"
GE_SD = "GE_SD"
GE_THRESH = "GE_THRESH"
SCORE = CO_EXP


class CoExpScoring:
    """
    Scoring of PPIs based on their gene co-expression correlation.

    Uses the GSE3431 gene expression data by Tu et al (2005).
    """

    def read_gene_expression(self, path: str) -> pl.DataFrame:
        """
        Reads the gene expression file (GSE3431).

        Args:
            path (str): _description_

        Returns:
            pl.DataFrame: _description_
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
            df_gene_exp (pl.DataFrame): _description_

        Returns:
            pl.Series: _description_
        """

        srs_gene_exp_prots = (
            df_gene_exp.lazy().select(pl.col(PROTEIN)).unique().collect().to_series()
        )
        return srs_gene_exp_prots

    def filter_edges(
        self, df_edges: pl.DataFrame, srs_gene_exp_prots: pl.Series
    ) -> pl.DataFrame:
        """
        Filters out edges which have a protein that does not have
        gene expression data.

        Args:
            df_edges (pl.DataFrame): _description_
            srs_gene_exp_prots (pl.Series): _description_

        Returns:
            pl.DataFrame: _description_
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

    def standardize_gene_exp(
        self, df_gene_exp: pl.DataFrame, time_points: List[str]
    ) -> pl.DataFrame:
        """
        NOTE: not used...

        Args:
            df_gene_exp (pl.DataFrame): _description_
            time_points (List[str]): _description_

        Returns:
            pl.DataFrame: _description_
        """

        n = len(time_points)
        df_gene_exp_std = (
            df_gene_exp.lazy()
            .with_columns((pl.sum(time_points) / n).alias(GE_MEAN))
            .with_columns(
                (
                    pl.sum([(pl.col(t) - pl.col(GE_MEAN)).pow(2) for t in time_points])
                    / (n - 1)
                )
                .sqrt()
                .alias(GE_SD)
            )
            .with_columns(
                [
                    ((pl.col(t) - pl.col(GE_MEAN)) / pl.col(GE_SD)).alias(t)
                    for t in time_points
                ]
            )
            .select([PROTEIN] + time_points)
            .collect()
        )

        return df_gene_exp_std

    def normalize_gene_exp(
        self, df_gene_exp: pl.DataFrame, time_points: List[str]
    ) -> pl.DataFrame:
        """
        NOTE: not used...

        Args:
            df_gene_exp (pl.DataFrame): _description_
            time_points (List[str]): _description_

        Returns:
            pl.DataFrame: _description_
        """

        df_gene_exp_norm = (
            df_gene_exp.lazy()
            .with_columns(
                [
                    (
                        (pl.col(t) - pl.col(t).min())
                        / (pl.col(t).max() - pl.col(t).min())
                    ).alias(t)
                    for t in time_points
                ]
            )
            .collect()
        )

        return df_gene_exp_norm

    def melt_edges_gene_exp(
        self,
        df_filtered: pl.DataFrame,
        df_gene_exp: pl.DataFrame,
        time_points: List[str],
    ) -> pl.DataFrame:
        df_edges_gene_exp = (
            df_filtered.lazy()
            .join(
                df_gene_exp.lazy(),
                left_on=PROTEIN_U,
                right_on=PROTEIN,
                how="left",
            )
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

        return df_melted

    def remove_negative_corr(self) -> pl.Expr:
        """
        Returns:
            pl.Expr: Removes protein pairs with negative gene expression correlation.
        """
        return (
            pl.when(pl.col(SCORE) < 0)
            .then(pl.lit(0))
            .otherwise(pl.col(SCORE))
            .alias(SCORE)
        )

    def normalize(self) -> pl.Expr:
        """
        Returns:
            pl.Expr: Normalizes the scores.
        """

        return (
            (pl.col(SCORE) - pl.col(SCORE).min())
            / (pl.col(SCORE).max() - pl.col(SCORE).min())
        ).alias(SCORE)

    def rmse(self, col_a: str, col_b: str) -> pl.Expr:
        return (pl.col(col_a) - pl.col(col_b)).pow(2).mean().sqrt()

    def score(
        self, df_filtered: pl.DataFrame, df_gene_exp: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Finds the correlation of the gene expression of each protein pair per cycle.

        Args:
            df_filtered (pl.DataFrame): _description_
            df_gene_exp (pl.DataFrame): _description_

        Returns:
            pl.DataFrame: _description_
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
            df_edges (pl.DataFrame): _description_
            df_gene_exp (pl.DataFrame): _description_

        Returns:
            pl.DataFrame: _description_
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

    print(f">>> [{SCORE}] Execution Time")
    print(time.time() - start_time)
