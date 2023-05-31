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

    def filter_ppis(
        self, df_ppin: pl.DataFrame, srs_gene_exp_prots: pl.Series
    ) -> pl.DataFrame:
        """
        Filters out PPIs who have a protein that does not have
        gene expression data.

        Args:
            df_ppin (pl.DataFrame): _description_
            srs_gene_exp_prots (pl.Series): _description_

        Returns:
            pl.DataFrame: _description_
        """

        df_ppin_filtered = (
            df_ppin.lazy()
            .filter(
                pl.col(PROTEIN_U).is_in(srs_gene_exp_prots)
                & pl.col(PROTEIN_V).is_in(srs_gene_exp_prots)
            )
            .collect()
        )

        return df_ppin_filtered

    def melt_ppin_gene_exp(
        self,
        df_ppin_filtered: pl.DataFrame,
        df_gene_exp_std: pl.DataFrame,
        time_points: List[str],
    ) -> pl.DataFrame:
        df_ppin_gene_exp = (
            df_ppin_filtered.lazy()
            .join(
                df_gene_exp_std.lazy(),
                left_on=PROTEIN_U,
                right_on=PROTEIN,
                how="left",
            )
            .rename({t: f"{t}_{PROTEIN_U}" for t in time_points})
            .join(
                df_gene_exp_std.lazy(), left_on=PROTEIN_V, right_on=PROTEIN, how="left"
            )
            .rename({t: f"{t}_{PROTEIN_V}" for t in time_points})
            .collect()
        )

        df_melted = pl.concat(
            [
                df_ppin_gene_exp.select(
                    [PROTEIN_U] + [f"{t}_{PROTEIN_U}" for t in time_points]
                ).melt(
                    id_vars=PROTEIN_U,
                    variable_name=f"T_{PROTEIN_U}",
                    value_name=f"GE_{PROTEIN_U}",
                ),
                df_ppin_gene_exp.select(
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

    def normalize_score(self) -> pl.Expr:
        """
        Normalizes the scores.

        Returns:
            pl.Expr: _description_
        """

        return (
            (pl.col(SCORE) - pl.col(SCORE).min())
            / (pl.col(SCORE).max() - pl.col(SCORE).min())
        ).alias(SCORE)

    def mse_expr(self, col_a: str, col_b: str) -> pl.Expr:
        return (pl.col(col_a) - pl.col(col_b)).pow(2).mean()

    def standardize_gene_exp(
        self, df_gene_exp: pl.DataFrame, time_points: List[str]
    ) -> pl.DataFrame:
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

    def score(
        self, df_ppin_filtered: pl.DataFrame, df_gene_exp: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Finds the correlation of the gene expression of each protein pair per cycle.

        Args:
            df_ppin_filtered (pl.DataFrame): _description_
            df_active_pr (pl.DataFrame): _description_
            time_points (List[str]): _description_
            n (int): _description_

        Returns:
            pl.DataFrame: _description_
        """
        df_w_ppin = df_ppin_filtered.select([PROTEIN_U, PROTEIN_V])

        n_cycles = 3  # 3 cycles
        for cycle_idx in range(n_cycles):
            n = 12  # 12 time points per cycle
            time_points = [
                f"T{i}" for i in range(cycle_idx * n + 1, (cycle_idx + 1) * n + 1)
            ]

            df_gene_exp_std = self.standardize_gene_exp(df_gene_exp, time_points)

            df_melted = self.melt_ppin_gene_exp(
                df_ppin_filtered,
                df_gene_exp_std,
                time_points,
            ).select(pl.exclude([f"T_{PROTEIN_U}", f"T_{PROTEIN_V}"]))

            df_corr = (
                df_melted.lazy()
                .groupby([PROTEIN_U, PROTEIN_V], maintain_order=True)
                .agg(
                    pl.corr(f"GE_{PROTEIN_U}", f"GE_{PROTEIN_V}").alias(
                        f"CORR_{cycle_idx}"
                    )
                )
                .collect()
            )

            df_w_ppin = df_w_ppin.join(df_corr, on=[PROTEIN_U, PROTEIN_V], how="left")

        df_w_ppin = df_w_ppin.with_columns(
            (
                pl.sum([f"CORR_{cycle_idx}" for cycle_idx in range(n_cycles)])
                / n_cycles
            ).alias(SCORE)
        ).with_columns(self.normalize_score())

        return df_w_ppin

    def main(self, df_ppin: pl.DataFrame) -> pl.DataFrame:
        """
        CoExpScoring main method.

        Args:
            df_ppin (pl.DataFrame): _description_
            df_gene_exp (pl.DataFrame): _description_

        Returns:
            pl.DataFrame: _description_
        """

        # Gene expression data
        df_gene_exp = self.read_gene_expression(
            "../data/databases/GSE3431_setA_family.pcl"
        )

        srs_gene_exp_prots = self.get_gene_exp_prots(df_gene_exp)
        df_ppin_filtered = self.filter_ppis(df_ppin, srs_gene_exp_prots)

        df_w_ppin = self.score(df_ppin_filtered, df_gene_exp)

        df_w_ppin = (
            df_ppin.join(df_w_ppin, on=[PROTEIN_U, PROTEIN_V], how="left")
            .fill_null(0.0)
            .select([PROTEIN_U, PROTEIN_V, SCORE])
        )

        return df_w_ppin


if __name__ == "__main__":
    start_time = time.time()

    df_ppin = pl.read_csv("../data/preprocessed/swc_composite_data.csv").select(
        [PROTEIN_U, PROTEIN_V]
    )

    co_exp_scoring = CoExpScoring()
    df_ppin_co_exp = co_exp_scoring.main(df_ppin)

    print(f">>> [{SCORE}] Scored PPIN")
    print(df_ppin_co_exp)

    df_ppin_co_exp.write_csv("../data/scores/swc_co_exp.csv", has_header=True)

    print(f">>> [{SCORE}] Execution Time")
    print(time.time() - start_time)
