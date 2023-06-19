import time

import polars as pl

from aliases import PROTEIN_U, PROTEIN_V, PUBMED, REL
from assertions import assert_prots_sorted

PLURALITY = "PLURALITY"
REPRODUCIBILITY = "REPRODUCIBILITY"
SUM_INV_PLURALITY = "SUM_INV_PLURALITY"
SCORE = REL


class RelScoring:
    """
    Scoring of PPIs based on experiment reliability and reproducibility.
    Implementation of MV scoring by Kritikos et al. (2011) with some modifications.
    """

    def get_plurality(self, df_pubmed: pl.DataFrame) -> pl.DataFrame:
        """
        Counts the plurality of each experiment.

        Args:
            df_pubmed (pl.DataFrame): _description_

        Returns:
            pl.DataFrame: _description_
        """

        df_plurality = (
            df_pubmed.select(pl.col(PUBMED))
            .groupby(pl.col(PUBMED))
            .count()
            .rename({"count": PLURALITY})
        )

        return df_plurality

    def get_reproducibility(self, df_pubmed: pl.DataFrame) -> pl.DataFrame:
        """
        Counts the number of experiments that report each PPI.

        Args:
            df_pubmed (pl.DataFrame): _description_

        Returns:
            pl.DataFrame: _description_
        """

        df_reproducibility = (
            df_pubmed.groupby([pl.col(PROTEIN_U), pl.col(PROTEIN_V)])
            .count()
            .rename({"count": REPRODUCIBILITY})
        )
        return df_reproducibility

    def score(self, lf_w_edges: pl.LazyFrame, alpha: float) -> pl.DataFrame:
        """
        Scores each PPI based on MV Scoring.

        Args:
            df_w_edges (pl.DataFrame): _description_
            alpha (float): _description_

        Returns:
            pl.DataFrame: _description_
        """

        sum_inv_plurality_expr = (1 / (pl.col(PLURALITY))).sum()
        df_w_edges = (
            lf_w_edges.groupby(
                [PROTEIN_U, PROTEIN_V, REPRODUCIBILITY], maintain_order=True
            )
            .agg(sum_inv_plurality_expr)
            .rename({"literal": SUM_INV_PLURALITY})
            .with_columns(
                (pl.col(REPRODUCIBILITY) ** alpha * pl.col(SUM_INV_PLURALITY)).alias(
                    SCORE
                )
            )
            .select([PROTEIN_U, PROTEIN_V, SCORE])
            .collect()
        )

        df_w_edges = self.post_process_scores(df_w_edges)

        return df_w_edges

    def post_process_scores(self, df_w_edges: pl.DataFrame) -> pl.DataFrame:
        """
        Standardizes, then bounds the outliers.

        Args:
            df_w_edges (pl.DataFrame): _description_

        Returns:
            pl.DataFrame: _description_
        """

        df_w_edges = (
            df_w_edges.lazy()
            .with_columns(self.log_standardize())
            .with_columns(self.bound_outliers())
            .collect()
        )

        return df_w_edges

    def log_standardize(self) -> pl.Expr:
        """
        An expression that standardizes the score.

        Returns:
            pl.Expr: _description_
        """
        outlier_std = 3.0

        return (
            (
                (
                    (
                        (pl.col(SCORE).log() - pl.col(SCORE).log().median())
                        / pl.col(SCORE).log().std()
                    )
                    + pl.lit(outlier_std)
                )
                / pl.lit(2 * outlier_std)
            )
        ).alias(SCORE)

    def bound_outliers(self) -> pl.Expr:
        """
        Bounds outliers.

        Returns:
            pl.Expr: _description_
        """

        return (
            pl.when(pl.col(SCORE) > 1.0)
            .then(pl.lit(1.0))
            .otherwise(
                pl.when(pl.col(SCORE) < 0.0).then(pl.lit(0.0)).otherwise(pl.col(SCORE))
            )
            .alias(SCORE)
        )

    def normalize(self) -> pl.Expr:
        """
        NOTE: not used.
        Normalizes the scores.

        Returns:
            pl.Expr: _description_
        """

        return (
            (pl.col(SCORE) - pl.col(SCORE).min())
            / (pl.col(SCORE).max() - pl.col(SCORE).min())
        ).alias(SCORE)

    def main(
        self, df_edges: pl.DataFrame, df_pubmed: pl.DataFrame, alpha: float = 2
    ) -> pl.DataFrame:
        """
        RelScoring main method.

        Args:
            df_edges (pl.DataFrame): _description_
            df_pubmed (pl.DataFrame): _description_
            alpha (float, optional): _description_. Defaults to 2.

        Returns:
            pl.DataFrame: _description_
        """
        assert_prots_sorted(df_edges)
        assert_prots_sorted(df_pubmed)

        df_plurality = self.get_plurality(df_pubmed)
        df_reproducibility = self.get_reproducibility(df_pubmed)

        lf_w_edges = (
            df_pubmed.lazy()
            .join(df_plurality.lazy(), on=PUBMED, how="inner")
            .join(df_reproducibility.lazy(), on=[PROTEIN_U, PROTEIN_V], how="inner")
        )

        df_w_edges = self.score(lf_w_edges, alpha)

        df_w_edges = df_edges.join(
            df_w_edges, on=[PROTEIN_U, PROTEIN_V], how="left"
        ).fill_null(0.0)

        return df_w_edges


if __name__ == "__main__":
    start_time = time.time()

    pl.Config.set_tbl_cols(40)
    pl.Config.set_tbl_rows(10)

    df_edges = pl.read_csv(
        "../data/preprocessed/swc_edges.csv",
        has_header=False,
        new_columns=[PROTEIN_U, PROTEIN_V],
    )
    df_pubmed = pl.read_csv("../data/preprocessed/irefindex_pubmed.csv")

    rel_scoring = RelScoring()
    df_w_edges = rel_scoring.main(df_edges, df_pubmed)

    print(f">>> [{SCORE}] Scored Edges")
    print(df_w_edges)
    print(df_w_edges.describe())

    df_w_edges.write_csv("../data/scores/rel_scores.csv", has_header=True)

    # --------------------------------------------------------------
    # for the DIP composite network
    df_dip_edges = pl.read_csv(
        "../data/preprocessed/dip_edges.csv",
        has_header=False,
        new_columns=[PROTEIN_U, PROTEIN_V],
    )
    df_pubmed = pl.read_csv("../data/preprocessed/irefindex_pubmed.csv")

    rel_scoring = RelScoring()
    df_dip_w_edges = rel_scoring.main(df_dip_edges, df_pubmed)

    print(f">>> [{SCORE}] DIP Scored Edges")
    print(df_dip_w_edges)
    print(df_dip_w_edges.describe())

    df_dip_w_edges.write_csv("../data/scores/dip_rel_scores.csv", has_header=True)

    print(f">>> [{SCORE}] Execution Time")
    print(time.time() - start_time)
