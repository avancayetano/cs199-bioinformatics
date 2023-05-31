import time

import polars as pl

from aliases import PROTEIN_U, PROTEIN_V, PUBMED, REL
from assertions import assert_df_normalized, assert_prots_sorted

PLURALITY = "PLURALITY"
REPRODUCIBILITY = "REPRODUCIBILITY"
SUM_INV_PLURALITY = "SUM_INV_PLURALITY"
SCORE = REL


class RelScoring:
    """
    Scoring of PPIs based on experiment reliability and reproducibility.
    Implementation of MV scoring by Kritikos et al. (2011) with some modifications.
    """

    def get_plurality(self, df_ppin_pubmed: pl.DataFrame) -> pl.DataFrame:
        """
        Counts the plurality of each experiment.

        Args:
            df_ppin_pubmed (pl.DataFrame): _description_

        Returns:
            pl.DataFrame: _description_
        """

        df_plurality = (
            df_ppin_pubmed.select(pl.col(PUBMED))
            .groupby(pl.col(PUBMED))
            .count()
            .rename({"count": PLURALITY})
        )

        return df_plurality

    def get_reproducibility(self, df_ppin_pubmed: pl.DataFrame) -> pl.DataFrame:
        """
        Counts the number of experiments that report each PPI.

        Args:
            df_ppin_pubmed (pl.DataFrame): _description_

        Returns:
            pl.DataFrame: _description_
        """

        df_reproducibility = (
            df_ppin_pubmed.groupby([pl.col(PROTEIN_U), pl.col(PROTEIN_V)])
            .count()
            .rename({"count": REPRODUCIBILITY})
        )
        return df_reproducibility

    def score(self, lf_w_ppin: pl.LazyFrame, alpha: float) -> pl.DataFrame:
        """
        Scores each PPI based on MV Scoring.

        Args:
            df_w_ppin (pl.DataFrame): _description_
            alpha (float): _description_

        Returns:
            pl.DataFrame: _description_
        """

        sum_inv_plurality_expr = (1 / (pl.col(PLURALITY))).sum()
        df_w_ppin = (
            lf_w_ppin.groupby(
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

        df_w_ppin = self.post_process_scores(df_w_ppin)

        assert_df_normalized(df_w_ppin, SCORE)
        return df_w_ppin

    def post_process_scores(self, df_w_ppin: pl.DataFrame) -> pl.DataFrame:
        """
        Standardizes, then bounds the outliers, then normalizes the scores.

        Args:
            df_w_ppin (pl.DataFrame): _description_

        Returns:
            pl.DataFrame: _description_
        """

        df = (
            df_w_ppin.lazy()
            .with_columns(self.log_standardize())
            .with_columns(self.bound_outliers())
            .with_columns(self.normalize())
            .collect()
        )

        return df

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
                        (pl.col(SCORE).log() - pl.col(SCORE).log().mean())
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
        Normalizes the scores.

        Returns:
            pl.Expr: _description_
        """

        return (
            (pl.col(SCORE) - pl.col(SCORE).min())
            / (pl.col(SCORE).max() - pl.col(SCORE).min())
        ).alias(SCORE)

    def main(
        self, df_ppin: pl.DataFrame, df_ppin_pubmed: pl.DataFrame, alpha: float = 2
    ) -> pl.DataFrame:
        """
        RelScoring main method.

        Args:
            df_ppin (pl.DataFrame): _description_
            df_ppin_pubmed (pl.DataFrame): _description_
            alpha (float, optional): _description_. Defaults to 2.

        Returns:
            pl.DataFrame: _description_
        """
        assert_prots_sorted(df_ppin)
        assert_prots_sorted(df_ppin_pubmed)

        df_plurality = self.get_plurality(df_ppin_pubmed)
        df_reproducibility = self.get_reproducibility(df_ppin_pubmed)

        lf_w_ppin = (
            df_ppin_pubmed.lazy()
            .join(df_plurality.lazy(), on=PUBMED, how="inner")
            .join(df_reproducibility.lazy(), on=[PROTEIN_U, PROTEIN_V], how="inner")
        )

        df_w_ppin = self.score(lf_w_ppin, alpha)

        df_w_ppin = df_ppin.join(
            df_w_ppin, on=[PROTEIN_U, PROTEIN_V], how="left"
        ).fill_null(0.0)

        return df_w_ppin


if __name__ == "__main__":
    start_time = time.time()

    pl.Config.set_tbl_cols(40)
    pl.Config.set_tbl_rows(10)

    df_ppin = pl.read_csv("../data/preprocessed/swc_composite_data.csv").select(
        [PROTEIN_U, PROTEIN_V]
    )
    df_ppin_pubmed = pl.read_csv("../data/preprocessed/irefindex_pubmed.csv")

    rel_scoring = RelScoring()
    df_ppin_rel = rel_scoring.main(df_ppin, df_ppin_pubmed)

    print(f">>> [{SCORE}] Scored PPIN")
    print(df_ppin_rel)

    df_ppin_rel.write_csv("../data/scores/swc_rel.csv", has_header=True)

    print(f">>> [{SCORE}] Execution Time")
    print(time.time() - start_time)
