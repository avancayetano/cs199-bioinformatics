import time

import polars as pl

from aliases import PROTEIN, PROTEIN_U, PROTEIN_V, TOPO
from utils import read_no_header_file

NEIGHBORS = "NEIGHBORS"
W_DEG = "W_DEG"
NUMERATOR = "NUMERATOR"
DENOMINATOR = "DENOMINATOR"
SCORE = TOPO


class TopoScoring:
    """
    Topological scoring of PPIs.
    Implementation of AdjustCD [Liu et al., 2009]
    """

    def get_all_proteins(self, df_ppin: pl.DataFrame) -> pl.Series:
        """
        Returns a series of all the unique proteins of the PPIN.

        Args:
            df_ppin (pl.DataFrame): _description_

        Returns:
            pl.Series: _description_
        """

        srs_proteins = (
            df_ppin.select(pl.col(PROTEIN_U).alias(PROTEIN))
            .to_series()
            .append(df_ppin.select(PROTEIN_V).to_series())
        ).unique()

        return srs_proteins

    def get_prot_weighted_deg(self) -> pl.Expr:
        """
        Expression that gets the weighted degree of each protein

        Returns:
            pl.Expr: _description_
        """

        return (
            pl.col(NEIGHBORS).arr.eval(pl.element().arr.get(1).cast(float)).arr.sum()
        ).alias(W_DEG)

    def get_neighbors(self, df_w_ppin: pl.DataFrame) -> pl.DataFrame:
        """
        Returns a DF where
        - first column is a unique protein;
        - second column is a list of its neighbors,
            together with their weights. (List[List[PROT, SCORE]])

        Args:
            df_w_ppin (pl.DataFrame): _description_

        Returns:
            pl.DataFrame: _description_

        TODO: Is there a better way to represent the second column?
        """

        df_neighbors = (
            df_w_ppin.vstack(
                df_w_ppin.select(
                    [
                        pl.col(PROTEIN_V).alias(PROTEIN_U),
                        pl.col(PROTEIN_U).alias(PROTEIN_V),
                        pl.col(SCORE),
                    ],
                )
            )
            .lazy()
            .groupby(pl.col(PROTEIN_U))
            .agg((pl.concat_list([pl.col(PROTEIN_V), pl.col(SCORE)])))
            .rename({PROTEIN_U: PROTEIN, PROTEIN_V: NEIGHBORS})
            .with_columns(self.get_prot_weighted_deg())
            .collect()
        )

        return df_neighbors

    def get_avg_prot_w_deg(self, df_w_ppin: pl.DataFrame, num_proteins: int) -> float:
        """
        Gets the average weighted degree of all the proteins.

        Args:
            df_w_ppin (pl.DataFrame): _description_
            num_proteins (int): _description_

        Returns:
            float: _description_
        """

        avg_weight = (
            df_w_ppin.lazy()
            .select((pl.col(SCORE).sum() * 2) / num_proteins)
            .collect()
            .item()
        )
        return avg_weight

    def join_prot_neighbors(
        self, df_w_ppin: pl.DataFrame, df_neighbors: pl.DataFrame, PROTEIN_X: str
    ) -> pl.DataFrame:
        """
        Augments the df_w_ppin such that the new DF contains
        - Neighbors col: list of PROTEIN_X's neighbors (List[List[PROT, SCORE]])
        - w_deg col: the weighted degree of PROTEIN_X

        Args:
            df_w_ppin (pl.DataFrame): _description_
            df_neighbors (pl.DataFrame): _description_
            PROTEIN_X (str): _description_

        Returns:
            pl.DataFrame: _description_
        """

        return (
            df_w_ppin.lazy()
            .select(pl.col(PROTEIN_X))
            .join(
                df_neighbors.lazy(),
                left_on=PROTEIN_X,
                right_on=PROTEIN,
                how="inner",
            )
            .rename(
                {
                    NEIGHBORS: f"{NEIGHBORS}_{PROTEIN_X}",
                    W_DEG: f"{W_DEG}_{PROTEIN_X}",
                }
            )
            .collect()
        )

    def numerator_expr(self) -> pl.Expr:
        """
        The numerator expression in the formula of AdjustCD.

        Returns:
            pl.Expr: _description_
        """

        return (
            pl.col(f"{NEIGHBORS}_{PROTEIN_U}")
            .arr.concat(f"{NEIGHBORS}_{PROTEIN_V}")
            .arr.eval(
                pl.element()
                .filter(pl.element().arr.get(0).is_duplicated())
                .arr.get(1)
                .cast(float)
            )  # the above eval gets the intersection of N_u and N_v
            .arr.sum()
        ).alias(NUMERATOR)

    def denominator_expr(self, avg_prot_w_deg: float) -> pl.Expr:
        """
        The denominator expression in the formula of AdjustCD.

        Args:
            avg_prot_w_deg (float): _description_

        Returns:
            pl.Expr: _description_
        """

        return (
            (
                pl.when(pl.col(f"{W_DEG}_{PROTEIN_U}") > pl.lit(avg_prot_w_deg))
                .then(pl.col(f"{W_DEG}_{PROTEIN_U}"))
                .otherwise(pl.lit(avg_prot_w_deg))
            )
            + (
                pl.when(pl.col(f"{W_DEG}_{PROTEIN_V}") > pl.lit(avg_prot_w_deg))
                .then(pl.col(f"{W_DEG}_{PROTEIN_V}"))
                .otherwise(pl.lit(avg_prot_w_deg))
            )
        ).alias(DENOMINATOR)

    def score(
        self,
        df_w_ppin: pl.DataFrame,
        df_neighbors: pl.DataFrame,
        avg_prot_w_deg: float,
    ) -> pl.DataFrame:
        """
        Scores the each PPI of the PPIN.

        Args:
            df_w_ppin (pl.DataFrame): _description_
            df_neighbors (pl.DataFrame): _description_
            avg_prot_w_deg (float): _description_

        Returns:
            pl.DataFrame: _description_
        """

        df_w_ppin = (
            pl.concat(
                [
                    self.join_prot_neighbors(df_w_ppin, df_neighbors, PROTEIN_U),
                    self.join_prot_neighbors(df_w_ppin, df_neighbors, PROTEIN_V),
                    df_w_ppin.select(pl.col(SCORE)),
                ],
                how="horizontal",
            )
            .lazy()
            .with_columns(
                [self.numerator_expr(), self.denominator_expr(avg_prot_w_deg)]
            )
            .drop(
                [
                    f"{NEIGHBORS}_{PROTEIN_U}",
                    f"{NEIGHBORS}_{PROTEIN_V}",
                    f"{W_DEG}_{PROTEIN_U}",
                    f"{W_DEG}_{PROTEIN_V}",
                ]
            )
            .with_columns((pl.col(NUMERATOR) / pl.col(DENOMINATOR)).alias(SCORE))
            .drop([NUMERATOR, DENOMINATOR])
            .select([PROTEIN_U, PROTEIN_V, SCORE])
            .collect()
        )

        return df_w_ppin

    def main(self, df_ppin: pl.DataFrame, k: int = 2) -> pl.DataFrame:
        """
        TopoScoring main method.
        Scores PPIs from the original PPIN only.
        Doesn't score indirect interactions.

        Args:
            df_ppin (pl.DataFrame): _description_
            k (int, optional): _description_. Defaults to 2.

        Returns:
            pl.DataFrame: _description_
        """

        # Weighted PPIN at k=0
        df_w_ppin = df_ppin.with_columns(pl.lit(1.0).alias(SCORE))

        for _ in range(k):
            df_neighbors = self.get_neighbors(df_w_ppin)
            num_proteins = df_neighbors.shape[0]
            print(">>> DF NEIGHBORS")
            print(df_neighbors)

            avg_prot_w_deg = self.get_avg_prot_w_deg(df_w_ppin, num_proteins)
            print(">>> AVG PROT SCORE")
            print(avg_prot_w_deg)

            df_w_ppin = self.score(df_w_ppin, df_neighbors, avg_prot_w_deg)
            print(">>> DF_W_PPIN")
            print(df_w_ppin)

        return df_w_ppin


if __name__ == "__main__":
    start_time = time.time()
    pl.Config.set_tbl_cols(40)
    pl.Config.set_tbl_rows(10)

    df_ppin = read_no_header_file(
        "../data/preprocessed/swc_ppin.csv", [PROTEIN_U, PROTEIN_V]
    )

    topo_scoring = TopoScoring()
    df_ppin_topo = topo_scoring.main(df_ppin, 2)

    print(f">>> [{SCORE}] Scored PPIN")
    print(df_ppin_topo)

    df_ppin_topo.write_csv("../data/scores/swc_topo.csv", has_header=True)

    print(f">>> [{SCORE}] Execution Time")
    print(time.time() - start_time)
