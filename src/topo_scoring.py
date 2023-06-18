import time

import polars as pl

from aliases import PROTEIN, PROTEIN_U, PROTEIN_V, TOPO, TOPO_L2
from assertions import assert_prots_sorted
from utils import sort_prot_cols

NEIGHBORS = "NEIGHBORS"
W_DEG = "W_DEG"
NUMERATOR = "NUMERATOR"
DENOMINATOR = "DENOMINATOR"


class TopoScoring:
    """
    Topological scoring of PPIs for the DIP PPIN.
    An Implementation of iterative AdjustCD [Liu et al., 2009].
    """

    def get_prot_weighted_deg(self) -> pl.Expr:
        """
        Expression that gets the weighted degree of each protein

        Returns:
            pl.Expr: _description_
        """

        return (
            pl.col(NEIGHBORS).list.eval(pl.element().list.get(1).cast(float)).list.sum()
        ).alias(W_DEG)

    def get_neighbors(self, df_w_ppin: pl.DataFrame, SCORE: str = TOPO) -> pl.LazyFrame:
        """
        Returns a DF where
        - first column is a unique protein;
        - second column is a list of its neighbors,
            together with their weights. (List[List[PROT, TOPO]])

        Args:
            df_w_ppin (pl.DataFrame): _description_

        Returns:
            pl.DataFrame: _description_

        TODO: Is there a better way to represent the second column?
        """

        lf_neighbors = (
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
        )

        return lf_neighbors

    def get_avg_prot_w_deg(
        self, df_w_ppin: pl.DataFrame, num_proteins: int, SCORE: str = TOPO
    ) -> float:
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
            .collect(streaming=True)
            .item()
        )
        return avg_weight

    def join_prot_neighbors(
        self, lf_w_ppin: pl.LazyFrame, lf_neighbors: pl.LazyFrame, PROTEIN_X: str
    ) -> pl.LazyFrame:
        """
        Augments the df_w_ppin such that the new DF contains
        - Neighbors col: list of PROTEIN_X's neighbors (List[List[PROT, TOPO]])
        - w_deg col: the weighted degree of PROTEIN_X

        Args:
            df_w_ppin (pl.DataFrame): _description_
            df_neighbors (pl.DataFrame): _description_
            PROTEIN_X (str): _description_

        Returns:
            pl.DataFrame: _description_
        """

        lf = (
            lf_w_ppin.select(pl.col(PROTEIN_X))
            .join(
                lf_neighbors,
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
        )

        return lf

    def numerator_expr(self) -> pl.Expr:
        """
        The numerator expression in the formula of AdjustCD.

        Returns:
            pl.Expr: _description_
        """

        return (
            pl.col(f"{NEIGHBORS}_{PROTEIN_U}")
            .list.concat(f"{NEIGHBORS}_{PROTEIN_V}")
            .list.eval(
                pl.element()
                .filter(pl.element().list.get(0).is_duplicated())
                .list.get(1)
                .cast(float)
            )  # the above eval gets the intersection of N_u and N_v
            .list.sum()
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
        df_w_ppin: pl.LazyFrame,
        df_neighbors: pl.LazyFrame,
        avg_prot_w_deg: float,
        SCORE: str = TOPO,
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

        lf_w_ppin = (
            pl.concat(
                [
                    self.join_prot_neighbors(
                        df_w_ppin, df_neighbors, PROTEIN_U
                    ).collect(streaming=True),
                    self.join_prot_neighbors(
                        df_w_ppin, df_neighbors, PROTEIN_V
                    ).collect(streaming=True),
                    df_w_ppin.select(pl.col(SCORE)).collect(streaming=True),
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
            # .collect(streaming=True)
        )

        return lf_w_ppin.collect(streaming=True)

    def construct_l2_network(self, df_ppin: pl.DataFrame) -> pl.DataFrame:
        df_ppin_rev = df_ppin.select(
            [pl.col(PROTEIN_U).alias(PROTEIN_V), pl.col(PROTEIN_V).alias(PROTEIN_U)]
        ).select([PROTEIN_U, PROTEIN_V])

        df_ppin = pl.concat([df_ppin, df_ppin_rev], how="vertical")
        lf_ppin = df_ppin.lazy()
        df_l2_ppin = (
            lf_ppin.join(lf_ppin, on=PROTEIN_V, how="inner")
            .drop(PROTEIN_V)
            .rename({f"{PROTEIN_U}_right": PROTEIN_V})
            .with_columns(sort_prot_cols(PROTEIN_U, PROTEIN_V))
            .filter(pl.col(PROTEIN_U) != pl.col(PROTEIN_V))
            .unique(maintain_order=True)
            .collect(streaming=True)
        )

        assert_prots_sorted(df_l2_ppin)
        return df_l2_ppin

    def weight(self, k: int, df_w_ppin: pl.DataFrame, SCORE: str = TOPO):
        for _ in range(k):
            lf_neighbors = self.get_neighbors(df_w_ppin, SCORE)

            num_proteins = lf_neighbors.collect().shape[0]

            avg_prot_w_deg = self.get_avg_prot_w_deg(df_w_ppin, num_proteins, SCORE)
            print(">>> AVG PROT TOPO")
            print(avg_prot_w_deg)

            df_w_ppin = self.score(
                df_w_ppin.lazy(), lf_neighbors, avg_prot_w_deg, SCORE
            )
            print(">>> DF_W_PPIN")
            print(df_w_ppin)

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
        df_l1_ppin = df_ppin.with_columns(pl.lit(1.0).alias(TOPO))
        df_l1_ppin = self.weight(2, df_l1_ppin, TOPO)

        print()
        print("Constructing level-2 network")
        df_l2_ppin = self.construct_l2_network(df_ppin).with_columns(
            pl.lit(1.0).alias(TOPO_L2)
        )

        df_l2_ppin = self.weight(2, df_l2_ppin, TOPO_L2)

        print(df_l2_ppin)

        print("------------------------")
        # df_w_ppin = df_l1_ppin.join(df_l2_ppin, on=[PROTEIN_U, PROTEIN_V], how="outer")

        print("-------------------- END: TOPO SCORING -------------------")

        return df_l1_ppin


if __name__ == "__main__":
    start_time = time.time()
    pl.Config.set_tbl_cols(40)
    pl.Config.set_tbl_rows(10)

    df_ppin = pl.read_csv(
        "../data/preprocessed/dip_ppin.csv",
        has_header=False,
        new_columns=[PROTEIN_U, PROTEIN_V],
    )

    assert_prots_sorted(df_ppin)
    print(df_ppin)
    topo_scoring = TopoScoring()
    df_w_ppin = topo_scoring.main(df_ppin, 2)

    print()
    print(f">>> [{TOPO} and {TOPO_L2}] Scored PPIN")
    print(df_w_ppin)

    df_w_ppin.write_csv("../data/scores/dip_topo.csv", has_header=True)

    print(f">>> [{TOPO}] Execution Time")
    print(time.time() - start_time)
