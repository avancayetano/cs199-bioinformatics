import time
from typing import List

import polars as pl

from aliases import PROTEIN, PROTEIN_U, PROTEIN_V, SWC_FEATS, TOPO, TOPO_L2
from assertions import assert_prots_sorted
from utils import get_unique_proteins, sort_prot_cols

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

    def get_neighbors(self, df_w_ppin: pl.DataFrame) -> pl.DataFrame:
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

        df_neighbors = (
            df_w_ppin.vstack(
                df_w_ppin.select(
                    [
                        pl.col(PROTEIN_V).alias(PROTEIN_U),
                        pl.col(PROTEIN_U).alias(PROTEIN_V),
                        pl.col(TOPO),
                    ],
                )
            )
            .lazy()
            .groupby(pl.col(PROTEIN_U), maintain_order=True)
            .agg(pl.concat_list([pl.col(PROTEIN_V), pl.col(TOPO)]))
            .rename({PROTEIN_U: PROTEIN, PROTEIN_V: NEIGHBORS})
            .with_columns(self.get_prot_weighted_deg())
            .collect()
        )

        return df_neighbors

    def get_avg_prot_w_deg(self, df_neighbors: pl.DataFrame) -> float:
        """
        Gets the average weighted degree of all the proteins.
        """

        avg_weight = df_neighbors.select(
            (pl.col(W_DEG).sum()) / pl.count(PROTEIN)
        ).item()
        return avg_weight

    def join_prot_neighbors(
        self, df_w_ppin: pl.DataFrame, df_neighbors: pl.DataFrame, PROTEIN_X: str
    ) -> pl.DataFrame:
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

        df = (
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
            .collect(streaming=True)
        )

        return df

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

    def score_batch(
        self,
        df_w_ppin_batch: pl.DataFrame,
        df_neighbors: pl.DataFrame,
        avg_prot_w_deg: float,
        SCORE: str,
    ):
        df_joined = pl.concat(
            [
                self.join_prot_neighbors(df_w_ppin_batch, df_neighbors, PROTEIN_U),
                self.join_prot_neighbors(df_w_ppin_batch, df_neighbors, PROTEIN_V),
                df_w_ppin_batch.select(pl.col(SCORE)),
            ],
            how="horizontal",
        )

        print(
            df_joined.filter(
                (pl.col(PROTEIN_U) == "YCR094W") & (pl.col(PROTEIN_V) == "YEL070W")
            )
        )
        # YCR094W,YJR091C
        # YCR094W   ┆ YEL070W
        df_w_ppin_batch = (
            df_joined.lazy()
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
            .collect(streaming=True)
        )

        return df_w_ppin_batch

    def score(
        self,
        df_w_ppin: pl.DataFrame,
        df_neighbors: pl.DataFrame,
        avg_prot_w_deg: float,
        SCORE: str = TOPO,
        n_batches: int = 1,
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
        if n_batches == 1 or n_batches == 0:
            df_w_ppin = self.score_batch(df_w_ppin, df_neighbors, avg_prot_w_deg, SCORE)
        else:
            print(f">>> DATAFRAME TOO LARGE, SCORING BY {n_batches} BATCHES")
            size_df = df_w_ppin.shape[0]
            batch_size = size_df // n_batches
            df_scored_batches: List[pl.DataFrame] = []
            for i in range(n_batches + 1):
                start = batch_size * i
                if start < size_df:
                    end = batch_size * (i + 1)
                    df_w_ppin_batch = df_w_ppin.slice(start, end - start)
                    df_w_ppin_batch = self.score_batch(
                        df_w_ppin_batch, df_neighbors, avg_prot_w_deg, SCORE
                    )
                    df_scored_batches.append(df_w_ppin_batch)
                    print(f">>> DONE SCORING BATCH={i} | BATCH_SIZE = {end - start}")
            df_w_ppin = pl.concat(df_scored_batches, how="vertical")
        return df_w_ppin

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
            .join(df_ppin.lazy(), on=[PROTEIN_U, PROTEIN_V], how="anti")
            .collect(streaming=True)
        )

        assert_prots_sorted(df_l2_ppin)
        return df_l2_ppin

    def weight(self, k: int, df_l1_ppin: pl.DataFrame, df_l2_ppin: pl.DataFrame):
        df_w_ppin = pl.DataFrame()
        for i in range(k):
            df_neighbors = self.get_neighbors(df_l1_ppin)
            print()

            print(f"-------------- ADJUSTCD ITERATION = {i} --------------------")
            print(">>> DF NEIGHBORS")
            print(df_neighbors)

            avg_prot_w_deg = self.get_avg_prot_w_deg(df_neighbors)
            print(">>> AVG PROT W_DEG")
            print(avg_prot_w_deg)

            df_l1_ppin = self.score(df_l1_ppin, df_neighbors, avg_prot_w_deg, TOPO)
            df_l2_ppin = self.score(df_l2_ppin, df_neighbors, avg_prot_w_deg, TOPO_L2)

            print(f">>> DF_W_PPIN | k = {k}")
            df_w_ppin = df_l1_ppin.join(
                df_l2_ppin, on=[PROTEIN_U, PROTEIN_V], how="outer"
            ).fill_null(0.0)
            print(df_w_ppin)
            print()
        df_w_ppin = df_w_ppin.filter((pl.col(TOPO_L2) > 0.1))
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
        df_l2_ppin = self.construct_l2_network(df_ppin).with_columns(
            pl.lit(1.0).alias(TOPO_L2)
        )
        df_w_ppin = self.weight(2, df_l1_ppin, df_l2_ppin)

        # print("ppppppppppppppppppppppppp")
        # print(df_l2_ppin.filter(pl.col(PROTEIN_U) == "YEL070W"))

        # df_l2_ppin = self.weight(2, df_l2_ppin, TOPO_L2, 10).filter(
        #     pl.col(TOPO_L2) > 0.1
        # )

        # print(df_l2_ppin)

        # print("------------------------")
        # df_w_ppin = df_l1_ppin.join(
        #     df_l2_ppin, on=[PROTEIN_U, PROTEIN_V], how="outer"
        # ).fill_null(0.0)

        print("-------------------- END: TOPO SCORING -------------------")

        return df_w_ppin


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
    num_proteins = get_unique_proteins(df_ppin).shape[0]

    print(f"Num of proteins: {num_proteins}")
    print("-------------------------------------------")

    topo_scoring = TopoScoring()
    df_w_ppin = topo_scoring.main(df_ppin, 2)

    print()
    print(f">>> [{TOPO} and {TOPO_L2}] Scored PPIN")
    print(df_w_ppin)

    df_w_ppin.write_csv("../data/scores/dip_topo.csv", has_header=True)

    # df_w_ppin = pl.read_csv("../data/scores/dip_topo.csv")
    # df_swc = pl.read_csv("../data/scores/swc_composite_scores.csv").drop(
    #     [TOPO, TOPO_L2]
    # )

    # df_dip = (
    #     df_w_ppin.join(df_swc, on=[PROTEIN_U, PROTEIN_V], how="outer")
    #     .fill_null(0.0)
    #     .filter(pl.sum(SWC_FEATS) > 0)
    # )

    # print(df_dip)

    print(f">>> [{TOPO}] Execution Time")
    print(time.time() - start_time)
