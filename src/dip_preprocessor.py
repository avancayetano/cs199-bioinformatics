import time
from typing import List

import polars as pl

from aliases import (
    CO_OCCUR,
    PROTEIN,
    PROTEIN_U,
    PROTEIN_V,
    STRING,
    SWC_FEATS,
    TOPO,
    TOPO_L2,
)
from assertions import assert_prots_sorted
from utils import get_unique_proteins, sort_prot_cols

NEIGHBORS = "NEIGHBORS"
W_DEG = "W_DEG"
NUMERATOR = "NUMERATOR"
DENOMINATOR = "DENOMINATOR"

"""
This is a script that enriches the DIP PPIN into a composite network by 
topological scoring and integrating CO_OCCUR and STRING.
"""


class TopoScoring:
    """
    Topological scoring of PPIs for the DIP PPIN.
    An implementation of iterative AdjustCD [Liu et al., 2009].
    """

    def __init__(self, n_batches: int = 1) -> None:
        """
        Args:
            n_batches (int, optional): Number of batches (to avoid out-of-memory errors).
                Defaults to 1.
        """
        self.n_batches = n_batches

    def construct_l2_network(self, df_ppin: pl.DataFrame) -> pl.DataFrame:
        """
        Construct a protein network containing only level-2 neighbors.

        Args:
            df_ppin (pl.DataFrame): The PPIN.

        Returns:
            pl.DataFrame: Level-2 protein network.
        """
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

    def get_prot_weighted_deg(self) -> pl.Expr:
        """
        Expression that gets the weighted degree of each protein

        Returns:
            pl.Expr: Expr that does the aforementioned.
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
            df_w_ppin (pl.DataFrame): Weighted PPIN at the previous iteration.

        Returns:
            pl.DataFrame: The dataframe described above.
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
            .collect(streaming=True)
        )

        return df_neighbors

    def get_avg_prot_w_deg(self, df_neighbors: pl.DataFrame) -> float:
        """
        Gets the average weighted degree of all the proteins.

        Args:
            df_neighbors (pl.DataFrame): Dataframe returned by get_neighbors.

        Returns:
            float: The average weighted degree of all the proteins.
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
            df_w_ppin (pl.DataFrame): Weighted PPIN at the previous iteration.
            df_neighbors (pl.DataFrame): Dataframe returned by get_neighbors.
            PROTEIN_X (str): Either PROTEIN_U or PROTEIN_V.

        Returns:
            pl.DataFrame: Dataframe described above.
        """
        df = (
            df_w_ppin.lazy()
            .select([PROTEIN_U, PROTEIN_V])
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
            pl.Expr: Expr of the above.
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
            avg_prot_w_deg (float): Average weighted degree returned by get_avg_prot_w_deg.

        Returns:
            pl.Expr: Expr of the above.
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
    ) -> pl.DataFrame:
        """
        Scores a batch of PPIN edges.

        Args:
            df_w_ppin_batch (pl.DataFrame): Weighted PPIN batch from the previous iteration.
            df_neighbors (pl.DataFrame): Neighbors dataframe.
            avg_prot_w_deg (float): Average protein weighted degree.
            SCORE (str): Either TOPO or TOPO_L2.

        Returns:
            pl.DataFrame: Weighted PPIN batch.
        """
        lf_joined = (
            self.join_prot_neighbors(df_w_ppin_batch, df_neighbors, PROTEIN_U)
            .lazy()
            .join(
                self.join_prot_neighbors(
                    df_w_ppin_batch, df_neighbors, PROTEIN_V
                ).lazy(),
                on=[PROTEIN_U, PROTEIN_V],
            )
            .join(df_w_ppin_batch.lazy(), on=[PROTEIN_U, PROTEIN_V])
        )

        df_w_ppin_batch = (
            lf_joined.with_columns(
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
    ) -> pl.DataFrame:
        """
        Scores each PPI of the PPIN.

        Args:
            df_w_ppin (pl.DataFrame): Weighted PPIN from the previous iteration.
            df_neighbors (pl.DataFrame): Neighbors dataframe.
            avg_prot_w_deg (float): Average protein weighted degree.
            SCORE (str, optional): Either TOPO or TOPO_L2. Defaults to TOPO.

        Returns:
            pl.DataFrame: _description_
        """
        if self.n_batches == 1:
            print(
                f">>> NOTE: if memory allocation failed, score by batches by adjusting n_batches"
            )
            df_w_ppin = self.score_batch(df_w_ppin, df_neighbors, avg_prot_w_deg, SCORE)
        else:
            print(f">>> DATAFRAME TOO LARGE, SCORING BY {self.n_batches} BATCHES")
            size_df = df_w_ppin.shape[0]
            batch_size = size_df // self.n_batches
            df_scored_batches: List[pl.DataFrame] = []
            for i in range(self.n_batches + 1):
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

    def weight(
        self, k: int, df_l1_ppin: pl.DataFrame, df_l2_ppin: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Weights the level-1 and level-2 PPIN via iterative AdjustCD.

        Args:
            k (int): Number of iterations.
            df_l1_ppin (pl.DataFrame): Level-1 PPIN.
            df_l2_ppin (pl.DataFrame): Level-2 PPIN.

        Returns:
            pl.DataFrame: Weighted level-1 and level-2 PPIN.
        """
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
        df_w_ppin = df_w_ppin.filter((pl.col(TOPO_L2) > 0.1) | (pl.col(TOPO) > 0))
        return df_w_ppin

    def main(self, df_ppin: pl.DataFrame, k: int = 2) -> pl.DataFrame:
        """
        TopoScoring main method.

        Args:
            df_ppin (pl.DataFrame): Original PPIN.
            k (int, optional): Number of iterations. Defaults to 2.

        Returns:
            pl.DataFrame: Weighted level-1 and level-2 PPIN.
        """

        # Weighted PPIN at k=0
        df_l1_ppin = df_ppin.with_columns(pl.lit(1.0).alias(TOPO))
        df_l2_ppin = self.construct_l2_network(df_ppin).with_columns(
            pl.lit(0.0).alias(TOPO_L2)
        )
        df_w_ppin = self.weight(k, df_l1_ppin, df_l2_ppin)

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

    topo_scoring = TopoScoring(n_batches=1)
    # due to rounding errors, need to bound the scores...
    df_w_ppin = topo_scoring.main(df_ppin, 2).select(
        [
            PROTEIN_U,
            PROTEIN_V,
            pl.when(pl.col(TOPO) > 1.0)
            .then(pl.lit(1.0))
            .otherwise(
                pl.when(pl.col(TOPO) < 0.0).then(pl.lit(0.0)).otherwise(pl.col(TOPO))
            )
            .alias(TOPO),
            pl.when(pl.col(TOPO_L2) > 1.0)
            .then(pl.lit(1.0))
            .otherwise(
                pl.when(pl.col(TOPO_L2) < 0.0)
                .then(pl.lit(0.0))
                .otherwise(pl.col(TOPO_L2))
            )
            .alias(TOPO_L2),
        ]
    )

    print()
    print(f">>> [{TOPO} and {TOPO_L2}] Scored PPIN")
    print(df_w_ppin)

    print(df_w_ppin.describe())

    df_swc = pl.read_csv("../data/scores/swc_composite_scores.csv").drop(
        [TOPO, TOPO_L2]
    )

    df_dip = (
        df_w_ppin.join(df_swc, on=[PROTEIN_U, PROTEIN_V], how="outer")
        .fill_null(0.0)
        .filter(pl.sum(SWC_FEATS) > 0)
        .sort([PROTEIN_U, PROTEIN_V])
    )

    print(f">>> PREPROCESSED DIP COMPOSITE NETWORK")
    print(df_dip)

    df_dip.write_csv("../data/scores/dip_swc_composite_scores.csv", has_header=True)

    df_dip.select([PROTEIN_U, PROTEIN_V]).write_csv(
        "../data/preprocessed/dip_edges.csv", has_header=False
    )

    print("REFORMATTING THE DIP NETWORK FOR THE SWC SOFTWARE")
    assert_prots_sorted(df_dip)
    mapping = {TOPO: "PPI", TOPO_L2: "PPIL2", CO_OCCUR: "PUBMED", STRING: STRING}
    df_dip_reformat = df_dip.rename(mapping).melt(
        id_vars=[PROTEIN_U, PROTEIN_V], variable_name="FEATURE", value_name="SCORE"
    )
    df_dip_reformat.write_csv(
        "../data/swc/dip_data_yeast.txt", has_header=False, separator="\t"
    )
    print(df_dip_reformat)

    print(f">>> [{TOPO}] Execution Time")
    print(time.time() - start_time)
