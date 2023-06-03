from typing import List, Literal, Union

import polars as pl

from aliases import PROTEIN_U, PROTEIN_V


def assert_prots_sorted(df: pl.DataFrame):
    """
    One of my major assumptions when working with PPIN is that
    the columns of proteins in the network are sorted.
    This function checks if this is satisfied.

    Args:
        df (pl.DataFrame): _description_
    """
    assert df.filter(pl.col(PROTEIN_U) > pl.col(PROTEIN_V)).shape[0] == 0


def assert_df_normalized(df: pl.DataFrame, cols: List[str]):
    """
    Checks if a dataframe column is normalized (between 0 and 1).

    Args:
        df (pl.DataFrame): _description_
        col (str): _description_
    """
    for col in cols:
        assert df.filter((pl.col(col) > 1.0) | (pl.col(col) < 0.0)).shape[0] == 0


def assert_no_null(df: pl.DataFrame, cols: Union[List[str], Literal["*"]] = "*"):
    null_count = (
        df.select(pl.col(cols))
        .null_count()
        .with_columns(pl.sum(pl.col("*")).alias("NULL_COUNT"))
        .select("NULL_COUNT")
        .item()
    )
    assert null_count == 0
