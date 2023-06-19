from typing import List, Literal, Union

import polars as pl

from aliases import PROTEIN_U, PROTEIN_V, WEIGHT


def assert_prots_sorted(df: pl.DataFrame):
    """
    One of my major assumptions when working with PPIN is that
    the columns of proteins in the network are sorted.
    This function checks if this is satisfied.

    Args:
        df (pl.DataFrame): _description_
    """
    assert df.filter(pl.col(PROTEIN_U) > pl.col(PROTEIN_V)).shape[0] == 0


def assert_df_bounded(df: pl.DataFrame, cols: List[str]):
    """
    Checks if a dataframe column is within the range [0, 1].

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


def assert_no_zero_weight(df: pl.DataFrame):
    assert df.filter(pl.col(WEIGHT) == 0).shape[0] == 0


def assert_same_edges(df1: pl.DataFrame, df2: pl.DataFrame):
    df1 = df1.select([PROTEIN_U, PROTEIN_V]).unique()
    df2 = df2.select([PROTEIN_U, PROTEIN_V]).unique()
    df = df1.join(df2, on=[PROTEIN_U, PROTEIN_V], how="inner")
    assert df.shape[0] == df1.shape[0] == df2.shape[0]


def assert_gene_exp_arranged(df_melted: pl.DataFrame):
    df = df_melted.filter(
        pl.col("T_PROTEIN_U").str.extract(r"T(\d+).+")
        != pl.col("T_PROTEIN_V").str.extract(r"T(\d+).+")
    )
    assert df.shape[0] == 0
