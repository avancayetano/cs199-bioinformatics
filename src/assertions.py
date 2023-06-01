import polars as pl

from aliases import PROTEIN_U, PROTEIN_V


def assert_prots_sorted(df: pl.DataFrame):
    """
    One of my major assumptions when working with PPIN is that
    the columns of proteins in a PPIN are sorted.
    This function checks if this is satisfied.

    Args:
        df (pl.DataFrame): _description_
    """
    assert df.filter(pl.col(PROTEIN_U) > pl.col(PROTEIN_V)).shape[0] == 0


def assert_df_normalized(df: pl.DataFrame, col: str):
    """
    Checks if a dataframe column is normalized (between 0 and 1).

    Args:
        df (pl.DataFrame): _description_
        col (str): _description_
    """

    assert df.filter((pl.col(col) > 1.0) | (pl.col(col) < 0.0)).shape[0] == 0
