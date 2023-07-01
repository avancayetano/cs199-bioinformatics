# This module contains utility functions.

from typing import List, Literal, Optional, Set, Tuple

import polars as pl

from aliases import (
    COMP_ID,
    COMP_INFO,
    COMP_PROTEINS,
    PROTEIN,
    PROTEIN_U,
    PROTEIN_V,
    XVAL_ITER,
)
from assertions import assert_prots_sorted


def sort_prot_cols(prot_u: str, prot_v: str) -> List[pl.Expr]:
    """
    Sorts the two protein columns such that the first protein
    is lexicographically less than the second.

    The aliases of the two sorted columns is PROTEIN_U and
    PROTEIN_V, respectively.

    Args:
        prot_u (str): Label for the first protein.
        prot_v (str): Label for the second protein.

    Returns:
        List[pl.Expr]: List of Expr that does the above sorting.
    """
    exp = [
        pl.when(pl.col(prot_u) < pl.col(prot_v))
        .then(pl.col(prot_u))
        .otherwise(pl.col(prot_v))
        .alias(PROTEIN_U),
        pl.when(pl.col(prot_u) < pl.col(prot_v))
        .then(pl.col(prot_v))
        .otherwise(pl.col(prot_u))
        .alias(PROTEIN_V),
    ]
    return exp


def construct_composite_network(
    dip: bool,
    features: Optional[List[str]] = None,
) -> pl.DataFrame:
    """
    Constructs the composite protein network based on the selected features.
    By default, gets all the features.

    Args:
        dip (bool): Whether to get the DIP or the original composite network.
        features (Optional[List[str]], optional): List of features to retrieve. Defaults to None.

    Returns:
        pl.DataFrame: Composite network.
    """
    prefix = "dip_" if dip else ""
    scores_files = [
        f"{prefix}co_exp_scores.csv",
        f"{prefix}go_ss_scores.csv",
        f"{prefix}rel_scores.csv",
        f"{prefix}swc_composite_scores.csv",
    ]

    lf_composite = pl.LazyFrame()
    for file in scores_files:
        lf_score = pl.scan_csv(
            f"../data/scores/{file}", null_values="None"
        ).with_columns(sort_prot_cols(PROTEIN_U, PROTEIN_V))

        if lf_composite.collect().is_empty():
            lf_composite = lf_score
        else:
            lf_composite = lf_composite.join(
                lf_score, on=[PROTEIN_U, PROTEIN_V], how="outer"
            )

    if features is None:
        df_composite = lf_composite.fill_null(0.0).collect()
    else:
        df_composite = (
            lf_composite.fill_null(0.0)
            .select([PROTEIN_U, PROTEIN_V, *features])
            .collect()
        )

    assert_prots_sorted(df_composite)
    return df_composite


def get_clusters_list(path: str) -> List[Set[str]]:
    """
    Gets the list of clusters.

    Args:
        path (str): Path to clusters file.

    Returns:
        List[Set[str]]: List of clusters.
    """
    clus: List[Set[str]] = []
    with open(path) as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            proteins = set(line.split("\t"))
            clus.append(proteins)

    clusters: List[Set[str]] = list(filter(lambda c: len(c) >= 4, clus))

    return clusters


def get_complexes_list(
    xval_iter: int, complex_type: Literal["train", "test"], dip: bool
) -> List[Set[str]]:
    """
    Gets the list of complexes based on cross-val iteration and complex type.

    Args:
        xval_iter (int): Cross-val iteration.
        complex_type (Literal['train', 'test']): Either train or test.
        dip (bool): Whether to retrieve complexes using DIP cross_val_table or not.

    Returns:
        List[Set[str]]: List of complexes.
    """
    df_complexes = get_all_cyc_complexes()

    if dip:
        df_cross_val = pl.read_csv("../data/preprocessed/dip_cross_val_table.csv")
        print("Using: dip_cross_val_table.csv")
    else:
        df_cross_val = pl.read_csv("../data/preprocessed/cross_val_table.csv")
        print("Using: cross_val_table.csv")
    df_complex_ids = df_cross_val.filter(
        pl.col(f"{XVAL_ITER}_{xval_iter}") == complex_type
    ).select(COMP_ID)
    cmps: List[List[str]] = (
        df_complexes.join(df_complex_ids, on=COMP_ID, how="inner")
        .select(COMP_PROTEINS)
        .to_series()
        .to_list()
    )

    complexes: List[Set[str]] = list(map(lambda c: set(c), cmps))
    return complexes


def get_all_cyc_complexes() -> pl.DataFrame:
    """
    Gets all CYC2008 complexes.

    Returns:
        pl.DataFrame: CYC2008 complexes dataframe.
    """
    df_complexes = (
        pl.scan_csv("../data/swc/complexes_CYC.txt", has_header=False, separator="\t")
        .rename(
            {
                "column_1": PROTEIN,
                "column_2": COMP_ID,
                "column_3": COMP_INFO,
            }
        )
        .groupby(pl.col(COMP_ID, COMP_INFO))
        .agg(pl.col(PROTEIN).alias(COMP_PROTEINS))
        .sort(pl.col(COMP_ID))
        .collect()
    )
    return df_complexes


def get_unique_proteins(df: pl.DataFrame) -> pl.Series:
    """
    Gets all the unique proteins in a dataframe.

    Args:
        df (pl.DataFrame): Any dataframe with PROTEIN_U and PROTEIN_V columns.

    Returns:
        pl.Series: Series of unique proteins.
    """
    srs_proteins = (
        df.lazy()
        .select([PROTEIN_U, PROTEIN_V])
        .melt(variable_name="PROTEIN_X", value_name=PROTEIN)
        .select(PROTEIN)
        .unique()
        .collect()
        .to_series()
    )

    return srs_proteins


def get_all_cyc_proteins() -> pl.Series:
    """
    Gets all the unique CYC2008 proteins.

    Returns:
        pl.Series: Series of all the unique CYC2008 proteins.
    """
    srs_proteins = (
        pl.scan_csv("../data/swc/complexes_CYC.txt", has_header=False, separator="\t")
        .rename(
            {
                "column_1": PROTEIN,
                "column_2": COMP_ID,
                "column_3": COMP_INFO,
            }
        )
        .select(PROTEIN)
        .unique()
        .collect()
        .to_series()
    )

    return srs_proteins


def get_cyc_comp_pairs(df_complex_ids: Optional[pl.DataFrame] = None) -> pl.DataFrame:
    """
    Gets all CYC2008 co-complex pairs from an input of complex ID list.
    If None, gets all co-complex pairs.

    Args:
        df_complex_ids (Optional[pl.DataFrame], optional): List of complex IDs. Defaults to None.

    Returns:
        pl.DataFrame: Dataframe with two columns: PROTEIN_U and PROTEIN_V, where each row
            is a co-complex pair.
    """

    df_all_complexes = get_all_cyc_complexes()
    if df_complex_ids is None:
        complexes: List[List[str]] = (
            df_all_complexes.select(COMP_PROTEINS).to_series().to_list()
        )
    else:
        complexes: List[List[str]] = (
            df_complex_ids.join(df_all_complexes, on=COMP_ID, how="left")
            .select(COMP_PROTEINS)
            .to_series()
            .to_list()
        )

    co_comp_pairs: List[Tuple[str, str]] = []
    for cmp in complexes:
        complex = list(cmp)
        pairs = [
            (prot_i, prot_j)
            for i, prot_i in enumerate(complex[:-1])
            for prot_j in complex[i + 1 :]
        ]
        co_comp_pairs.extend(pairs)

    df_comp_pairs = (
        pl.LazyFrame(co_comp_pairs, orient="row")
        .rename({"column_0": "u", "column_1": "v"})
        .with_columns(sort_prot_cols("u", "v"))
        .select([PROTEIN_U, PROTEIN_V])
        .unique()
        .collect()
    )
    assert_prots_sorted(df_comp_pairs)

    return df_comp_pairs


def get_cyc_train_test_comp_pairs(
    xval_iter: int, dip: bool
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Gets the train, test co-complex pairs for a certain cross-val iteration.

    Args:
        xval_iter (int): Cross-val iteration.
        dip (bool): Whether to retrieve complexes using DIP cross_val_table or not.

    Returns:
        Tuple[pl.DataFrame, pl.DataFrame]: Tuple of train, test co-complex pairs.
    """
    if dip:
        df_cross_val = pl.read_csv("../data/preprocessed/dip_cross_val_table.csv")
        print("Using: dip_cross_val_table.csv")
    else:
        df_cross_val = pl.read_csv("../data/preprocessed/cross_val_table.csv")
        print("Using: cross_val_table.csv")
    df_train_ids = df_cross_val.filter(
        pl.col(f"{XVAL_ITER}_{xval_iter}") == "train"
    ).select(COMP_ID)

    df_test_ids = df_cross_val.filter(
        pl.col(f"{XVAL_ITER}_{xval_iter}") == "test"
    ).select(COMP_ID)

    print(
        f"Train complexes: {df_train_ids.shape[0]} | Test complexes: {df_test_ids.shape[0]}"
    )

    df_train = get_cyc_comp_pairs(df_train_ids)
    df_test = get_cyc_comp_pairs(df_test_ids)

    return df_train, df_test


def get_clusters_filename(
    n_edges: str,
    method: str,
    supervised: bool,
    inflation: int,
    dip: bool,
    xval_iter: int = -1,
) -> str:
    """
    Gets the clusters filename given the parameters.

    Args:
        n_edges (str): Either 20k_edges or all_edges.
        method (str): Weighting method.
        supervised (bool): Supervised method or not.
        inflation (int): MCL inflation parameter.
        dip (bool): Whether to retrieve DIP clusters or not.
        xval_iter (int, optional): Cross-val iteration. Defaults to -1.

    Returns:
        str: Path to clusters file.
    """
    prefix = "dip_" if dip else ""
    method = method.lower()
    if method == "unweighted":
        return f"../data/clusters/out.{prefix}{method}.csv.I{inflation}0"

    suffix = "_20k" if n_edges == "20k_edges" else ""
    if supervised:
        return f"../data/clusters/{n_edges}/cross_val/out.{prefix}{method}{suffix}_iter{xval_iter}.csv.I{inflation}0"
    return f"../data/clusters/{n_edges}/features/out.{prefix}{method}{suffix}.csv.I{inflation}0"


def get_weighted_filename(
    method: str, supervised: bool, dip: bool, xval_iter: int = -1
) -> str:
    """
    Gets the weighted network filename given the parameters.

    Args:
        method (str): Weighting method.
        supervised (bool): Supervised method or not.
        dip (bool): Whether to retrieve DIP weighted network or not.
        xval_iter (int, optional): Cross-val iteration. Defaults to -1.

    Returns:
        str: Path to weighted network file.
    """
    prefix = "dip_" if dip else ""
    method = method.lower()
    if method == "unweighted":
        return f"../data/weighted/{prefix}{method}.csv"

    if supervised:
        return (
            f"../data/weighted/all_edges/cross_val/{prefix}{method}_iter{xval_iter}.csv"
        )
    return f"../data/weighted/all_edges/features/{prefix}{method}.csv"
