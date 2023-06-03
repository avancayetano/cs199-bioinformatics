from typing import List, Optional, Set, Tuple

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
    Sort the two protein columns such that the first protein
    is lexicographically less than the second.

    The aliases of the two sorted columns is PROTEIN_U and
    PROTEIN_V, respectively.
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


def construct_composite_network(features: Optional[List[str]] = None) -> pl.DataFrame:
    """
    Constructs the composite protein network based on the selected features.
    By default, gets all the features.

    Returns:
        pl.DataFrame: _description_
    """

    scores_files = [
        "co_exp_scores.csv",
        "go_ss_scores.csv",
        "rel_scores.csv",
        "swc_composite_scores.csv",
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
    clusters: List[Set[str]] = []
    with open(path) as file:
        lines = file.readlines()
        for line in lines:
            proteins = set(line.split("\t"))
            clusters.append(proteins)

    return clusters


def get_complexes_list() -> List[Set[str]]:
    df_complexes = get_all_cyc_complexes()
    cmps: List[List[str]] = df_complexes.select(COMP_PROTEINS).to_series().to_list()
    complexes: List[Set[str]] = list(map(lambda cmp: set(cmp), cmps))

    return complexes


def get_all_cyc_complexes() -> pl.DataFrame:
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


def get_cyc_comp_pairs(df_complex_ids: Optional[pl.DataFrame] = None):
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


def get_cyc_train_test_comp_pairs(iter: int) -> Tuple[pl.DataFrame, pl.DataFrame]:
    df_cross_val = pl.read_csv("../data/preprocessed/cross_val_table.csv")

    df_train_ids = df_cross_val.filter(pl.col(f"{XVAL_ITER}_{iter}") == "train").select(
        COMP_ID
    )

    df_test_ids = df_cross_val.filter(pl.col(f"{XVAL_ITER}_{iter}") == "test").select(
        COMP_ID
    )

    print(
        f"Train complexes: {df_train_ids.shape[0]} | Test complexes: {df_test_ids.shape[0]}"
    )

    df_train = get_cyc_comp_pairs(df_train_ids)
    df_test = get_cyc_comp_pairs(df_test_ids)

    return df_train, df_test
