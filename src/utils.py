from typing import List, Optional, Tuple

import polars as pl

from aliases import (
    COMPLEX_DESCRIPTION,
    COMPLEX_ID,
    COMPLEX_PROTEINS,
    CROSS_VAL_ITER,
    PROTEIN,
    PROTEIN_U,
    PROTEIN_V,
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


def get_all_cyc_complexes() -> pl.DataFrame:
    df_complexes = (
        pl.scan_csv("../data/swc/complexes_CYC.txt", has_header=False, separator="\t")
        .rename(
            {
                "column_1": PROTEIN,
                "column_2": COMPLEX_ID,
                "column_3": COMPLEX_DESCRIPTION,
            }
        )
        .groupby(pl.col(COMPLEX_ID, COMPLEX_DESCRIPTION))
        .agg(pl.col(PROTEIN).alias(COMPLEX_PROTEINS))
        .sort(pl.col(COMPLEX_ID))
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
                "column_2": COMPLEX_ID,
                "column_3": COMPLEX_DESCRIPTION,
            }
        )
        .select(PROTEIN)
        .unique()
        .collect()
        .to_series()
    )

    return srs_proteins


def get_cyc_complex_pairs(df_complex_ids: Optional[pl.DataFrame] = None):
    df_all_complexes = get_all_cyc_complexes()
    if df_complex_ids is None:
        complexes: List[List[str]] = (
            df_all_complexes.select(COMPLEX_PROTEINS).to_series().to_list()
        )
    else:
        complexes: List[List[str]] = (
            df_complex_ids.join(df_all_complexes, on=COMPLEX_ID, how="left")
            .select(COMPLEX_PROTEINS)
            .to_series()
            .to_list()
        )

    co_complex_pairs: List[Tuple[str, str]] = []
    for cmp in complexes:
        complex = list(cmp)
        pairs = [
            (prot_i, prot_j)
            for i, prot_i in enumerate(complex[:-1])
            for prot_j in complex[i + 1 :]
        ]
        co_complex_pairs.extend(pairs)

    df_cmp_pairs = (
        pl.LazyFrame(co_complex_pairs, orient="row")
        .rename({"column_0": "u", "column_1": "v"})
        .with_columns(sort_prot_cols("u", "v"))
        .select([PROTEIN_U, PROTEIN_V])
        .unique()
        .collect()
    )
    assert_prots_sorted(df_cmp_pairs)

    return df_cmp_pairs


def get_cyc_train_test_cmp_pairs(iter: int) -> Tuple[pl.DataFrame, pl.DataFrame]:
    df_cross_val = pl.read_csv("../data/preprocessed/cross_val_table.csv")

    df_train_ids = df_cross_val.filter(
        pl.col(f"{CROSS_VAL_ITER}_{iter}") == "train"
    ).select(COMPLEX_ID)

    df_test_ids = df_cross_val.filter(
        pl.col(f"{CROSS_VAL_ITER}_{iter}") == "test"
    ).select(COMPLEX_ID)

    df_train = get_cyc_complex_pairs(df_train_ids)
    df_test = get_cyc_complex_pairs(df_test_ids)

    return df_train, df_test
