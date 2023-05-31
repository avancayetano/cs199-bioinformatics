from typing import List, Tuple

import polars as pl

from aliases import (
    CO_EXP,
    CO_OCCUR,
    COMPLEX_DESCRIPTION,
    COMPLEX_ID,
    COMPLEX_PROTEINS,
    GO_BP,
    GO_CC,
    GO_MF,
    PROTEIN,
    PROTEIN_U,
    PROTEIN_V,
    REL,
    STRING,
    TOPO,
    TOPO_L2,
)
from assertions import assert_prots_sorted


def read_no_header_file(path: str, cols: List[str]) -> pl.DataFrame:
    df = pl.read_csv(path, has_header=False).rename(
        {f"column_{idx+1}": col for idx, col in enumerate(cols)}
    )

    return df


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


def construct_composite_ppin(
    features: List[str] = [
        TOPO,
        TOPO_L2,
        STRING,
        CO_OCCUR,
        REL,
        CO_EXP,
        GO_CC,
        GO_BP,
        GO_MF,
    ]
) -> pl.DataFrame:
    """
    Constructs the composite PPIN based on the selected features.
    By default, gets all the features.

    Returns:
        pl.DataFrame: _description_
    """

    swc_features = [TOPO, TOPO_L2, STRING, CO_OCCUR]

    df_swc_composite = pl.read_csv("../data/preprocessed/swc_composite_data.csv")

    if len(features) == 0:
        assert_prots_sorted(df_swc_composite)
        return df_swc_composite.select([PROTEIN_U, PROTEIN_V])

    scores_files = {
        REL: f"../data/scores/swc_rel.csv",
        CO_EXP: f"../data/scores/swc_co_exp.csv",
    }

    go_features: List[str] = []
    lf_features: List[pl.LazyFrame] = []
    for F in features:
        if F in [GO_CC, GO_BP, GO_MF]:
            go_features.append(F)
        elif F in swc_features:
            lf_features.append(
                df_swc_composite.select([PROTEIN_U, PROTEIN_V, F]).lazy()
            )
        else:
            lf_features.append(
                pl.scan_csv(scores_files[F], has_header=True)
                .with_columns(sort_prot_cols(PROTEIN_U, PROTEIN_V))
                .select([PROTEIN_U, PROTEIN_V, F])
            )

    if len(go_features) > 0:
        lf_features.append(
            pl.scan_csv(
                f"../data/scores/swc_go_ss.csv",
                has_header=True,
                null_values="None",
            )
            .with_columns(sort_prot_cols(PROTEIN_U, PROTEIN_V))
            .select([PROTEIN_U, PROTEIN_V, *go_features])
        )

    lf_composite_ppin = lf_features[0]

    for lf in lf_features[1:]:
        lf_composite_ppin = lf_composite_ppin.join(
            other=lf, on=[PROTEIN_U, PROTEIN_V], how="outer"
        )

    df_composite_ppin = lf_composite_ppin.fill_null(0.0).collect()
    assert_prots_sorted(df_composite_ppin)
    return df_composite_ppin


def get_cyc_complexes() -> pl.DataFrame:
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


def get_cyc_proteins() -> pl.Series:
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


def get_all_cyc_complex_pairs() -> pl.DataFrame:
    complexes: List[List[str]] = (
        get_cyc_complexes().select(COMPLEX_PROTEINS).to_series().to_list()
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

    # Why do column names start with column_0 and not column_1 ???
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


def get_xval_complex_pairs(xval_iter: int) -> pl.DataFrame:
    """
    Gets all the co-complex pairs from the training/testing protein complexes.

    TODO: Redo this.
    """

    pass
    return pl.DataFrame()

    # co_complex_pairs: List[Tuple[str, str]] = []

    # for cmp in complexes:
    #     complex = list(cmp)
    #     pairs = [
    #         (prot_i, prot_j)
    #         for i, prot_i in enumerate(complex[:-1])
    #         for prot_j in complex[i + 1 :]
    #     ]
    #     co_complex_pairs.extend(pairs)

    # # Why do column names start with column_0 and not column_1 ???
    # df_cmp_pairs = (
    #     pl.LazyFrame(co_complex_pairs, orient="row")
    #     .rename({"column_0": "u", "column_1": "v"})
    #     .with_columns(sort_prot_cols("u", "v"))
    #     .select([PROTEIN_U, PROTEIN_V])
    #     .unique()
    #     .collect()
    # )
    # assert_prots_sorted(df_cmp_pairs)

    # return df_cmp_pairs
