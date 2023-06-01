# pyright: basic

import os
import time
from typing import Dict, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.preprocessing import KBinsDiscretizer

from aliases import (
    CO_EXP,
    CO_OCCUR,
    GO_BP,
    GO_CC,
    GO_MF,
    PROBA_CO_COMP,
    PROBA_NON_CO_COMP,
    PROTEIN,
    PROTEIN_U,
    PROTEIN_V,
    REL,
    STRING,
    TOPO,
    TOPO_L2,
    WEIGHT,
)
from assertions import assert_prots_sorted
from utils import read_no_header_file, sort_prot_cols
from weighting_model import WeightingModel

SCORE = WEIGHT


class MultinomialNBWeighting(WeightingModel):
    """
    Multinomial Naive Bayes ML model for co-complex membership weighting.

    1) Discretize the range of values of each feature. (?)
    2) Train the model to learn the maximum-likelihood parameters.
    3) Use the parameters to weight each PPI.
    """

    def __init__(self, features: List[str], ppin: str):
        super().__init__(features, ppin, MultinomialNB())

    def main(self, df_train_cmp_pairs: pl.DataFrame) -> Optional[pl.DataFrame]:
        """
        MultinomialNBWeighting main function.

        Args:
            df_train_cmp_pairs (pl.DataFrame): _description_

        Returns:
            Optional[pl.DataFrame]: _description_
        """

        df_composite_ppin = self.construct_composite_ppin()
        print(df_composite_ppin)

        if df_composite_ppin is None:
            return None

        df_labeled = self.equalize_classes(
            self.label_ppin(df_composite_ppin, df_train_cmp_pairs)
        )

        df_w_ppin = (
            self.weight(df_composite_ppin, df_labeled)
            .rename({PROBA_CO_COMP: SCORE})
            .select([PROTEIN_U, PROTEIN_V, SCORE])
        )

        return df_w_ppin


if __name__ == "__main__":
    start_time = time.time()
    pl.enable_string_cache(True)

    df_train_cmp_pairs = read_no_header_file(
        "../data/preprocessed/train_cmp_pairs.csv", [PROTEIN_U, PROTEIN_V]
    )

    features = [
        REL,
        CO_EXP,
        GO_CC,
        GO_BP,
        GO_MF,
        TOPO,
        TOPO_L2,
        STRING,
        CO_OCCUR,
    ]

    ppin = "swc"

    weighting = MultinomialNBWeighting(features, ppin)
    df_w_ppin = weighting.main(df_train_cmp_pairs)

    if df_w_ppin is not None:
        df_w_ppin_20k = df_w_ppin.sort(pl.col(SCORE), descending=True).head(20000)

        print(">>> WEIGHTED NETWORK [ALL]")
        print(df_w_ppin)

        print(">>> WEIGHTED NETWORK [20K edges]")
        print(df_w_ppin_20k)

        df_w_ppin.write_csv(
            f"../data/weighted/{ppin}_weighted_mnb.csv",
            has_header=False,
            separator="\t",
        )
        df_w_ppin_20k.write_csv(
            f"../data/weighted/{ppin}_weighted_mnb_20k.csv",
            has_header=False,
            separator="\t",
        )

    print()
    print(">>> Execution Time")
    print(f"{time.time() - start_time} seconds")
