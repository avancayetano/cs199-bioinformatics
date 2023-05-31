# pyright: basic

import os
import time
from typing import Dict, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import KBinsDiscretizer

from aliases import (
    CO_EXP,
    CO_OCCUR,
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
from utils import read_no_header_file, sort_prot_cols
from weighting_model import WeightingModel

PROB_CO_COMP = "P(co-comp|F=f)"
PROB_NON_CO_COMP = "P(non-co-comp|F=f)"
SCORE = "WEIGHT"


class MLPWeighting(WeightingModel):
    """
    Gaussian Naive Bayes ML model for co-complex membership weighting.

    1) Train the model to learn the maximum-likelihood parameters.
    2) Use the parameters to weight each PPI.
    """

    def __init__(self, features: List[str], ppin: str):
        super().__init__(
            features,
            ppin,
            MLPClassifier(hidden_layer_sizes=(200, 200, 200), max_iter=3000),
        )

    def main(self, df_train_cmp_pairs: pl.DataFrame) -> Optional[pl.DataFrame]:
        """
        Main function.

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
            .rename({PROB_CO_COMP: SCORE})
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

    weighting = MLPWeighting(features, ppin)
    df_w_ppin = weighting.main(df_train_cmp_pairs)

    if df_w_ppin is not None:
        df_w_ppin_20k = df_w_ppin.sort(pl.col(SCORE), descending=True).head(20000)

        print(">>> WEIGHTED NETWORK [ALL]")
        print(df_w_ppin)

        print(">>> WEIGHTED NETWORK [20K edges]")
        print(df_w_ppin_20k)

        df_w_ppin.write_csv(
            f"../data/weighted/{ppin}_weighted_mlp.csv",
            has_header=False,
            separator="\t",
        )
        df_w_ppin_20k.write_csv(
            f"../data/weighted/{ppin}_weighted_mlp_20k.csv",
            has_header=False,
            separator="\t",
        )

    print()
    print(">>> Execution Time")
    print(f"{time.time() - start_time} seconds")
