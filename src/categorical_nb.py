# pyright: basic

import os
import time
from typing import Dict, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import KBinsDiscretizer

from aliases import (
    CO_EXP,
    CO_OCCUR,
    GO_BP,
    GO_CC,
    GO_MF,
    PROB_CO_COMP,
    PROB_NON_CO_COMP,
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
LABELS = {True: "co-comp", False: "non-co-comp"}


class CategoricalNBWeighting(WeightingModel):
    """
    Categorical Naive Bayes ML model for co-complex membership weighting.

    1) Discretize data
    2) Train the model to learn the maximum-likelihood parameters.
    3) Use the parameters to weight each PPI.
    """

    def __init__(self, features: List[str], ppin: str):
        super().__init__(features, ppin, CategoricalNB())

    def discretize(
        self, df_composite_ppin: pl.DataFrame, n_bins: int
    ) -> Tuple[pl.DataFrame, np.ndarray, np.ndarray]:
        """
        _summary_

        Args:
            df_composite_ppin (pl.DataFrame): _description_
            n_bins (int): _description_

        Returns:
            Tuple[pl.DataFrame, np.ndarray, np.ndarray]: _description_
        """

        ndarr_scores = df_composite_ppin.select(
            pl.exclude([PROTEIN_U, PROTEIN_V])
        ).to_numpy()

        discretizer = KBinsDiscretizer(
            n_bins=n_bins, encode="ordinal", strategy="uniform"
        )
        ndarr_scores_binned = discretizer.fit_transform(ndarr_scores)

        edges = discretizer.bin_edges_
        bins = discretizer.n_bins_

        df_scores_binned = pl.from_numpy(
            ndarr_scores_binned, schema=self.features_
        ).with_columns(pl.col(F) for F in self.features_)

        df_ppin_binned = pl.concat(
            [
                df_composite_ppin.select([PROTEIN_U, PROTEIN_V]),
                df_scores_binned,
            ],
            how="horizontal",
        )
        return df_ppin_binned, edges, bins

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

        n_bins = 20

        df_binned, edges, bins = self.discretize(df_composite_ppin, n_bins)
        self.model_.set_params(min_categories=bins)

        df_labeled = self.equalize_classes(
            self.label_ppin(df_binned, df_train_cmp_pairs)
        )

        df_w_ppin = (
            self.weight(df_binned, df_labeled)
            .rename({PROB_CO_COMP: SCORE})
            .select([PROTEIN_U, PROTEIN_V, SCORE])
        )

        if type(self.model_) == CategoricalNB:
            arr = np.hstack(
                [
                    (
                        np.hstack(
                            (
                                np.array(feature)[0].reshape((n_bins, -1)),
                                np.array(feature)[1].reshape((n_bins, -1)),
                            )
                        )
                    )
                    for feature in self.model_.category_count_
                ]
            )

            column_names = [
                f"P({F}=f|{LABELS[class_]})"
                for F in self.features_
                for class_ in self.model_.classes_
            ]

            df_params = (
                pl.from_numpy(arr)
                .rename(
                    {
                        f"column_{i}": column_names[i]
                        for i in range(len(self.model_.classes_) * len(self.features_))
                    }
                )
                .with_columns(
                    [
                        (pl.col(f"P({F}=f|{LABELS[class_]})") + 1.0)
                        / (pl.lit(self.model_.class_count_[idx]) + n_bins)
                        for F in self.features_
                        for idx, class_ in enumerate(self.model_.classes_)
                    ]
                )
                .with_columns(
                    [
                        (
                            pl.col(f"P({F}=f|co-comp)")
                            / pl.col(f"P({F}=f|non-co-comp)")
                        ).alias(f"{F} likelihood")
                        for F in self.features_
                    ]
                )
                .with_row_count()
                .rename({"row_nr": "CATEGORY"})
            )

            df_params.write_csv(f"../data/training/{self.ppin_}_cnb_params.csv")

            print(df_params)

        return df_w_ppin


if __name__ == "__main__":
    start_time = time.time()
    pl.enable_string_cache(True)

    df_train_cmp_pairs = read_no_header_file(
        "../data/preprocessed/all_cmp_pairs.csv", [PROTEIN_U, PROTEIN_V]
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

    weighting = CategoricalNBWeighting(features, ppin)
    df_w_ppin = weighting.main(df_train_cmp_pairs)

    if df_w_ppin is not None:
        df_w_ppin_20k = df_w_ppin.sort(pl.col(SCORE), descending=True).head(20000)

        print(">>> WEIGHTED NETWORK [ALL]")
        print(df_w_ppin)

        print(">>> WEIGHTED NETWORK [20K edges]")
        print(df_w_ppin_20k)

        df_w_ppin.write_csv(
            f"../data/weighted/{ppin}_weighted_cnb_all_train_swc.csv",
            has_header=False,
            separator="\t",
        )
        df_w_ppin_20k.write_csv(
            f"../data/weighted/{ppin}_weighted_cnb_all_train_swc_20k.csv",
            has_header=False,
            separator="\t",
        )

    print()
    print(">>> Execution Time")
    print(f"{time.time() - start_time} seconds")
