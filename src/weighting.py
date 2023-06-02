# pyright: basic

from typing import List, Union

import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.neural_network import MLPClassifier

from aliases import (
    FEATURES,
    IS_CO_COMP,
    PROBA_CO_COMP,
    PROBA_NON_CO_COMP,
    PROTEIN_U,
    PROTEIN_V,
    SUPER_FEATS,
    WEIGHT,
    WeightingModel,
)
from assertions import assert_df_normalized
from co_comp_classifier import CoCompClassifier
from model_preprocessor import ModelPreprocessor
from utils import construct_composite_network, get_cyc_train_test_comp_pairs


class FeatureWeighting:
    def main(
        self, df_composite: pl.DataFrame, features: List[str], name: str
    ) -> pl.DataFrame:
        df_w_composite = (
            df_composite.lazy()
            .select([PROTEIN_U, PROTEIN_V, *features])
            .with_columns((pl.sum(features) / len(features)).alias(WEIGHT))
            .select([PROTEIN_U, PROTEIN_V, WEIGHT])
            .filter(pl.col(WEIGHT) > 0)
            .sort(pl.col(WEIGHT), descending=True)
            .collect()
        )

        df_w_composite.write_csv(
            f"../data/weighted/all_edges/features/{name.lower()}.csv",
            has_header=False,
            separator="\t",
        )
        df_w_composite.head(20_000).write_csv(
            f"../data/weighted/20k_edges/features/{name.lower()}_20k.csv",
            has_header=False,
            separator="\t",
        )

        return df_w_composite


if __name__ == "__main__":
    # pl.Config.set_tbl_cols(40)
    pl.Config.set_tbl_rows(20)

    df_composite = construct_composite_network()

    print("------------- BEGIN: FEATURE WEIGHTING ----------------------")
    feat_weighting = FeatureWeighting()
    for f in FEATURES:
        df_f_weighted = feat_weighting.main(df_composite, [f], f)
        print(f"FEATURE: {f}")
        print(df_f_weighted)
        assert_df_normalized(df_f_weighted, WEIGHT)
    print("------------- END: FEATURE WEIGHTING ----------------------\n\n")

    print("------------- BEGIN: SUPER FEATURE WEIGHTING ----------------------")
    for f in SUPER_FEATS:
        df_f_weighted = feat_weighting.main(df_composite, f["features"], f["name"])
        print(f"FEATURE: {f}")
        print(df_f_weighted)
        assert_df_normalized(df_f_weighted, WEIGHT)
    print("------------- END: SUPER FEATURE WEIGHTING ----------------------\n\n")

    print("------------- BEGIN: SUPERVISED WEIGHTING ----------------------")
    n_iters = 10
    print(f"CROSS-VALIDATION ITERATIONS: {n_iters}")

    model_prep = ModelPreprocessor()
    df_composite = model_prep.normalize_features(df_composite, FEATURES)

    # Weighting Models
    rf = CoCompClassifier(FEATURES, RandomForestClassifier(), "rf")
    cnb = CoCompClassifier(FEATURES, CategoricalNB(), "cnb")

    for xval_iter in range(n_iters):
        print(f"------------------- BEGIN: ITER {xval_iter} ---------------------\n")
        df_train_pairs, df_test_pairs = get_cyc_train_test_comp_pairs(xval_iter)
        df_labeled = model_prep.label_composite(
            df_composite, df_train_pairs, IS_CO_COMP, xval_iter
        )
        df_composite_binned = model_prep.discretize_composite(
            df_composite, df_labeled, FEATURES, IS_CO_COMP, xval_iter
        )

        df_w_rf = rf.main(
            df_composite, df_labeled, df_train_pairs, df_test_pairs, xval_iter
        )
        df_w_cnb = cnb.main(
            df_composite_binned, df_labeled, df_train_pairs, df_test_pairs, xval_iter
        )

        df_w_rf.write_csv(
            f"../data/weighted/all_edges/cross_val/{rf.name}_iter{xval_iter}.csv",
            has_header=False,
            separator="\t",
        )
        df_w_cnb.write_csv(
            f"../data/weighted/all_edges/cross_val/{cnb.name}_iter{xval_iter}.csv",
            has_header=False,
            separator="\t",
        )

        print(f"------------------- END: ITER {xval_iter} ---------------------")
