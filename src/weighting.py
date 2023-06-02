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


class RFWeighting(CoCompClassifier):
    def __init__(
        self,
        features: List[str],
        model: RandomForestClassifier,
        name: str,
    ):
        super().__init__(features, model, name)

    def main(
        self, df_composite: pl.DataFrame, features: List[str], name: str
    ) -> pl.DataFrame:
        df_train_pairs, df_test_pairs = get_cyc_train_test_comp_pairs(i)

        df_labeled = rf.label_composite(df_composite, df_train_pairs)
        df_labeled = rf.equalize_classes(df_labeled, i)

        df_w_composite = rf.weight(df_composite, df_labeled)
        rf.validate(df_w_composite, df_train_pairs, df_test_pairs)

        df_check = df_labeled.join(
            df_w_composite, on=[PROTEIN_U, PROTEIN_V], how="left"
        ).select([PROTEIN_U, PROTEIN_V, PROBA_NON_CO_COMP, PROBA_CO_COMP, IS_CO_COMP])

        print(df_check)

        df_w_composite = df_w_composite.rename({PROBA_CO_COMP: WEIGHT}).select(
            [PROTEIN_U, PROTEIN_V, WEIGHT]
        )

        df_w_composite.write_csv(
            f"../data/weighted/all_edges/cross_val/{rf.name}_iter{i}.csv",
            has_header=False,
            separator="\t",
        )


if __name__ == "__main__":
    pl.Config.set_tbl_cols(40)
    pl.Config.set_tbl_rows(7)

    df_composite = construct_composite_network()

    print("------------- BEGIN: FEATURE WEIGHTING ----------------------")
    feat_weighting = FeatureWeighting()
    for f in FEATURES:
        df_f_weighted = feat_weighting.main(df_composite, [f], f)
        print(f"FEATURE: {f}")
        print(df_f_weighted)
        assert_df_normalized(df_f_weighted, WEIGHT)
    print("------------- END: FEATURE WEIGHTING ----------------------")

    print("------------- BEGIN: SUPER FEATURE WEIGHTING ----------------------")
    for f in SUPER_FEATS:
        df_f_weighted = feat_weighting.main(df_composite, f["features"], f["name"])
        print(f"FEATURE: {f}")
        print(df_f_weighted)
        assert_df_normalized(df_f_weighted, WEIGHT)
    print("------------- END: SUPER FEATURE WEIGHTING ----------------------")

    print("------------- BEGIN: SUPERVISED WEIGHTING ----------------------")
    iters = 10
    print(f"CROSS-VALIDATION ITERATIONS: {iters}")

    # feature_prep = ModelPreprocessor()
    # rf = CoCompClassifier(features, RandomForestClassifier(), "rf")
    # cnb = CoCompClassifier(features, CategoricalNB(), "cnb")

    # df_composite = feature_prep.normalize_features(df_composite, features)

    # for i in range(iters):
    #     print(f"------------------- BEGIN: ITER {i} ---------------------")
    #     df_train_pairs, df_test_pairs = get_cyc_train_test_comp_pairs(i)

    #     df_labeled = rf.label_composite(df_composite, df_train_pairs)
    #     df_labeled = rf.equalize_classes(df_labeled, i)

    #     df_w_composite = rf.weight(df_composite, df_labeled)
    #     rf.validate(df_w_composite, df_train_pairs, df_test_pairs)

    #     df_check = df_labeled.join(
    #         df_w_composite, on=[PROTEIN_U, PROTEIN_V], how="left"
    #     ).select([PROTEIN_U, PROTEIN_V, PROBA_NON_CO_COMP, PROBA_CO_COMP, IS_CO_COMP])

    #     print(df_check)

    #     df_w_composite = df_w_composite.rename({PROBA_CO_COMP: WEIGHT}).select(
    #         [PROTEIN_U, PROTEIN_V, WEIGHT]
    #     )

    #     df_w_composite.write_csv(
    #         f"../data/weighted/all_edges/cross_val/{rf.name}_iter{i}.csv",
    #         has_header=False,
    #         separator="\t",
    #     )

    #     print(f"------------------- END: ITER {i} ---------------------")
