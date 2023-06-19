"""
A script that weights the composite network using RFW and the features.
"""

import time
from typing import List

import polars as pl
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from xgboost import XGBClassifier, XGBRFClassifier

from aliases import (
    FEATURES,
    IS_CO_COMP,
    PROTEIN_U,
    PROTEIN_V,
    SUPER_FEATS,
    TOPO,
    WEIGHT,
)
from assertions import assert_df_bounded, assert_no_zero_weight
from model_preprocessor import ModelPreprocessor
from supervised_weighting import SupervisedWeighting
from utils import (
    construct_composite_network,
    get_cyc_comp_pairs,
    get_cyc_train_test_comp_pairs,
)


class FeatureWeighting:
    def main(
        self, df_composite: pl.DataFrame, features: List[str], name: str
    ) -> pl.DataFrame:
        df_w_composite = (
            df_composite.lazy()
            .select([PROTEIN_U, PROTEIN_V, *features])
            .with_columns((pl.sum(features) / len(features)).alias(WEIGHT))
            .select([PROTEIN_U, PROTEIN_V, WEIGHT])
            .collect()
        )

        df_w_composite.write_csv(
            f"../data/weighted/all_edges/features/{name.lower()}.csv",
            has_header=False,
            separator="\t",
        )

        df_w_20k = df_w_composite.sort(pl.col(WEIGHT), descending=True).head(20_000)
        assert_no_zero_weight(df_w_20k)
        df_w_20k.write_csv(
            f"../data/weighted/20k_edges/features/{name.lower()}_20k.csv",
            has_header=False,
            separator="\t",
        )

        return df_w_composite


if __name__ == "__main__":
    pl.Config.set_tbl_rows(15)

    start = time.time()

    model_prep = ModelPreprocessor()
    df_composite = construct_composite_network()
    df_composite = model_prep.normalize_features(df_composite, FEATURES)

    # print("---------------------------------------------------------")
    # print("Writing unweighted network")
    # df_unweighted = (
    #     df_composite.filter(pl.col(TOPO) > 0)
    #     .with_columns(pl.lit(1.0).alias(WEIGHT))
    #     .select([PROTEIN_U, PROTEIN_V, WEIGHT])
    # )

    # df_unweighted.write_csv(
    #     "../data/weighted/unweighted.csv", has_header=False, separator="\t"
    # )
    # print("Done writing unweighted network")
    # print()

    # print("------------- BEGIN: FEATURE WEIGHTING ----------------------")
    # feat_weighting = FeatureWeighting()
    # for f in FEATURES:
    #     df_f_weighted = feat_weighting.main(df_composite, [f], f)
    #     print(f"Done feature weighting using: {f}")
    #     assert_df_bounded(df_f_weighted, [WEIGHT])
    # print("------------- END: FEATURE WEIGHTING ----------------------\n\n")

    # print("------------- BEGIN: SUPER FEATURE WEIGHTING ----------------------")
    # for f in SUPER_FEATS:
    #     df_f_weighted = feat_weighting.main(df_composite, f["features"], f["name"])
    #     print(f"Done feature weighting using: {f['name']} - {f['features']}")
    #     assert_df_bounded(df_f_weighted, [WEIGHT])
    # print("------------- END: SUPER FEATURE WEIGHTING ----------------------\n\n")

    print("------------- BEGIN: SUPERVISED WEIGHTING ----------------------")
    n_iters = 10
    print(f"Cross-validation iterations: {n_iters}")
    print()

    # Supervised co-complex probability weighting
    rfw = SupervisedWeighting(
        RandomForestClassifier(
            n_estimators=3000,
            criterion="entropy",
            max_features="sqrt",
            n_jobs=-1,
        ),
        "RFW",
    )
    xgw_params = {
        "objective": "binary:logistic",
        "n_estimators": 10000,
        "alpha": 30,
        "max_depth": 3,
        "subsample": 0.5,
        "n_jobs": -1,
        "learning_rate": 0.01,
    }
    xgw = SupervisedWeighting(XGBClassifier(**xgw_params), "XGW")
    # gbw = SupervisedWeighting(
    #     HistGradientBoostingClassifier(
    #         max_iter=3000, l2_regularization=0.5, learning_rate=0.05
    #     ),
    #     "GBW",
    # )
    # xgrfw_params = {
    #     "n_estimators": 3000,
    # }

    # xgrfw = SupervisedWeighting(XGBRFClassifier(**xgrfw_params), "XGRFW")
    df_comp_pairs = get_cyc_comp_pairs()

    for xval_iter in range(n_iters):
        print(f"------------------- BEGIN: ITER {xval_iter} ---------------------")
        df_train_pairs, df_test_pairs = get_cyc_train_test_comp_pairs(xval_iter)
        df_train_labeled = model_prep.label_composite(
            df_composite, df_train_pairs, IS_CO_COMP, xval_iter, "subset", False
        )

        # SWC cross-validation scores
        df_w_swc = (
            pl.read_csv(
                f"../data/swc/raw_weighted/swc scored_edges iter{xval_iter}.txt",
                new_columns=[PROTEIN_U, PROTEIN_V, WEIGHT],
                separator=" ",
                has_header=False,
            )
            .join(df_composite, on=[PROTEIN_U, PROTEIN_V])
            .select([PROTEIN_U, PROTEIN_V, WEIGHT])
        )

        # Rewrite SWC scores to another file as a form of preprocessing as
        # it needs to be formatted before doing MCL
        df_w_swc.write_csv(
            f"../data/weighted/all_edges/cross_val/swc_iter{xval_iter}.csv",
            has_header=False,
            separator="\t",
        )

        # Write top 20k edges of SWC
        df_w_swc.sort(pl.col(WEIGHT), descending=True).head(20_000).write_csv(
            f"../data/weighted/20k_edges/cross_val/swc_20k_iter{xval_iter}.csv",
            has_header=False,
            separator="\t",
        )

        # Weight the network using RFW
        # df_w_rfw = rfw.main(df_composite, df_train_labeled, xval_iter)
        df_w_xgw = xgw.main(df_composite, df_train_labeled, xval_iter)
        # df_w_gbw = gbw.main(df_composite, df_train_labeled, xval_iter)
        # df_w_xgrfw = xgrfw.main(df_composite, df_train_labeled, xval_iter)

        print(f"------------------- END: ITER {xval_iter} ---------------------\n\n")

    print()
    print(f"All {n_iters} iterations done!")
    print(f"Execution time: {time.time() - start}")
