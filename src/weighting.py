"""
A script that weights the composite network using RFW and the features.
"""

import time
from typing import List

import polars as pl
from xgboost import XGBClassifier

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
from utils import construct_composite_network, get_cyc_train_test_comp_pairs


class FeatureWeighting:
    def __init__(self, dip: bool = False) -> None:
        self.prefix = "dip_" if dip else ""

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
            f"../data/weighted/all_edges/features/{self.prefix}{name.lower()}.csv",
            has_header=False,
            separator="\t",
        )

        df_w_20k = df_w_composite.sort(pl.col(WEIGHT), descending=True).head(20_000)
        assert_no_zero_weight(df_w_20k)
        df_w_20k.write_csv(
            f"../data/weighted/20k_edges/features/{self.prefix}{name.lower()}_20k.csv",
            has_header=False,
            separator="\t",
        )

        return df_w_composite


class Weighting:
    def __init__(self):
        self.model_prep = ModelPreprocessor()

    def main(self, dip: bool = False):
        prefix = "dip_" if dip else ""
        df_composite = construct_composite_network(dip=dip)

        # df_composite = self.model_prep.normalize_features(df_composite, FEATURES)

        print()
        print(f"========================================================")
        print(
            f"WEIGHTING: {'DIP COMPOSITE NETWORK' if dip else 'SWC COMPOSITE NETWORK'}"
        )
        print()

        print("---------------------------------------------------------")
        print("Writing unweighted network")
        df_unweighted = (
            df_composite.filter(pl.col(TOPO) > 0)
            .with_columns(pl.lit(1.0).alias(WEIGHT))
            .select([PROTEIN_U, PROTEIN_V, WEIGHT])
        )

        df_unweighted.write_csv(
            f"../data/weighted/{prefix}unweighted.csv", has_header=False, separator="\t"
        )
        print("Done writing unweighted network")
        print()

        print("------------- BEGIN: FEATURE WEIGHTING ----------------------")
        feat_weighting = FeatureWeighting(dip=dip)
        for f in FEATURES:
            df_f_weighted = feat_weighting.main(df_composite, [f], f)
            print(f"Done feature weighting using: {f}")
            assert_df_bounded(df_f_weighted, [WEIGHT])
        print("------------- END: FEATURE WEIGHTING ----------------------\n\n")

        print("------------- BEGIN: SUPER FEATURE WEIGHTING ----------------------")
        for f in SUPER_FEATS:
            df_f_weighted = feat_weighting.main(df_composite, f["features"], f["name"])
            print(f"Done feature weighting using: {f['name']} - {f['features']}")
            assert_df_bounded(df_f_weighted, [WEIGHT])
        print("------------- END: SUPER FEATURE WEIGHTING ----------------------\n\n")

        print()

        print("------------- BEGIN: SUPERVISED WEIGHTING ----------------------")
        n_iters = 10
        print(f"Cross-validation iterations: {n_iters}")
        print()

        # Supervised co-complex probability weighting
        xgw_params = {
            "objective": "binary:logistic",
            "n_estimators": 10000,
            "alpha": 30,
            "max_depth": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_jobs": -1,
            "learning_rate": 0.01,
        }
        xgw = SupervisedWeighting(XGBClassifier(**xgw_params), "XGW")

        for xval_iter in range(n_iters):
            print(f"------------------- BEGIN: ITER {xval_iter} ---------------------")
            df_train_pairs, _ = get_cyc_train_test_comp_pairs(xval_iter)
            df_train_labeled = self.model_prep.label_composite(
                df_composite, df_train_pairs, IS_CO_COMP, xval_iter, "subset", False
            )

            # SWC cross-validation scores
            df_w_swc = (
                pl.read_csv(
                    f"../data/swc/raw_weighted/{prefix}swc scored_edges iter{xval_iter}.txt",
                    new_columns=[PROTEIN_U, PROTEIN_V, WEIGHT],
                    separator=" ",
                    has_header=False,
                )
                .join(df_composite, on=[PROTEIN_U, PROTEIN_V])
                .select([PROTEIN_U, PROTEIN_V, WEIGHT])
            )

            # Rewrite SWC scores to another file as a form of preprocessing because
            # it needs to be formatted before running MCL.
            df_w_swc.write_csv(
                f"../data/weighted/all_edges/cross_val/{prefix}swc_iter{xval_iter}.csv",
                has_header=False,
                separator="\t",
            )

            # Write top 20k edges of SWC
            df_w_swc.sort(pl.col(WEIGHT), descending=True).head(20_000).write_csv(
                f"../data/weighted/20k_edges/cross_val/{prefix}swc_20k_iter{xval_iter}.csv",
                has_header=False,
                separator="\t",
            )

            # Weight the network using XGW
            df_w_xgw = xgw.main(df_composite, df_train_labeled, xval_iter)
            print(df_w_xgw)

            print(
                f"------------------- END: ITER {xval_iter} ---------------------\n\n"
            )

        print()
        print(f"All {n_iters} iterations done!")


if __name__ == "__main__":
    pl.Config.set_tbl_rows(15)

    start = time.time()
    weighting = Weighting()

    print(
        "------------------------ Weighting the composite network --------------------"
    )
    weighting.main(dip=False)
    print()

    # print(
    #     "------------------------ Weighting the DIP composite network --------------------"
    # )
    # weighting.main(dip=True)
    # print()

    print(f"Execution time: {time.time() - start}")
