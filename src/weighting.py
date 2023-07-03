"""
A script that weights the composite network using XGW and the features.
"""

import time
from typing import Dict, List, Union

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
    def __init__(self, dip: bool) -> None:
        self.prefix = "dip_" if dip else ""

    def main(
        self, df_composite: pl.DataFrame, features: List[str], name: str
    ) -> pl.DataFrame:
        """
        Weighting method for feature weighting methods.
        NOTE:
        1. If method is a super feature, this gets the average of all the features
            of the super feature.
        2. Only gets non-zero weights.

        Args:
            df_composite (pl.DataFrame): Composite network.
            features (List[str]): List of features.
            name (str): Name of the method.

        Returns:
            pl.DataFrame: Weighted network.
        """
        df_w_composite = (
            df_composite.lazy()
            .select([PROTEIN_U, PROTEIN_V, *features])
            .with_columns((pl.sum(features) / len(features)).alias(WEIGHT))
            .select([PROTEIN_U, PROTEIN_V, WEIGHT])
            .filter(pl.col(WEIGHT) > 0)
            .collect()
        )

        df_w_composite.write_csv(
            f"../data/weighted/all_edges/features/{self.prefix}{name.lower()}.csv",
            has_header=False,
            separator="\t",
        )

        df_w_20k = (
            df_w_composite.sort(pl.col(WEIGHT), descending=True)
            .head(20_000)
            .filter(pl.col(WEIGHT) > 0)
        )

        assert_no_zero_weight(df_w_20k)
        df_w_20k.write_csv(
            f"../data/weighted/20k_edges/features/{self.prefix}{name.lower()}_20k.csv",
            has_header=False,
            separator="\t",
        )

        return df_w_composite


class Weighting:
    def __init__(self, dip: bool):
        """
        Does all the weighting of all the methods and determines the feature importances.

        Args:
            dip (bool): Whether to weight DIP network or not.
        """
        self.model_prep = ModelPreprocessor()
        self.df_composite = construct_composite_network(dip=dip)

        assert_df_bounded(self.df_composite, FEATURES)
        print("All scores bounded!")
        self.dip = dip

    def main(self, re_weight: bool = True):
        """
        Main method.

        Args:
            re_weight (bool, optional): Whether to re-weight the network or not.
                If set to False, this method will only graph the feature importances.
                Defaults to True.
        """
        prefix = "dip_" if self.dip else ""
        if re_weight:
            print("---------------------------------------------------------")
            print("Writing unweighted network")

            df_unweighted = (
                self.df_composite.filter(pl.col(TOPO) > 0)
                .with_columns(pl.lit(1.0).alias(WEIGHT))
                .select([PROTEIN_U, PROTEIN_V, WEIGHT])
            )

            df_unweighted.write_csv(
                f"../data/weighted/{prefix}unweighted.csv",
                has_header=False,
                separator="\t",
            )
            print("Done writing unweighted network")
            print()

            print("------------- BEGIN: FEATURE WEIGHTING ----------------------")
            feat_weighting = FeatureWeighting(dip=self.dip)
            for f in FEATURES:
                df_f_weighted = feat_weighting.main(self.df_composite, [f], f)
                print(f"Done feature weighting using: {f}")
                assert_df_bounded(df_f_weighted, [WEIGHT])

            print("------------- END: FEATURE WEIGHTING ----------------------\n\n")

            print("------------- BEGIN: SUPER FEATURE WEIGHTING ----------------------")
            for f in SUPER_FEATS:
                df_f_weighted = feat_weighting.main(
                    self.df_composite, f["features"], f["name"]
                )
                print(f"Done feature weighting using: {f['name']} - {f['features']}")
                assert_df_bounded(df_f_weighted, [WEIGHT])
            print(
                "------------- END: SUPER FEATURE WEIGHTING ----------------------\n\n"
            )

            print()

            print("------------- BEGIN: SUPERVISED WEIGHTING ----------------------")
            n_iters = 10
            print(f"Cross-validation iterations: {n_iters}")
            print()

            # Supervised co-complex probability weighting
            # For hyperparameter tuning. NOTE: takes too much time...
            # I already pre-tuned the hyperparameters; they can be found in data/training/.
            # Recommendations: make the model as conservative as possible to avoid
            # overfitting to the training set since the training set is highly different
            # from the testing set (i.e. mostly 2 or 3-sized train complexes
            # vs >= 4-sized test complexes.)
            xgw_params_grid = {
                "objective": ["binary:logistic"],
                "n_estimators": [1000],
                "max_depth": [3, 4],
                "gamma": [0.0, 0.5],
                "lambda": [50, 100],  # recommended to have high regularization
                "subsample": [0.6, 0.8],
                "colsample_bytree": [0.6, 0.8],
                "n_jobs": [-1],
                "learning_rate": [0.01],
                "tree_method": ["exact"],
            }

            xgw_model = XGBClassifier()
            xgw = SupervisedWeighting(xgw_model, "XGW", dip=self.dip)

            all_importances: List[Dict[str, Union[float, int]]] = []
            for xval_iter in range(n_iters):
                print(
                    f"------------------- BEGIN: ITER {xval_iter} ---------------------"
                )
                df_train_pairs, _ = get_cyc_train_test_comp_pairs(xval_iter, self.dip)
                df_train_labeled = self.model_prep.label_composite(
                    self.df_composite,
                    df_train_pairs,
                    IS_CO_COMP,
                    xval_iter,
                    "subset",
                    False,
                )

                # SWC cross-validation scores
                df_w_swc = (
                    pl.read_csv(
                        f"../data/swc/raw_weighted/{prefix}swc scored_edges iter{xval_iter}.txt",
                        new_columns=[PROTEIN_U, PROTEIN_V, WEIGHT],
                        separator=" ",
                        has_header=False,
                    )
                    .join(self.df_composite, on=[PROTEIN_U, PROTEIN_V])
                    .select([PROTEIN_U, PROTEIN_V, WEIGHT])
                )

                assert_df_bounded(df_w_swc, [WEIGHT])

                print()
                print("SWC SCORES DESCRIPTION")
                print(df_w_swc.describe())
                print()

                # Rewrite SWC scores to another file as a form of preprocessing because
                # it needs to be formatted before running MCL.
                df_w_swc.write_csv(
                    f"../data/weighted/all_edges/cross_val/{prefix}swc_iter{xval_iter}.csv",
                    has_header=False,
                    separator="\t",
                )

                # Write top 20k edges of SWC
                df_w_swc_20k = df_w_swc.sort(pl.col(WEIGHT), descending=True).head(
                    20_000
                )
                df_w_swc_20k.write_csv(
                    f"../data/weighted/20k_edges/cross_val/{prefix}swc_20k_iter{xval_iter}.csv",
                    has_header=False,
                    separator="\t",
                )

                # Weight the network using XGW
                df_w_xgw, feature_importances = xgw.main(
                    df_composite=self.df_composite,
                    df_train_labeled=df_train_labeled,
                    xval_iter=xval_iter,
                    tune=False,
                    params_grid=xgw_params_grid,
                    use_pretuned_params=True,
                )

                all_importances.append(feature_importances)

                assert_df_bounded(df_w_xgw, [WEIGHT])

                print()
                print("XGW SCORES DESCRIPTION")
                print(df_w_xgw.describe())
                print()

                print(
                    f"------------------- END: ITER {xval_iter} ---------------------\n\n"
                )

            print()
            print(f"All {n_iters} iterations done!")
            df_importances = pl.DataFrame(all_importances)
            df_importances.write_csv(
                f"../data/training/{prefix}xgw_feat_importances.csv", has_header=True
            )

        df_importances = pl.read_csv(
            f"../data/training/{prefix}xgw_feat_importances.csv", has_header=True
        )
        print(f"Feature importances")
        print(df_importances)

        # TODO: Plot the importances


if __name__ == "__main__":
    pl.Config.set_tbl_rows(15)
    pl.Config.set_tbl_cols(15)
    start = time.time()

    print("====================  Weighting the composite network ====================")
    weighting = Weighting(dip=False)
    weighting.main(re_weight=False)
    print()

    print("================== Weighting the DIP composite network ===================")
    weighting = Weighting(dip=True)
    weighting.main(re_weight=False)
    print()

    print(f"Execution time: {time.time() - start}")
