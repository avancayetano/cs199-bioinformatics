# pyright: basic

"""
A script that weights the composite network.
"""


from typing import List, TypedDict

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    auc,
    average_precision_score,
    mean_squared_error,
    precision_recall_curve,
)

from aliases import (
    CO_OCCUR,
    FEATURES,
    IS_CO_COMP,
    MAE,
    MODEL,
    MSE,
    PROTEIN_U,
    PROTEIN_V,
    SCENARIO,
    STRING,
    SUPER_FEATS,
    SWC_FEATS,
    TOPO,
    TOPO_L2,
    WEIGHT,
    XVAL_ITER,
)
from assertions import assert_df_bounded, assert_no_zero_weight
from model_preprocessor import ModelPreprocessor
from supervised_weighting import SupervisedWeighting
from utils import (
    construct_composite_network,
    get_cyc_comp_pairs,
    get_cyc_train_test_comp_pairs,
)

METHOD = "METHOD"
AP = "AP"
AUC = "AUC"
RMSE = "RMSE"


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

    model_prep = ModelPreprocessor()
    df_composite = construct_composite_network()
    df_composite = model_prep.normalize_features(df_composite, FEATURES)
    df_composite_swc = df_composite.select([PROTEIN_U, PROTEIN_V, *SWC_FEATS])

    print("Writing unweighted network")
    df_unweighted = (
        df_composite.filter(pl.col(TOPO) > 0)
        .with_columns(pl.lit(1.0).alias(WEIGHT))
        .select([PROTEIN_U, PROTEIN_V, WEIGHT])
    )

    df_unweighted.write_csv(
        "../data/weighted/unweighted.csv", has_header=False, separator="\t"
    )
    print("Done writing unweighted network")
    print()

    print("------------- BEGIN: FEATURE WEIGHTING ----------------------")
    feat_weighting = FeatureWeighting()
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

    print("------------- BEGIN: SUPERVISED WEIGHTING ----------------------")
    n_iters = 10
    print(f"Cross-validation iterations: {n_iters}")
    print()

    # Supervised co-complex probability weighting
    rf_params = {
        "n_estimators": 2000,
        "criterion": "entropy",
        "max_features": "sqrt",
        "n_jobs": -1,
    }  # from grid searching

    rfw_swc = SupervisedWeighting(RandomForestClassifier(**rf_params), "RFW_SWC")
    rfw = SupervisedWeighting(RandomForestClassifier(**rf_params), "RFW")

    df_comp_pairs = get_cyc_comp_pairs()
    y_true = (
        model_prep.label_composite(
            df_composite, df_comp_pairs, IS_CO_COMP, -1, "all", False
        )
        .select(IS_CO_COMP)
        .to_numpy()
        .ravel()
    )

    sv_methods = ["SWC", "RFW_SWC", "RFW"]

    precision_evals = {method: [] for method in sv_methods}
    recall_evals = {method: [] for method in sv_methods}

    evals = []

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

        # Rewrite SWC scores to another file as a form of preprocessing before MCL clustering
        df_w_swc.write_csv(
            f"../data/weighted/all_edges/cross_val/swc_iter{xval_iter}.csv",
            has_header=False,
            separator="\t",
        )

        # Write top 20k edges of SWC
        df_w_swc_20k = (
            df_w_swc.sort(pl.col(WEIGHT), descending=True)
            .head(20_000)
            .write_csv(
                f"../data/weighted/20k_edges/cross_val/swc_20k_iter{xval_iter}.csv",
                has_header=False,
                separator="\t",
            )
        )

        # Run the classifiers
        df_w_rfw_swc = rfw_swc.main(df_composite_swc, df_train_labeled, xval_iter)
        df_w_rfw = rfw.main(df_composite, df_train_labeled, xval_iter)

        w_networks = {"SWC": df_w_swc, "RFW_SWC": df_w_rfw_swc, "RFW": df_w_rfw}

        for method in sv_methods:
            y_pred = w_networks[method].select(WEIGHT).to_numpy().ravel()

            precision, recall, thresholds = precision_recall_curve(
                y_true, y_pred, pos_label=1
            )
            pr_auc = auc(recall, precision)
            ap = average_precision_score(y_true, y_pred, pos_label=1)
            rmse = mean_squared_error(y_true, y_pred, squared=False)

            precision_evals[method].append(precision)
            recall_evals[method].append(recall)

            print()
            print(f"Evaluations of {method}")
            print(f"Precision-Recall AUC: {pr_auc}")
            print(f"Average precision: {pr_auc}")
            print(f"RMSE: {rmse}")

            evals.append(
                {METHOD: method, XVAL_ITER: xval_iter, AUC: pr_auc, AP: ap, RMSE: rmse}
            )
        print(f"------------------- END: ITER {xval_iter} ---------------------\n\n")

    print(f"All {n_iters} iterations done!")

    # sns.set_palette("deep")

    print()
    print("Average of all evaluations on all the cross-val iterations")
    df_evals = pl.DataFrame(evals).groupby(METHOD).mean()
    print(df_evals)

    
    # plt.show()
