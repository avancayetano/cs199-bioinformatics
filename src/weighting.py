# pyright: basic

from typing import List

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

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
)
from assertions import assert_df_bounded, assert_no_zero_weight
from co_comp_classifier import CoCompClassifier
from evaluator import Evaluator
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

    df_composite = construct_composite_network()

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

    features = FEATURES[:]
    model_prep = ModelPreprocessor()
    df_composite = model_prep.normalize_features(df_composite, features)
    df_composite_swc = model_prep.normalize_features(df_composite, SWC_FEATS).select(
        [PROTEIN_U, PROTEIN_V, *SWC_FEATS]
    )

    # Weighting Models
    rf = CoCompClassifier(RandomForestClassifier(), "RF")
    cnb = CoCompClassifier(CategoricalNB(), "CNB")
    mlp = CoCompClassifier(MLPClassifier(max_iter=3000), "MLP")
    rf_swc = CoCompClassifier(RandomForestClassifier(), "RF_SWC")

    rf_mlp_vote = CoCompClassifier(
        VotingClassifier(
            estimators=[
                ("rf", RandomForestClassifier()),
                ("mlp", MLPClassifier(max_iter=3000)),
            ],
            voting="soft",
        ),
        "RF_MLP_VOTE",
    )

    rf_mlp_stack = CoCompClassifier(
        StackingClassifier(
            estimators=[
                ("rf", RandomForestClassifier()),
            ],
            final_estimator=MLPClassifier(max_iter=3000),
            stack_method="predict_proba",
        ),
        "RF_MLP_STACK",
    )

    evaluator = Evaluator()

    df_evals = pl.DataFrame(
        schema={
            MODEL: pl.Utf8,
            SCENARIO: pl.Utf8,
            MAE: pl.Float64,
            MSE: pl.Float64,
        }
    )

    for xval_iter in range(n_iters):
        print(f"------------------- BEGIN: ITER {xval_iter} ---------------------")
        df_train_pairs, df_test_pairs = get_cyc_train_test_comp_pairs(xval_iter)
        df_train_labeled = model_prep.label_composite(
            df_composite, df_train_pairs, IS_CO_COMP, xval_iter, "subset", False
        )

        df_train_labeled_swc = model_prep.label_composite(
            df_composite_swc, df_train_pairs, IS_CO_COMP, xval_iter, "subset", False
        )

        # Discretize the network using MDLP
        # df_composite_binned = model_prep.discretize_composite(
        #     df_composite, df_train_labeled, features, IS_CO_COMP, xval_iter
        # )

        # SWC cross-validation scores
        df_w_swc = pl.read_csv(
            f"../data/swc/raw_weighted/swc scored_edges iter{xval_iter}.txt",
            new_columns=[PROTEIN_U, PROTEIN_V, WEIGHT],
            separator=" ",
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
        # df_w_cnb = cnb.main(df_composite_binned, df_train_labeled, xval_iter)
        df_w_rf = rf.main(df_composite, df_train_labeled, xval_iter)
        df_w_mlp = mlp.main(df_composite, df_train_labeled, xval_iter)
        df_w_rf_mlp_vote = rf_mlp_vote.main(df_composite, df_train_labeled, xval_iter)
        df_w_rf_mlp_stack = rf_mlp_stack.main(df_composite, df_train_labeled, xval_iter)
        df_w_rf_swc = rf_swc.main(df_composite_swc, df_train_labeled_swc, xval_iter)

        # Preparing labeled data for performance evaluations
        df_test_labeled = model_prep.label_composite(
            df_composite_swc, df_test_pairs, IS_CO_COMP, xval_iter, "all", False
        )
        df_all_pairs = pl.concat([df_train_pairs, df_test_pairs])
        df_all_labeled = model_prep.label_composite(
            df_composite_swc, df_all_pairs, IS_CO_COMP, xval_iter, "all", False
        )

        # Evaluate the classifiers
        # df_cnb_eval = evaluator.evaluate_co_comp_classifier(
        #     cnb.name, df_w_cnb, df_test_labeled, df_all_labeled, IS_CO_COMP
        # )
        df_rf_eval = evaluator.evaluate_co_comp_classifier(
            rf.name, df_w_rf, df_test_labeled, df_all_labeled, IS_CO_COMP
        )
        df_mlp_eval = evaluator.evaluate_co_comp_classifier(
            mlp.name, df_w_mlp, df_test_labeled, df_all_labeled, IS_CO_COMP
        )
        df_rf_mlp_vote_eval = evaluator.evaluate_co_comp_classifier(
            rf_mlp_vote.name,
            df_w_rf_mlp_vote,
            df_test_labeled,
            df_all_labeled,
            IS_CO_COMP,
        )
        df_rf_mlp_stack_eval = evaluator.evaluate_co_comp_classifier(
            rf_mlp_stack.name,
            df_w_rf_mlp_stack,
            df_test_labeled,
            df_all_labeled,
            IS_CO_COMP,
        )

        # # Evaluate SWC as well
        df_swc_eval = evaluator.evaluate_co_comp_classifier(
            "SWC", df_w_swc, df_test_labeled, df_all_labeled, IS_CO_COMP
        )
        df_rf_swc_eval = evaluator.evaluate_co_comp_classifier(
            rf_swc.name, df_w_rf_swc, df_test_labeled, df_all_labeled, IS_CO_COMP
        )
        print()
        print(f"Evaluation summary of the models on xval_iter={xval_iter}")

        df_iter_evals = (
            pl.concat(
                [
                    df_rf_eval,
                    df_mlp_eval,
                    df_swc_eval,
                    df_rf_swc_eval,
                    df_rf_mlp_vote_eval,
                    df_rf_mlp_stack_eval,
                ]
            )
            # pl.concat([df_cnb_eval, df_rf_eval, df_mlp_eval, df_swc_eval])
            # pl.concat([df_rf_swc_eval, df_rf_eval, df_rf_mlp_eval])
            # pl.concat([df_rf_eval, df_mlp_eval, df_swc_eval])
            # pl.concat([df_mlp_eval, df_swc_eval])
            .melt(
                id_vars=[MODEL, SCENARIO],
                variable_name="ERROR_KIND",
                value_name="ERROR_VAL",
            ).pivot(
                values="ERROR_VAL",
                index=[MODEL, "ERROR_KIND"],
                columns=SCENARIO,
                aggregate_function="first",
                maintain_order=True,
            )
        )
        print(df_iter_evals)
        # df_evals = pl.concat(
        #     [df_evals, df_cnb_eval, df_rf_eval, df_mlp_eval, df_swc_eval]
        # )
        df_evals = pl.concat(
            [
                df_rf_eval,
                df_mlp_eval,
                df_swc_eval,
                df_rf_swc_eval,
                df_rf_mlp_vote_eval,
                df_rf_mlp_stack_eval,
            ]
        )

        # df_evals = pl.concat([df_evals, df_rf_eval, df_mlp_eval, df_swc_eval])
        # df_evals = pl.concat([df_evals, df_mlp_eval, df_swc_eval])

        print(f"------------------- END: ITER {xval_iter} ---------------------\n\n")

    print(f"All {n_iters} iterations done!")

    sns.set_palette("deep")
    df_pd_evals = df_evals.to_pandas()

    plt.figure()
    mae_fig = sns.barplot(data=df_pd_evals, x=SCENARIO, y=MAE, hue=MODEL)
    mae_fig.set_title("MAE of various co-complex classifiers.")

    plt.figure()
    mse_fig = sns.barplot(data=df_pd_evals, x=SCENARIO, y=MSE, hue=MODEL)
    mse_fig.set_title("MSE of various co-complex classifiers.")

    plt.show()
