# pyright: basic

from typing import List

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.neural_network import MLPClassifier

from aliases import (
    FEATURES,
    IS_CO_COMP,
    MAE,
    MODEL,
    MSE,
    PROTEIN_U,
    PROTEIN_V,
    SCENARIO,
    SUPER_FEATS,
    WEIGHT,
    XVAL_ITER,
)
from assertions import assert_df_normalized
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
        df_w_composite.sort(pl.col(WEIGHT), descending=True).head(20_000).write_csv(
            f"../data/weighted/20k_edges/features/{name.lower()}_20k.csv",
            has_header=False,
            separator="\t",
        )

        return df_w_composite


if __name__ == "__main__":
    pl.Config.set_tbl_rows(15)

    df_composite = construct_composite_network()

    print("------------- BEGIN: FEATURE WEIGHTING ----------------------")
    feat_weighting = FeatureWeighting()
    for f in FEATURES:
        df_f_weighted = feat_weighting.main(df_composite, [f], f)
        print(f"Done feature weighting using: {f}")
        assert_df_normalized(df_f_weighted, [WEIGHT])
    print("------------- END: FEATURE WEIGHTING ----------------------\n\n")

    print("------------- BEGIN: SUPER FEATURE WEIGHTING ----------------------")
    for f in SUPER_FEATS:
        df_f_weighted = feat_weighting.main(df_composite, f["features"], f["name"])
        print(f"Done feature weighting using: {f['name']} - {f['features']}")
        assert_df_normalized(df_f_weighted, [WEIGHT])
    print("------------- END: SUPER FEATURE WEIGHTING ----------------------\n\n")

    print("------------- BEGIN: SUPERVISED WEIGHTING ----------------------")
    n_iters = 10
    print(f"Cross-validation iterations: {n_iters}")
    print()

    features = FEATURES[:]
    model_prep = ModelPreprocessor()
    df_composite = model_prep.normalize_features(df_composite, features)

    # Weighting Models
    rf = CoCompClassifier(RandomForestClassifier(), "rf")
    cnb = CoCompClassifier(CategoricalNB(), "cnb")
    mlp = CoCompClassifier(MLPClassifier(max_iter=2000), "mlp")

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
            df_composite, df_train_pairs, IS_CO_COMP, xval_iter
        )
        # Preparing data for evaluations
        df_test_labeled = model_prep.label_composite(
            df_composite, df_test_pairs, IS_CO_COMP, xval_iter
        )
        df_all_pairs = pl.concat([df_train_pairs, df_test_pairs])
        df_all_labeled = model_prep.label_composite(
            df_composite, df_all_pairs, IS_CO_COMP, xval_iter
        )

        # Discretize the network using MDLP
        df_composite_binned = model_prep.discretize_composite(
            df_composite, df_train_labeled, features, IS_CO_COMP, xval_iter
        )

        # SWC cross-val output
        df_w_swc = pl.read_csv(
            f"../data/swc/all_edges/cross_val/swc scored_edges iter{xval_iter}.txt",
            new_columns=[PROTEIN_U, PROTEIN_V, WEIGHT],
            separator=" ",
        )

        # Run the classifiers
        df_w_cnb = cnb.main(df_composite_binned, df_train_labeled, xval_iter)
        df_w_rf = rf.main(df_composite, df_train_labeled, xval_iter)
        df_w_mlp = mlp.main(df_composite, df_train_labeled, xval_iter)

        # Evaluate the classifiers
        df_cnb_eval = evaluator.evaluate_co_comp_classifier(
            cnb.name, df_w_cnb, df_test_labeled, df_all_labeled, IS_CO_COMP
        )
        df_rf_eval = evaluator.evaluate_co_comp_classifier(
            rf.name, df_w_rf, df_test_labeled, df_all_labeled, IS_CO_COMP
        )
        df_mlp_eval = evaluator.evaluate_co_comp_classifier(
            mlp.name, df_w_mlp, df_test_labeled, df_all_labeled, IS_CO_COMP
        )

        # Evaluate SWC as well
        df_swc_eval = evaluator.evaluate_co_comp_classifier(
            "swc", df_w_swc, df_test_labeled, df_all_labeled, IS_CO_COMP
        )

        print()
        print(f"Evaluation summary of the models on xval_iter={xval_iter}")

        df_iter_evals = (
            pl.concat([df_cnb_eval, df_rf_eval, df_mlp_eval, df_swc_eval])
            .melt(
                id_vars=[MODEL, SCENARIO],
                variable_name="ERROR_KIND",
                value_name="ERROR_VAL",
            )
            .pivot(
                values="ERROR_VAL",
                index=[MODEL, "ERROR_KIND"],
                columns=SCENARIO,
                aggregate_function="first",
                maintain_order=True,
            )
        )
        print(df_iter_evals)

        df_evals = pl.concat(
            [df_evals, df_cnb_eval, df_rf_eval, df_mlp_eval, df_swc_eval]
        )

        print(f"------------------- END: ITER {xval_iter} ---------------------\n\n")

    print(f"All {n_iters} iterations done!")
    print("Comparing evaluation scores of the co-complex classifiers")

    sns.set_palette("deep")
    df_pd_evals = df_evals.to_pandas()

    plt.figure()
    mae_fig = sns.barplot(data=df_pd_evals, x=SCENARIO, y=MAE, hue=MODEL)
    mae_fig.set_title("MAE of various co-complex classifiers.")

    plt.figure()
    mse_fig = sns.barplot(data=df_pd_evals, x=SCENARIO, y=MSE, hue=MODEL)
    mse_fig.set_title("MSE of various co-complex classifiers.")

    plt.show()
