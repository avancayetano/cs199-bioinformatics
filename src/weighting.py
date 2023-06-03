# pyright: basic

from typing import List

import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.neural_network import MLPClassifier

from aliases import FEATURES, IS_CO_COMP, PROTEIN_U, PROTEIN_V, SUPER_FEATS, WEIGHT
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
        assert_df_normalized(df_f_weighted, WEIGHT)
    print("------------- END: FEATURE WEIGHTING ----------------------\n\n")

    print("------------- BEGIN: SUPER FEATURE WEIGHTING ----------------------")
    for f in SUPER_FEATS:
        df_f_weighted = feat_weighting.main(df_composite, f["features"], f["name"])
        print(f"Done feature weighting using: {f['name']} - {f['features']}")
        assert_df_normalized(df_f_weighted, WEIGHT)
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

    df_rf_eval_summary = pl.DataFrame()
    df_cnb_eval_summary = pl.DataFrame()

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

        # Run the classifiers

        # Discretize the network using MDLP
        df_composite_binned = model_prep.discretize_composite(
            df_composite, df_train_labeled, features, IS_CO_COMP, xval_iter
        )

        # Run the random forest classifier
        df_w_cnb = cnb.main(df_composite_binned, df_train_labeled, xval_iter)

        # Evaluate categorical Naive-Bayes performance
        df_cnb_eval = cnb.evaluate(df_w_cnb, df_test_labeled, df_all_labeled, xval_iter)

        if df_cnb_eval_summary.is_empty():
            df_cnb_eval_summary = df_cnb_eval
        else:
            df_cnb_eval_summary = pl.concat([df_cnb_eval_summary, df_cnb_eval])

        # Run the random forest classifier
        df_w_rf = rf.main(df_composite, df_train_labeled, xval_iter)

        # Evaluate random forest performance
        df_rf_eval = rf.evaluate(df_w_rf, df_test_labeled, df_all_labeled, xval_iter)
        if df_rf_eval_summary.is_empty():
            df_rf_eval_summary = df_rf_eval
        else:
            df_rf_eval_summary = pl.concat([df_rf_eval_summary, df_rf_eval])

        print()
        print(f"Evaluation summary of the models on xval_iter={xval_iter}")
        print(df_cnb_eval)
        print(df_rf_eval)

        print(f"------------------- END: ITER {xval_iter} ---------------------\n\n")

    print(f"All {n_iters} iterations done!")
    print("Comparing evaluation scores of the co-complex classifiers")
    print(df_rf_eval_summary)
    print(df_cnb_eval_summary)
