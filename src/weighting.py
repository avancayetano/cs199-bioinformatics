# pyright: basic

import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import CategoricalNB

from aliases import (
    FEATURES,
    IS_CO_COMP,
    PROBA_CO_COMP,
    PROBA_NON_CO_COMP,
    PROTEIN_U,
    PROTEIN_V,
    WEIGHT,
)
from co_comp_classifier import CoCompClassifier
from feature_preprocessor import FeaturePreprocessor
from utils import construct_composite_network, get_cyc_train_test_cmp_pairs

if __name__ == "__main__":
    pl.Config.set_tbl_cols(40)
    pl.Config.set_tbl_rows(7)

    features = FEATURES
    iters = 10

    feature_prep = FeaturePreprocessor()
    rf = CoCompClassifier(features, RandomForestClassifier(n_estimators=500), "rf")
    cnb = CoCompClassifier(features, CategoricalNB(), "cnb")

    df_composite = construct_composite_network()
    df_composite = feature_prep.normalize_features(df_composite, features)

    for i in range(iters):
        print(f"------------------- BEGIN: ITER {i} ---------------------")
        df_train_pairs, df_test_pairs = get_cyc_train_test_cmp_pairs(i)

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

        print(f"------------------- END: ITER {i} ---------------------")
