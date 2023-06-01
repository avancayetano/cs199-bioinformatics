# pyright: basic

import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import CategoricalNB

from aliases import FEATURES
from co_comp_classifier import CoCompClassifier
from feature_preprocessor import FeaturePreprocessor
from utils import construct_composite_network, get_cyc_train_test_cmp_pairs

if __name__ == "__main__":
    features = FEATURES
    iters = 10

    feature_prep = FeaturePreprocessor()
    rf = CoCompClassifier(features, RandomForestClassifier(), "rf")
    cnb = CoCompClassifier(features, CategoricalNB(), "cnb")

    df_composite = construct_composite_network()
    df_composite = feature_prep.normalize_features(df_composite, features)

    for i in range(iters):
        print(f"------------------- BEGIN: ITER {i} ---------------------")
        df_train_pairs, df_test_pairs = get_cyc_train_test_cmp_pairs(i)
        print("Train pairs")
        print(df_train_pairs)
        print("Test pairs")
        print(df_test_pairs)
        df_labeled = rf.label_composite(df_composite, df_train_pairs)
        print("Labeled")
        print(df_labeled)

        df_w_composite = rf.weight(df_labeled)

        df_w_composite.write_csv(
            f"../data/weighted/all_edges/cross_val/{rf.name}_iter{i}.csv"
        )
        print(df_w_composite)
        rf.validate(df_w_composite, df_train_pairs, df_test_pairs)

        print(f"------------------- END: ITER {i} ---------------------")
