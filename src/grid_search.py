# pyright: basic

import time
from typing import Dict, Union

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import CategoricalNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from aliases import (
    FEATURES,
    IS_CO_COMP,
    PROBA_CO_COMP,
    PROBA_NON_CO_COMP,
    PROTEIN_U,
    PROTEIN_V,
    WEIGHT,
)
from model_preprocessor import ModelPreprocessor
from utils import (
    construct_composite_network,
    get_cyc_comp_pairs,
    get_cyc_train_test_comp_pairs,
)

Model = Union[
    RandomForestClassifier,
    CategoricalNB,
    MLPClassifier,
    Pipeline,
    VotingClassifier,
]


class GridSearch:
    """
    Goals:
    1. Find best hyperparameters for RF.
    2. Find the best labeling method.

    Use all co-complex pairs for cross-validation.

    Two labeling methods for hyperparameter tuning. Need to find which is better.
    1. mode = "all", balanced = True
    2. mode = "subset", balanced = False

    Hyperparameter tuning

    Testing:
    - For each labeling method, validate the best estimator against
        the whole composite network labeled (mode=all, balanced=False)
    """

    def __init__(self):
        self.model_prep = ModelPreprocessor()
        self.df_composite = self.model_prep.normalize_features(
            construct_composite_network(), FEATURES
        )
        self.comp_pairs = get_cyc_comp_pairs()

        # Two labeled network
        df_all_bal_labels = self.model_prep.label_composite(
            df_composite=self.df_composite,
            df_positive_pairs=self.comp_pairs,
            label=IS_CO_COMP,
            seed=0,
            mode="all",
            balanced=True,
        )

        df_subset_unbal_labels = self.model_prep.label_composite(
            df_composite=self.df_composite,
            df_positive_pairs=self.comp_pairs,
            label=IS_CO_COMP,
            seed=0,
            mode="subset",
            balanced=False,
        )

        # Labeled data with the feature values
        self.df_all_bal = self.df_composite.join(
            df_all_bal_labels, on=[PROTEIN_U, PROTEIN_V], how="inner"
        ).select(pl.exclude([PROTEIN_U, PROTEIN_V]))

        self.df_subset_unbal = self.df_composite.join(
            df_subset_unbal_labels, on=[PROTEIN_U, PROTEIN_V], how="inner"
        ).select(pl.exclude([PROTEIN_U, PROTEIN_V]))

        # True labels
        self.y_true = (
            self.model_prep.label_composite(
                df_composite=self.df_composite,
                df_positive_pairs=self.comp_pairs,
                label=IS_CO_COMP,
                seed=0,
                mode="all",
                balanced=False,
            )
            .select(IS_CO_COMP)
            .to_numpy()
            .ravel()
        )

    def tune(self, model: Model) -> Dict[str, Model]:
        feat_labels = {"all_bal": self.df_all_bal, "subset_unbal": self.df_subset_unbal}
        tuned = {}
        print(f"Tuning {model.__class__.__name__}...")
        for label_mode in feat_labels:
            X = feat_labels[label_mode].select(FEATURES).to_numpy()
            y = feat_labels[label_mode].select(IS_CO_COMP).to_numpy().ravel()

            if type(model) == RandomForestClassifier:
                parameters = {
                    "n_estimators": [500, 1000, 2000],
                    "criterion": ["gini", "entropy", "log_loss"],
                    "max_features": ["sqrt", "log2", None],
                }
            else:
                parameters = {}

            clf = GridSearchCV(
                model, parameters, n_jobs=-1, cv=5, scoring="average_precision"
            )

            clf.fit(X, y)
            print(f"Label mode: {label_mode}:")
            print(f"Best params: {clf.best_params_}")
            print(f"Best score (AP): {clf.best_score_}")
            tuned[label_mode] = clf.best_estimator_

        return tuned

    def evaluate(self, tuned_model: Model, label_mode: str):
        """
        This is to determine the best labeling method.

        Args:
            tuned_model (Model): _description_
            label_mode (str): _description_
        """

        CLASS_PROBA = [PROBA_NON_CO_COMP, PROBA_CO_COMP]
        y_clf_proba = tuned_model.predict_proba(
            self.df_composite.select(FEATURES).to_numpy()
        )
        df_pred_proba = pl.from_numpy(
            y_clf_proba, schema=[CLASS_PROBA[c] for c in tuned_model.classes_]
        )
        y_pred = df_pred_proba.select(PROBA_CO_COMP).to_numpy().ravel()
        print(f"Label mode: {label_mode}")
        print(f"AP: {average_precision_score(self.y_true, y_pred)}")

    def main(self):
        rf = RandomForestClassifier(random_state=12345, n_jobs=-1)

        models = [rf]
        for model in models:
            print("----------------------------------")
            tuned = self.tune(model)
            print()
            print("Evaluating the best_estimators...")
            for label_mode in tuned:
                self.evaluate(tuned[label_mode], label_mode)
            print("-------------------------------")


if __name__ == "__main__":
    start_time = time.time()
    grid_search = GridSearch()
    grid_search.main()

    print(f"Execution time: {time.time() - start_time}")
    print("================ END ====================")
