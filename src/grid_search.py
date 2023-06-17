# pyright: basic

import time
from typing import Any, Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import CategoricalNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC

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


    Conclusion:
    The best model is Random Forest with the parameters:
    - n_estimators = 2000+, criterion = "entropy", max_features = "sqrt"
    """

    def __init__(self):
        self.model_prep = ModelPreprocessor()
        self.df_composite = self.model_prep.normalize_features(
            construct_composite_network(), FEATURES
        )

        train1, _ = get_cyc_train_test_comp_pairs(8)
        train2, _ = get_cyc_train_test_comp_pairs(3)

        self.comp_pairs = pl.concat([train1, train2], how="vertical").unique()

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

        print("Shapes...")
        print(self.df_all_bal.shape)
        print(self.df_subset_unbal.shape)

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
        """
        Find the best hyperparameters of a model.
        """
        # feat_labels = {"all_bal": self.df_all_bal, "subset_unbal": self.df_subset_unbal}
        feat_labels = {"subset_unbal": self.df_subset_unbal}
        tuned = {}
        print(f"Tuning {model.__class__.__name__}...")
        for label_mode in feat_labels:
            X = feat_labels[label_mode].select(FEATURES).to_numpy()
            y = feat_labels[label_mode].select(IS_CO_COMP).to_numpy().ravel()

            if type(model) == RandomForestClassifier:
                parameters = {
                    "n_estimators": [500],
                    "criterion": ["entropy"],
                    "max_features": ["sqrt"],
                }
                """
                Best parameters:
                - n_estimators: 1000
                - criterion: entropy
                - max_features: sqrt
                """

            elif type(model) == MLPClassifier:
                parameters = {
                    "hidden_layer_sizes": [(200,), (50, 50, 50), (500,)],
                    "activation": ["relu", "logistic"],
                    "solver": ["sgd", "adam"],
                    "alpha": [0.1],
                }
                #  {'activation': 'relu', 'hidden_layer_sizes': (200,), 'solver': 'sgd'}
                # parameters = {
                #     "hidden_layer_sizes": [(100,)],
                #     "activation": ["relu"],
                #     "solver": ["adam"],
                # }

                """
                Best parameters:
                - n_estimators: 1000
                - criterion: entropy
                - max_features: sqrt
                """
            elif type(model) == VotingClassifier:
                parameters = {
                    "rf__n_estimators": [1000],
                    "rf__criterion": ["entropy"],
                    "rf__max_features": ["sqrt"],
                    "mlp__hidden_layer_sizes": [
                        (200,),
                        (100,),
                        (100, 100),
                        (500,),
                        (200, 100, 100),
                    ],
                    "mlp__activation": ["relu", "logistic"],
                    "mlp__solver": ["adam", "sgd"],
                }
            else:
                parameters = {}

            clf = GridSearchCV(
                model,
                parameters,
                n_jobs=-1,
                cv=2,
                scoring="neg_root_mean_squared_error",
                refit=False,
            )

            clf.fit(X, y)
            print(f"Label mode: {label_mode}:")
            print(f"Best params: {clf.best_params_}")
            print(f"Best score (AP): {clf.best_score_}")
            tuned[label_mode] = self.refit(model, clf.best_params_)

        return tuned

    def refit(self, model: Model, best_params: Dict[str, Any]):
        print(f"Refitting {model.__class__.__name__}")
        model.set_params(**best_params)
        train, _ = get_cyc_train_test_comp_pairs(4)

        df_subset_unbal_labels = self.model_prep.label_composite(
            df_composite=self.df_composite,
            df_positive_pairs=self.comp_pairs,
            label=IS_CO_COMP,
            seed=0,
            mode="subset",
            balanced=False,
        )

        # Labeled data with the feature values
        df_subset_unbal = self.df_composite.join(
            df_subset_unbal_labels, on=[PROTEIN_U, PROTEIN_V], how="inner"
        ).select(pl.exclude([PROTEIN_U, PROTEIN_V]))

        X = df_subset_unbal.select(FEATURES).to_numpy()
        y = df_subset_unbal.select(IS_CO_COMP).to_numpy().ravel()

        model.fit(X, y)

        return model

    def evaluate(self, tuned_model: Model, label_mode: str):
        """
        This is to determine the best labeling method.
        NOTE: This is not an accurate evaluation of co-complex edge classification
            because tuned_model is fitted over the whole labeled data.
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
        print(f"Parameters: {tuned_model.get_params(False)}")
        print(f"RMSE: {mean_squared_error(self.y_true, y_pred) ** (0.5)}")

    def main(self):
        rf = RandomForestClassifier(random_state=12345, n_jobs=-1)
        mlp = MLPClassifier(max_iter=10000, random_state=12345)

        rf_mlp = VotingClassifier(
            estimators=[
                (
                    "rf",
                    RandomForestClassifier(random_state=12345, n_jobs=-1),
                ),
                (
                    "mlp",
                    MLPClassifier(max_iter=10000, random_state=12345),
                ),
            ],
            voting="soft",
        )

        models = [mlp]
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

    # rf = SupervisedWeighting(
    #     RandomForestClassifier(
    #         n_estimators=2000,
    #         criterion="entropy",
    #         max_features="sqrt",
    #         n_jobs=-1,
    #         random_state=6789,
    #     ),
    #     "RF",
    # )
    # mlp = SupervisedWeighting(
    #     MLPClassifier(
    #         hidden_layer_sizes=(200,),
    #         solver="sgd",
    #         activation="logistic",
    #         alpha=0.1,
    #         random_state=6789,
    #         max_iter=10000,
    #     ),
    #     "MLP",
    # )
    # rf_mlp = SupervisedWeighting(
    #     VotingClassifier(
    #         estimators=[
    #             (
    #                 "rf",
    #                 RandomForestClassifier(
    #                     n_estimators=2000,
    #                     criterion="entropy",
    #                     max_features="sqrt",
    #                     n_jobs=-1,
    #                     random_state=6789,
    #                 ),
    #             ),
    #             (
    #                 "mlp",
    #                 MLPClassifier(
    #                     hidden_layer_sizes=(200,),
    #                     solver="sgd",
    #                     activation="logistic",
    #                     alpha=0.1,
    #                     random_state=6789,
    #                     max_iter=10000,
    #                 ),
    #             ),
    #         ],
    #         voting="soft",
    #     ),
    #     "RF_MLP",
    # )
