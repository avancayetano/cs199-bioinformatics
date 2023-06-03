# pyright: basic

from typing import Union

import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.naive_bayes import CategoricalNB
from sklearn.neural_network import MLPClassifier

from aliases import (
    IS_CO_COMP,
    PROBA_CO_COMP,
    PROBA_NON_CO_COMP,
    PROTEIN_U,
    PROTEIN_V,
    WEIGHT,
)

WeightingModel = Union[RandomForestClassifier, CategoricalNB, MLPClassifier]


class CoCompClassifier:
    """
    Co-complex classifier.
    """

    def __init__(
        self,
        model: WeightingModel,
        name: str,
    ):
        self.label = IS_CO_COMP
        self.model = model
        self.name = name

    def weight(
        self, df_composite: pl.DataFrame, df_train_labeled: pl.DataFrame, xval_iter: int
    ) -> pl.DataFrame:
        """
        Weight composite network based on co-complex probability.

        Args:
            df_composite (pl.DataFrame): _description_
            df_train_labeled (pl.DataFrame): _description_

        Returns:
            pl.DataFrame: _description_
        """

        selected_features = df_composite.select(
            pl.exclude([PROTEIN_U, PROTEIN_V, self.label])
        ).columns
        print(f"Weighting model: {self.name}")
        print(f"Selected features: {selected_features}")

        df_feat_label = df_train_labeled.join(
            df_composite, on=[PROTEIN_U, PROTEIN_V], how="left"
        )
        X_train = df_feat_label.select(selected_features).to_numpy()
        y_train = df_feat_label.select(self.label).to_numpy().ravel()

        n_samples = X_train.shape[0]
        co_comp_samples = y_train[y_train == 1].shape[0]
        non_co_comp_samples = n_samples - co_comp_samples

        print(f"Training the model")
        print(
            f"Train samples: {n_samples} | Co-comp samples: {co_comp_samples} | Non-co-comp samples: {non_co_comp_samples}"
        )
        self.model.fit(X_train, y_train)  # training the model

        # After learning the parameters, weight all protein pairs
        X_test = df_composite.select(selected_features).to_numpy()
        ndarr_pred = self.model.predict_proba(X_test)

        CLASS_PROBA = [PROBA_NON_CO_COMP, PROBA_CO_COMP]

        df_weights = pl.from_numpy(
            ndarr_pred, schema=[CLASS_PROBA[c] for c in self.model.classes_]
        )

        df_w_composite = pl.concat([df_composite, df_weights], how="horizontal").select(
            [PROTEIN_U, PROTEIN_V, PROBA_NON_CO_COMP, PROBA_CO_COMP]
        )

        df_w_composite.write_csv(
            f"../data/training/{self.name.lower()}_probas_iter{xval_iter}.csv"
        )

        return df_w_composite

    def evaluate(
        self,
        df_w_composite: pl.DataFrame,
        df_test_labeled: pl.DataFrame,
        df_all_labeled: pl.DataFrame,
        xval_iter: int,
    ) -> pl.DataFrame:
        """
        Evaluate the results in terms of:
        1. Predicting co-comp pairs in test co-comp pairs
        2. Predicting co-comp pairs in (train + test) co-comp pairs
        3. Predicting co-comp and non-co-comp pairs in (test) (co-comp + non-co-comp) pairs
        3. Predicting co-comp and non-co-comp pairs in (train + test) (co-comp + non-co-comp) pairs

        NOTE:
        - Labeled: Both co-comp and non-co-comp labeled pairs

        Scenario codes:
        1. TEST_CO
        2. ALL_CO
        3. TEST_CO-NONCO
        4. ALL_CO-NONCO

        Args:
            df_w_composite (pl.DataFrame): _description_
            df_train_labeled (pl.DataFrame): _description_
            df_test_labeled (pl.DataFrame): _description_
            xval_iter (int): _description_
        """

        df_w_test_labeled = df_w_composite.join(
            df_test_labeled, on=[PROTEIN_U, PROTEIN_V], how="inner"
        )
        df_w_all_labeled = df_w_composite.join(
            df_all_labeled, on=[PROTEIN_U, PROTEIN_V], how="inner"
        )
        scenarios = {
            "TEST_CO": df_w_test_labeled.filter(pl.col(IS_CO_COMP) == 1),
            "ALL_CO": df_w_all_labeled.filter(pl.col(IS_CO_COMP) == 1),
            "TEST_CO-NONCO": df_w_test_labeled,
            "ALL_CO-NONCO": df_w_all_labeled,
        }
        eval_summary = {
            "MODEL": [],
            "XVAL_ITER": [],
            "SCENARIO": [],
            "MAE": [],
        }
        for s in scenarios:
            eval_info = self.get_eval_info(scenarios[s])
            eval_summary["MODEL"].append(self.name)
            eval_summary["XVAL_ITER"].append(xval_iter)
            eval_summary["SCENARIO"].append(s)
            eval_summary["MAE"].append(eval_info["MAE"])

        # Add summary row
        eval_summary["MODEL"].append(self.name)
        eval_summary["XVAL_ITER"].append(xval_iter)
        eval_summary["SCENARIO"].append("AVG")
        eval_summary["MAE"].append(sum(eval_summary["MAE"]) / len(scenarios.keys()))

        df_eval_summary = pl.DataFrame(eval_summary)

        return df_eval_summary

    def get_eval_info(self, df_pred_label: pl.DataFrame):
        y_pred = df_pred_label.select(WEIGHT).to_series().to_numpy()
        y_true = df_pred_label.select(self.label).to_series().to_numpy()
        return {
            "MAE": mean_absolute_error(y_true, y_pred),
        }

    def main(
        self, df_composite: pl.DataFrame, df_train_labeled: pl.DataFrame, xval_iter: int
    ) -> pl.DataFrame:
        print()
        df_w_composite = self.weight(df_composite, df_train_labeled, xval_iter)

        df_w_composite = (
            df_w_composite.lazy()
            .rename({PROBA_CO_COMP: WEIGHT})
            .select([PROTEIN_U, PROTEIN_V, WEIGHT])
            .collect()
        )

        df_w_composite.write_csv(
            f"../data/weighted/all_edges/cross_val/{self.name}_iter{xval_iter}.csv",
            has_header=False,
            separator="\t",
        )

        df_w_composite.sort(pl.col(WEIGHT), descending=True).head(20_000).write_csv(
            f"../data/weighted/20k_edges/cross_val/{self.name}_iter{xval_iter}.csv",
            has_header=False,
            separator="\t",
        )
        print()

        return df_w_composite
