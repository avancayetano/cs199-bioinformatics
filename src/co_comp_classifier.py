# pyright: basic

from typing import Union

import polars as pl
from sklearn.ensemble import RandomForestClassifier
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
from assertions import assert_no_zero_weight

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

        if self.model is None:
            return df_composite

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

        print(
            f"Train samples: {n_samples} | Co-comp: {co_comp_samples} | Non-co-comp: {non_co_comp_samples}"
        )
        self.model.fit(X_train, y_train)  # training the model
        print("Training done!")

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

        print("Weighting done!")
        return df_w_composite

    def main(
        self, df_composite: pl.DataFrame, df_train_labeled: pl.DataFrame, xval_iter: int
    ) -> pl.DataFrame:
        df_w_composite = self.weight(df_composite, df_train_labeled, xval_iter)

        df_w_composite = (
            df_w_composite.lazy()
            .rename({PROBA_CO_COMP: WEIGHT})
            .select([PROTEIN_U, PROTEIN_V, WEIGHT])
            .collect()
        )

        df_w_composite.write_csv(
            f"../data/weighted/all_edges/cross_val/{self.name.lower()}_iter{xval_iter}.csv",
            has_header=False,
            separator="\t",
        )

        df_w_20k = df_w_composite.sort(pl.col(WEIGHT), descending=True).head(20_000)
        assert_no_zero_weight(df_w_20k)
        df_w_20k.write_csv(
            f"../data/weighted/20k_edges/cross_val/{self.name.lower()}_20k_iter{xval_iter}.csv",
            has_header=False,
            separator="\t",
        )

        return df_w_composite
