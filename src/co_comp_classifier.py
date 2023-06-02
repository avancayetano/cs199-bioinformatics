# pyright: basic

from typing import List, Optional, Set, Union

import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.neural_network import MLPClassifier

from aliases import (
    CO_EXP,
    CO_OCCUR,
    GO_BP,
    GO_CC,
    GO_MF,
    IS_CO_COMP,
    IS_NIP,
    PROBA_CO_COMP,
    PROBA_NIP,
    PROBA_NON_CO_COMP,
    PROBA_NON_NIP,
    PROTEIN,
    PROTEIN_U,
    PROTEIN_V,
    REL,
    STRING,
    TOPO,
    TOPO_L2,
    WEIGHT,
)

SCORE = WEIGHT
RESIDUAL = "RESIDUAL"


class CoCompClassifier:
    """
    Co-complex classifier.
    """

    def __init__(
        self,
        features: List[str],
        model: Union[CategoricalNB, RandomForestClassifier, MLPClassifier],
        name: str,
    ):
        self.features = features
        self.label = IS_CO_COMP
        self.model = model
        self.name = name

    def label_composite(self, df_composite: pl.DataFrame, df_train_pairs: pl.DataFrame):
        """
        Labels the PPIN subset.

        Args:
            df_composite (pl.DataFrame): _description_
            df_train_pairs (pl.DataFrame): _description_

        Returns:
            _type_: _description_
        """

        # all proteins in the training complexes
        srs_proteins = (
            df_train_pairs.select(pl.col(PROTEIN_U))
            .to_series()
            .append(df_train_pairs.select(pl.col(PROTEIN_V)).to_series())
            .unique()
            .alias(PROTEIN)
        )

        df_labeled = (
            df_composite.lazy()
            .filter(
                pl.col(PROTEIN_U).is_in(srs_proteins)
                & pl.col(PROTEIN_V).is_in(srs_proteins)
            )
            .join(
                df_train_pairs.lazy().with_columns(pl.lit(1).alias(self.label)),
                on=[PROTEIN_U, PROTEIN_V],
                how="left",
            )
            .fill_null(0)
            .collect()
        )

        return df_labeled

    def equalize_classes(
        self, df_labeled: pl.DataFrame, xval_iter: int
    ) -> pl.DataFrame:
        """
        Equalizes the size of the classes of the labeled set.

        Args:
            df_labeled (pl.DataFrame): _description_

        Returns:
            pl.DataFrame: _description_
        """

        df_positive = df_labeled.filter(pl.col(self.label) == 1)
        df_negative = df_labeled.filter(pl.col(self.label) == 0)

        if df_positive.shape[0] < df_negative.shape[0]:
            df_negative = df_negative.sample(df_positive.shape[0], seed=xval_iter)
        elif df_positive.shape[0] > df_negative.shape[0]:
            df_positive = df_positive.sample(df_negative.shape[0], seed=xval_iter)

        df_labeled = pl.concat([df_positive, df_negative], how="vertical")

        return df_labeled

    def weight(
        self, df_composite: pl.DataFrame, df_labeled: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Uses the learned parameters to weight each protein pair.

        Args:
            df_composite (pl.DataFrame): _description_
            df_labeled (pl.DataFrame): _description_

        Returns:
            pl.DataFrame: _description_
        """

        print(">>> Weighting")
        X_train = df_labeled.select(self.features).to_numpy()
        y_train = df_labeled.select(self.label).to_numpy().ravel()

        print("Training the model...")
        print(f"Number of training samples: {X_train.shape[0]}")
        self.model.fit(X_train, y_train)  # training the model

        # After learning the parameters, weight all protein pairs
        X_test = df_composite.select(self.features).to_numpy()
        ndarr_pred = self.model.predict_proba(X_test)

        CLASS_PROBA = [PROBA_NON_CO_COMP, PROBA_CO_COMP]

        df_weights = pl.from_numpy(
            ndarr_pred, schema=[CLASS_PROBA[c] for c in self.model.classes_]
        )

        df_w_composite = pl.concat([df_composite, df_weights], how="horizontal").select(
            [PROTEIN_U, PROTEIN_V, PROBA_NON_CO_COMP, PROBA_CO_COMP]
        )

        df_w_composite.write_csv(f"../data/training/{self.name}_probas.csv")

        return df_w_composite

    def residual(self, col_a: str, col_b: str) -> pl.Expr:
        return (pl.col(col_a) - pl.col(col_b)).abs().mean()

    def validate(
        self,
        df_w_composite: pl.DataFrame,
        df_train_pairs: pl.DataFrame,
        df_test_pairs: pl.DataFrame,
    ):
        print("Validating vs cross-val testing set...")

        lf_test = (
            df_w_composite.lazy()
            .select(pl.exclude(self.label))
            .join(
                df_train_pairs.lazy(),
                on=[PROTEIN_U, PROTEIN_V],
                how="anti",
            )
            .join(
                df_test_pairs.lazy().with_columns(pl.lit(1).alias(self.label)),
                on=[PROTEIN_U, PROTEIN_V],
                how="left",
            )
        )

        df_test_all = lf_test.fill_null(pl.lit(0)).collect()
        residual = df_test_all.select(
            self.residual(self.label, PROBA_CO_COMP).alias(RESIDUAL)
        )
        # print(residual)

        print("Compare with co-complex edges only")
        df_test_positive = lf_test.drop_nulls(subset=self.label).collect()
        residual = df_test_positive.select(
            self.residual(self.label, PROBA_CO_COMP).alias(RESIDUAL)
        )
        # print(residual)

        print("Validating vs the whole reference complexes set (actual class)...")
        df_cmp_pairs = pl.concat(
            [df_train_pairs, df_test_pairs], how="vertical"
        ).unique(maintain_order=True)

        lf = (
            df_w_composite.lazy()
            .select(pl.exclude(self.label))
            .join(
                df_cmp_pairs.lazy().with_columns(pl.lit(1).alias(self.label)),
                on=[PROTEIN_U, PROTEIN_V],
                how="left",
            )
        )

        df_all = lf.fill_null(pl.lit(0)).collect()
        residual = df_all.select(
            self.residual(self.label, PROBA_CO_COMP).alias(RESIDUAL)
        )
        # print(residual)

        print("Compare with co-complex edges only")

        df_positive = lf.drop_nulls(subset=self.label).collect()
        residual = df_positive.select(
            self.residual(self.label, PROBA_CO_COMP).alias(RESIDUAL)
        )
        print(residual)
