# pyright: basic

from typing import List, Optional, Set, Union

import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier, MLPRegressor

from aliases import (
    CO_EXP,
    CO_OCCUR,
    GO_BP,
    GO_CC,
    GO_MF,
    IS_CO_COMPLEX,
    PROBA_CO_COMP,
    PROBA_NON_CO_COMP,
    PROTEIN,
    PROTEIN_U,
    PROTEIN_V,
    REL,
    STRING,
    TOPO,
    TOPO_L2,
    WEIGHT,
)
from assertions import assert_prots_sorted
from utils import read_no_header_file, sort_prot_cols

SCORE = WEIGHT
CLASS_PRED = "CLASS_PRED"


class WeightingModel:
    def __init__(
        self,
        features: List[str],
        ppin: str,
        model: Union[
            MultinomialNB,
            GaussianNB,
            CategoricalNB,
            ComplementNB,
            MLPClassifier,
        ],
    ):
        self.features_ = features
        self.class_ = IS_CO_COMPLEX
        self.ppin_ = ppin
        self.model_ = model

    # def construct_composite_ppin(self) -> Optional[pl.DataFrame]:
    #     """
    #     Constructs the composite PPIN based on the selected features.

    #     Returns:
    #         Optional[pl.DataFrame]: _description_
    #     """

    #     swc_features = [TOPO, TOPO_L2, STRING, CO_OCCUR]

    #     df_swc_composite = pl.read_csv("../data/preprocessed/swc_composite_data.csv")

    #     assert_prots_sorted(df_swc_composite)

    #     if len(self.features_) == 0:
    #         return None

    #     scores_files = {
    #         REL: f"../data/scores/{self.ppin_}_rel.csv",
    #         CO_EXP: f"../data/scores/{self.ppin_}_co_exp.csv",
    #     }

    #     go_features: List[str] = []
    #     lf_features: List[pl.LazyFrame] = []
    #     for F in self.features_:
    #         if F in [GO_CC, GO_BP, GO_MF]:
    #             go_features.append(F)
    #         elif F in swc_features:
    #             lf_features.append(
    #                 df_swc_composite.select([PROTEIN_U, PROTEIN_V, F]).lazy()
    #             )
    #         else:
    #             lf_features.append(
    #                 pl.scan_csv(scores_files[F], has_header=True)
    #                 .with_columns(sort_prot_cols(PROTEIN_U, PROTEIN_V))
    #                 .select([PROTEIN_U, PROTEIN_V, F])
    #             )

    #     if len(go_features) > 0:
    #         lf_features.append(
    #             pl.scan_csv(
    #                 f"../data/scores/{self.ppin_}_go_ss.csv",
    #                 has_header=True,
    #                 null_values="None",
    #             )
    #             .with_columns(sort_prot_cols(PROTEIN_U, PROTEIN_V))
    #             .select([PROTEIN_U, PROTEIN_V, *go_features])
    #         )

    #     lf_composite_ppin = lf_features[0]

    #     for lf in lf_features[1:]:
    #         lf_composite_ppin = lf_composite_ppin.join(
    #             other=lf, on=[PROTEIN_U, PROTEIN_V], how="outer"
    #         )

    #     df_composite_ppin = lf_composite_ppin.fill_null(0.0).collect()

    #     return df_composite_ppin

    def label_ppin(
        self, df_ppin: pl.DataFrame, df_train_cmp_pairs: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Labels the PPIN subset.

        Args:
            df_ppin (pl.DataFrame): _description_
            df_train_cmp_pairs (pl.DataFrame): _description_

        Returns:
            pl.DataFrame: _description_
        """

        # all proteins in the training complexes
        srs_proteins = (
            df_train_cmp_pairs.select(pl.col(PROTEIN_U))
            .to_series()
            .append(df_train_cmp_pairs.select(pl.col(PROTEIN_V)).to_series())
            .unique()
            .alias(PROTEIN)
        )

        df_labeled = (
            df_ppin.lazy()
            .filter(
                pl.col(PROTEIN_U).is_in(srs_proteins)
                & pl.col(PROTEIN_V).is_in(srs_proteins)
            )
            .join(
                df_train_cmp_pairs.lazy().with_columns(pl.lit(True).alias(self.class_)),
                on=[PROTEIN_U, PROTEIN_V],
                how="left",
            )
            .fill_null(False)
            .collect()
        )

        return df_labeled

    def equalize_classes(self, df_labeled: pl.DataFrame) -> pl.DataFrame:
        """
        Equalizes the size of the classes of the labeled set.

        Args:
            df_labeled (pl.DataFrame): _description_

        Returns:
            pl.DataFrame: _description_
        """

        df_positive = df_labeled.filter(pl.col(self.class_) == True)
        df_negative = df_labeled.filter(pl.col(self.class_) == False)

        if df_positive.shape[0] < df_negative.shape[0]:
            df_negative = df_negative.sample(df_positive.shape[0], seed=12345)
        elif df_positive.shape[0] > df_negative.shape[0]:
            df_positive = df_positive.sample(df_negative.shape[0], seed=12345)

        df = pl.concat([df_positive, df_negative], how="vertical")

        return df

    def validate(self, df_w_ppin: pl.DataFrame):
        """
        Validates the performance of the model.

        Args:
            df_w_ppin (pl.DataFrame): Weighted and classified network.
        """

        print(">>> Testing the model...")

        df_test_cmp_pairs = read_no_header_file(
            "../data/preprocessed/test_cmp_pairs.csv", [PROTEIN_U, PROTEIN_V]
        ).with_columns(pl.lit(True).alias(self.class_))
        assert_prots_sorted(df_test_cmp_pairs)
        assert_prots_sorted(df_w_ppin)

        # all proteins in the testing complexes
        srs_proteins = (
            df_test_cmp_pairs.select(pl.col(PROTEIN_U))
            .to_series()
            .append(df_test_cmp_pairs.select(pl.col(PROTEIN_V)).to_series())
            .unique()
            .alias(PROTEIN)
        )

        df_test = (
            df_w_ppin.lazy()
            .filter(
                pl.col(PROTEIN_U).is_in(srs_proteins)
                & pl.col(PROTEIN_V).is_in(srs_proteins)
            )
            .select([PROTEIN_U, PROTEIN_V, CLASS_PRED])
            .join(df_test_cmp_pairs.lazy(), on=[PROTEIN_U, PROTEIN_V], how="left")
            .fill_null(False)
            .collect()
        )

        df_correct = df_test.filter(pl.col(CLASS_PRED) == pl.col(self.class_))

        print(">>> VALIDATION ACCURACY...")
        print(
            f"Testing Samples: {df_test.shape[0]} | Number of correctly classified: {df_correct.shape[0]}"
        )
        print(f"Accuracy: {round(df_correct.shape[0] * 100 / df_test.shape[0], 6)} %")

        print(">>> END OF VALIDATION -------------------")

    def weight(self, df_ppin: pl.DataFrame, df_labeled: pl.DataFrame) -> pl.DataFrame:
        """
        Uses the learned parameters to weight each PPI.

        Args:
            df_ppin (pl.DataFrame): Whole composite network.
            df_labeled (pl.DataFrame): Balanced training set.

        Returns:
            pl.DataFrame: Weighted composite network.
        """

        print(">>> Weighting")
        X_train = df_labeled.select(self.features_).to_numpy()
        y_train = df_labeled.select(self.class_).to_numpy().ravel()

        print("Training the model...")
        print(f"Number of samples: {X_train.shape[0]}")
        self.model_.fit(X_train, y_train)  # training the model

        X_test = df_ppin.select(self.features_).to_numpy()
        ndarr_pred = self.model_.predict_proba(X_test)
        ndarr_pred_class = self.model_.predict(X_test)

        df_weights = pl.from_numpy(
            ndarr_pred,
            schema=[
                PROBA_CO_COMP if c else PROBA_NON_CO_COMP
                for c in self.model_.classes_.tolist()
            ],
        )

        df_classes = pl.from_numpy(
            ndarr_pred_class,
            schema=[CLASS_PRED],
        )

        df_w_ppin = pl.concat(
            [df_ppin, df_weights, df_classes],
            how="horizontal",
        )

        print(df_w_ppin)

        df_w_ppin.write_csv(
            f"../data/training/{self.ppin_}_{self.model_.__class__.__name__}.csv"
        )

        # print(">>> All classified Co-complex edges...")
        # print(df_w_ppin.filter(pl.col(PROBA_CO_COMP) > 0.5))

        # self.validate(df_w_ppin)

        return df_w_ppin
