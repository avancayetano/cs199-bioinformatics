# pyright: basic

import os
from typing import Dict, List, Literal, Tuple, Union

import polars as pl
from sklearn.preprocessing import MinMaxScaler

from aliases import PROTEIN_U, PROTEIN_V
from assertions import assert_df_bounded
from utils import get_unique_proteins


class ModelPreprocessor:
    """
    Collection of preprocessing methods for the machine learning models.

    - [X] Normalizing features
    - [X] Labeling composite network
    """

    def normalize_features(
        self, df_composite: pl.DataFrame, features: List[str]
    ) -> pl.DataFrame:
        """
        Normalizes features. NOTE: Not used in this study.

        Args:
            df_composite (pl.DataFrame): Composite network.
            features (List[str]): Features to normalize.

        Returns:
            pl.DataFrame: Normalized composite network.
        """

        scaler = MinMaxScaler()

        df_pd_composite = df_composite.to_pandas()
        ndarr_ppin = scaler.fit_transform(df_pd_composite[features])
        df_composite = pl.concat(
            [
                df_composite.select(pl.exclude(features)),
                pl.from_numpy(ndarr_ppin, schema=features),
            ],
            how="horizontal",
        )

        return df_composite

    def label_composite(
        self,
        df_composite: pl.DataFrame,
        df_positive_pairs: pl.DataFrame,
        label: str,
        seed: int = 0,
        mode: Union[Literal["all"], Literal["subset"]] = "subset",
        balanced: bool = True,
    ) -> pl.DataFrame:
        """
        Labels the composite network. The output dataframe has the following
        columns: [PROTEIN_U, PROTEIN_V, LABEL].

        Args:
            df_composite (pl.DataFrame): Composite network.
            df_positive_pairs (pl.DataFrame): Positive (co-complex) pairs.
            label (str): Set to IS_CO_COMP.
            seed (int, optional): Random seed. Uses the xval_iter number. Defaults to 0.
            mode (Union[Literal['all'], Literal['subset']]): Mode (see below). Defaults to "subset".
                all: from the entire network, label non-positive pairs as 0
                subset: from network subset, label non-positive pairs as 0
            balanced (bool, optional): Whether to balance negative and positive classes. Defaults to True.

        Raises:
            Exception: Invalid mode.

        Returns:
            pl.DataFrame: Labeled composite network.
        """
        df_labeled = (
            df_composite.lazy()
            .join(
                df_positive_pairs.lazy().with_columns(pl.lit(1).alias(label)),
                on=[PROTEIN_U, PROTEIN_V],
                how="left",
            )
            .fill_null(0)
            .select([PROTEIN_U, PROTEIN_V, label])
            .collect()
        )

        if mode == "all":
            pass
        elif mode == "subset":
            srs_proteins = get_unique_proteins(df_positive_pairs)

            df_labeled = df_labeled.filter(
                pl.col(PROTEIN_U).is_in(srs_proteins)
                & pl.col(PROTEIN_V).is_in(srs_proteins)
            )
        else:
            raise Exception("Invalid mode")

        if balanced:
            df_labeled = self.balance_labels(df_labeled, label, seed)

        return df_labeled

    def balance_labels(
        self, df_labeled: pl.DataFrame, label: str, seed: int
    ) -> pl.DataFrame:
        """
        Balances the classes of the labeled dataset.

        Args:
            df_labeled (pl.DataFrame): Labeled dataset.
            label (str): Set to IS_CO_COMP.
            seed (int): Random seed.

        Returns:
            pl.DataFrame: Balanced version of df_labeled.
        """
        df_positive = df_labeled.filter(pl.col(label) == 1)
        df_negative = df_labeled.filter(pl.col(label) == 0)

        if df_positive.shape[0] < df_negative.shape[0]:
            df_negative = df_negative.sample(df_positive.shape[0], seed=seed)
        elif df_positive.shape[0] > df_negative.shape[0]:
            df_positive = df_positive.sample(df_negative.shape[0], seed=seed)

        df_labeled = pl.concat([df_positive, df_negative], how="vertical")

        return df_labeled
