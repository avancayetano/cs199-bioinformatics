# pyright: basic

from typing import List, Literal, Union

import polars as pl
from imodels.discretization.mdlp import MDLPDiscretizer
from sklearn.preprocessing import MinMaxScaler

from aliases import PROTEIN_U, PROTEIN_V
from utils import get_all_cyc_proteins, get_cyc_comp_pairs, get_unique_proteins


class ModelPreprocessor:
    """
    Collection of preprocessing methods for the machine learning models.

    - [X] Normalizing features
    - [ ] Discretizing features
    - [X] Labeling composite network
    """

    def normalize_features(
        self, df_composite: pl.DataFrame, features: List[str]
    ) -> pl.DataFrame:
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

    def discretize_features(
        self, df_composite: pl.DataFrame, features: List[str], class_label: str
    ):
        srs_cyc_proteins = get_all_cyc_proteins()
        df_co_comp_pairs = get_cyc_comp_pairs()

        df_pd_composite = (
            df_composite.lazy()
            .filter(
                pl.col(PROTEIN_U).is_in(srs_cyc_proteins)
                & pl.col(PROTEIN_V).is_in(srs_cyc_proteins)
            )
            .join(
                df_co_comp_pairs.lazy().with_columns(pl.lit(True).alias(class_label)),
                on=[PROTEIN_U, PROTEIN_V],
                how="left",
            )
            .fill_null(False)
            .collect()
            .to_pandas()
        )

        MDLPDiscretizer(
            df_pd_composite,
            class_label=class_label,
            features=features,
            out_path_data="../data/ml_data/discretized_data.csv",
            out_path_bins="../data/ml_data/discretized_bins.csv",
        )

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
        Labels the composite network.

        Args:
            df_composite (pl.DataFrame): _description_
            df_positive_pairs (pl.DataFrame): _description_
            seed (int): _description_. Defaults to 0.
            mode (Union[Literal['all'], Literal['subset']]): Defaults to "subset".
                all: from the entire network, label non-df_positive_pairs as 0
                subset: from network subset, label non-df_positive_pairs as 0
            balanced (bool): _description_. Defaults to True.

        Raises:
            Exception: _description_

        Returns:
            pl.DataFrame: _description_
        """

        df_labeled = (
            df_composite.lazy()
            .join(
                df_positive_pairs.lazy().with_columns(pl.lit(1).alias(label)),
                on=[PROTEIN_U, PROTEIN_V],
                how="left",
            )
            .fill_null(0)
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
        Balances the size of the labels of the labeled set.

        Args:
            df_labeled (pl.DataFrame): _description_

        Returns:
            pl.DataFrame: _description_
        """

        df_positive = df_labeled.filter(pl.col(label) == 1)
        df_negative = df_labeled.filter(pl.col(label) == 0)

        if df_positive.shape[0] < df_negative.shape[0]:
            df_negative = df_negative.sample(df_positive.shape[0], seed=seed)
        elif df_positive.shape[0] > df_negative.shape[0]:
            df_positive = df_positive.sample(df_negative.shape[0], seed=seed)

        df_labeled = pl.concat([df_positive, df_negative], how="vertical")

        return df_labeled
