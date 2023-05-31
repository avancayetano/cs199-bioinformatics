# pyright: basic

from typing import List

import polars as pl
from imodels.discretization.mdlp import MDLPDiscretizer
from sklearn.preprocessing import MinMaxScaler

from aliases import PROTEIN_U, PROTEIN_V
from utils import get_all_cyc_complex_pairs, get_cyc_proteins


class FeaturePreprocessor:
    """
    Preprocesses features.

    - [ ] Normalizing features
    - [ ] Discretizing features
    """

    def normalize_features(
        self, df_ppin: pl.DataFrame, features: List[str]
    ) -> pl.DataFrame:
        scaler = MinMaxScaler()

        df_pd_pandas = df_ppin.to_pandas()
        ndarr_ppin = scaler.fit_transform(df_pd_pandas[features])
        df_ppin = pl.concat(
            [
                df_ppin.select(pl.exclude(features)),
                pl.from_numpy(ndarr_ppin, schema=features),
            ],
            how="horizontal",
        )

        return df_ppin

    def discretize_features(
        self, df_ppin: pl.DataFrame, features: List[str], class_label: str
    ):
        srs_cyc_proteins = get_cyc_proteins()
        df_co_complex_pairs = get_all_cyc_complex_pairs()

        df_pandas = (
            df_ppin.lazy()
            .filter(
                pl.col(PROTEIN_U).is_in(srs_cyc_proteins)
                & pl.col(PROTEIN_V).is_in(srs_cyc_proteins)
            )
            .join(
                df_co_complex_pairs.lazy().with_columns(
                    pl.lit(True).alias(class_label)
                ),
                on=[PROTEIN_U, PROTEIN_V],
                how="left",
            )
            .fill_null(False)
            .collect()
            .to_pandas()
        )

        MDLPDiscretizer(
            df_pandas,
            class_label=class_label,
            features=features,
            out_path_data="../data/ml_data/discretized_data.csv",
            out_path_bins="../data/ml_data/discretized_bins.csv",
        )
