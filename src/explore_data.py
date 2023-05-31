# pyright: basic
from typing import List

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from aliases import (
    CO_EXP,
    CO_OCCUR,
    GO_BP,
    GO_CC,
    GO_MF,
    IS_CO_COMPLEX,
    IS_NIP_PAIR,
    PROTEIN,
    PROTEIN_U,
    PROTEIN_V,
    REL,
    STRING,
    TOPO,
    TOPO_L2,
)
from feature_preprocessor import FeaturePreprocessor
from utils import (
    construct_composite_ppin,
    get_all_cyc_complex_pairs,
    get_cyc_proteins,
    read_no_header_file,
)


class ExploratoryDataAnalysis:
    """
    Exploratory Data Analysis

    - [X] Heatmap of features correlation
    - [X] Distribution of each feature
    - [X] Explore co-complexes features properties
    - [X] Explore NIPs feature properties
    """

    def __init__(self, features: List[str]):
        sns.set_palette("deep")
        self.features = features
        self.df_ppin = construct_composite_ppin(features=self.features)

        self.feature_preprocessor = FeaturePreprocessor()

    def describe_features(self):
        print(self.df_ppin.select(self.features).describe())

    def features_heatmap(self):
        plt.figure()
        df_pd_ppin = self.df_ppin.to_pandas()
        df_corr_matrix = df_pd_ppin[self.features].corr()
        ax = sns.heatmap(
            df_corr_matrix,
            annot=True,
            vmin=-1,
            vmax=1,
            fmt=".2f",
            cmap="vlag",
        )
        ax.set_title("Correlation among the features.")

    def features_dist_hist(self):
        plt.figure()
        df_pd_ppin = self.df_ppin.to_pandas()
        ax = sns.histplot(
            data=df_pd_ppin[self.features],
            binwidth=0.05,
            stat="percent",
            element="step",
        )
        ax.set_title("Feature values distribution.")

    def explore_co_complexes(self):
        plt.figure()
        srs_cyc_prots = get_cyc_proteins()
        df_relevant = self.df_ppin.filter(
            pl.col(PROTEIN_U).is_in(srs_cyc_prots)
            & pl.col(PROTEIN_V).is_in(srs_cyc_prots)
        )

        df_cmp_pairs = get_all_cyc_complex_pairs().with_columns(
            pl.lit(True).alias(IS_CO_COMPLEX)
        )

        df_pd_display = (
            df_relevant.join(df_cmp_pairs, on=[PROTEIN_U, PROTEIN_V], how="left")
            .fill_null(pl.lit(False))
            .melt(
                id_vars=[PROTEIN_U, PROTEIN_V, IS_CO_COMPLEX],
                variable_name="FEATURE",
                value_name="VALUE",
            )
            .to_pandas()
        )

        ax = sns.barplot(data=df_pd_display, x="FEATURE", y="VALUE", hue=IS_CO_COMPLEX)
        ax.set_title(
            "Mean feature values of co-complex and non-co-complex protein pairs."
        )

    def explore_nip_pairs(self):
        plt.figure()
        df_nip_pairs = read_no_header_file(
            "../data/preprocessed/yeast_nips.csv", cols=[PROTEIN_U, PROTEIN_V]
        ).with_columns(pl.lit(True).alias(IS_NIP_PAIR))

        srs_nips = (
            df_nip_pairs.select(pl.col(PROTEIN_U).alias(PROTEIN))
            .to_series()
            .append(df_nip_pairs.select(pl.col(PROTEIN_V)).to_series())
            .unique()
        )

        df_relevant = self.df_ppin.filter(
            pl.col(PROTEIN_U).is_in(srs_nips) & pl.col(PROTEIN_V).is_in(srs_nips)
        )

        df_pd_display = (
            df_relevant.join(df_nip_pairs, on=[PROTEIN_U, PROTEIN_V], how="left")
            .fill_null(pl.lit(False))
            .melt(
                id_vars=[PROTEIN_U, PROTEIN_V, IS_NIP_PAIR],
                variable_name="FEATURE",
                value_name="VALUE",
            )
            .to_pandas()
        )

        ax = sns.barplot(data=df_pd_display, x="FEATURE", y="VALUE", hue=IS_NIP_PAIR)
        ax.set_title("Mean feature values of NIP and non-NIP pairs.")

    def main(self):
        self.describe_features()
        self.df_ppin = self.feature_preprocessor.normalize_features(
            self.df_ppin, self.features
        )
        self.describe_features()
        self.features_heatmap()
        self.features_dist_hist()
        self.explore_co_complexes()
        self.explore_nip_pairs()

        plt.show()


if __name__ == "__main__":
    pl.Config.set_tbl_cols(40)
    pl.Config.set_tbl_rows(7)

    features = [TOPO, TOPO_L2, STRING, CO_OCCUR, REL, CO_EXP, GO_CC, GO_BP, GO_MF]
    eda = ExploratoryDataAnalysis(features)
    eda.main()
