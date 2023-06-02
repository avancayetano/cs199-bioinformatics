# pyright: basic
from typing import List

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from aliases import FEATURES, IS_CO_COMP, IS_NIP, PROTEIN_U, PROTEIN_V
from model_preprocessor import ModelPreprocessor
from utils import construct_composite_network, get_cyc_comp_pairs


class ExploratoryDataAnalysis:
    """
    Exploratory Data Analysis.

    - [X] Heatmap of features correlation
    - [X] Distribution of each feature
    - [X] Explore co-complexes features properties
    - [X] Explore NIPs feature properties
    """

    def __init__(self, features: List[str]):
        sns.set_palette("deep")
        self.features = features
        self.df_composite = construct_composite_network(features=self.features)
        self.model_prep = ModelPreprocessor()

    def features_heatmap(self):
        plt.figure()
        df_pd_composite = self.df_composite.to_pandas()
        df_corr_matrix = df_pd_composite[self.features].corr()
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
        df_pd_composite = self.df_composite.to_pandas()
        ax = sns.histplot(
            data=df_pd_composite[self.features],
            binwidth=0.05,
            stat="percent",
            element="step",
        )
        ax.set_title("Feature values distribution.")

    def explore_co_complexes(self):
        plt.figure()
        label = IS_CO_COMP

        df_comp_pairs = get_cyc_comp_pairs()
        df_labeled = self.model_prep.label_composite(
            self.df_composite,
            df_comp_pairs,
            label,
        )

        df_pd_display = df_labeled.melt(
            id_vars=[PROTEIN_U, PROTEIN_V, label],
            variable_name="FEATURE",
            value_name="VALUE",
        ).to_pandas()

        ax = sns.barplot(data=df_pd_display, x="FEATURE", y="VALUE", hue=label)
        ax.set_title(
            "Mean feature values of co-complex and non-co-complex protein pairs."
        )

    def explore_nip_pairs(self):
        plt.figure()
        label = IS_NIP

        df_nip_pairs = pl.read_csv(
            "../data/preprocessed/yeast_nips.csv",
            has_header=False,
            new_columns=[PROTEIN_U, PROTEIN_V],
        )
        df_labeled = self.model_prep.label_composite(
            self.df_composite,
            df_nip_pairs,
            label,
            mode="all",
            balanced=False,
        )
        df_pd_display = df_labeled.melt(
            id_vars=[PROTEIN_U, PROTEIN_V, label],
            variable_name="FEATURE",
            value_name="VALUE",
        ).to_pandas()

        ax = sns.barplot(data=df_pd_display, x="FEATURE", y="VALUE", hue=label)
        ax.set_title("Mean feature values of NIP and non-NIP pairs.")

    def main(self):
        print(self.df_composite.select(self.features).describe())
        self.df_composite = self.model_prep.normalize_features(
            self.df_composite, self.features
        )
        print(self.df_composite.select(self.features).describe())
        self.features_heatmap()
        self.features_dist_hist()
        self.explore_co_complexes()
        self.explore_nip_pairs()
        plt.show()


if __name__ == "__main__":
    pl.Config.set_tbl_cols(40)
    pl.Config.set_tbl_rows(7)

    features = FEATURES
    eda = ExploratoryDataAnalysis(features)
    eda.main()
