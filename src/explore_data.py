# pyright: basic
from typing import List

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from aliases import (
    CO_OCCUR,
    COMP_ID,
    FEATURES,
    IS_CO_COMP,
    IS_NIP,
    PROTEIN,
    PROTEIN_U,
    PROTEIN_V,
    STRING,
    SWC_FEATS,
    TOPO,
    TOPO_L2,
    XVAL_ITER,
)
from model_preprocessor import ModelPreprocessor
from utils import (
    construct_composite_network,
    get_all_cyc_complexes,
    get_clusters_list,
    get_complexes_list,
    get_cyc_comp_pairs,
    get_unique_proteins,
)


class ExploratoryDataAnalysis:
    """
    Exploratory Data Analysis and some scratch/explorations I did.
    This is undocumented, but there is no need to read this.

    - [X] Heatmap of features correlation
    - [X] Distribution of each feature
    - [X] Explore co-complexes features properties
    - [X] Explore NIPs feature properties
    """

    def __init__(self, features: List[str]):
        sns.set_palette("deep")
        self.features = features
        self.df_composite = construct_composite_network(
            features=self.features, dip=False
        )
        self.df_dip_composite = construct_composite_network(
            features=self.features, dip=True
        )
        self.cyc_comp_pairs = get_cyc_comp_pairs()
        self.model_prep = ModelPreprocessor()

    def check_cross_val(self):
        df_cross_val = pl.read_csv("../data/preprocessed/cross_val_table.csv")
        df_dip_cross_val = pl.read_csv("../data/preprocessed/dip_cross_val_table.csv")

        df = pl.DataFrame()
        df_dip = pl.DataFrame()

        for xval_iter in range(10):
            test_complexes = get_complexes_list(xval_iter, "test", False)
            for cmp in test_complexes:
                if len(cmp) <= 3:
                    assert "ERROR"
            dip_test_complexes = get_complexes_list(xval_iter, "test", True)
            for cmp in dip_test_complexes:
                if len(cmp) <= 3:
                    assert "ERROR"

            # check if all test complexes are covered
            df_complex_ids = df_cross_val.filter(
                pl.col(f"{XVAL_ITER}_{xval_iter}") == "test"
            ).select(COMP_ID)

            df_dip_complex_ids = df_dip_cross_val.filter(
                pl.col(f"{XVAL_ITER}_{xval_iter}") == "test"
            ).select(COMP_ID)
            print(df_complex_ids.shape[0])
            print(df_dip_complex_ids.shape[0])
            df = pl.concat([df, df_complex_ids])
            df_dip = pl.concat([df_dip, df_dip_complex_ids])

        df = df.groupby(pl.col(COMP_ID)).count()
        df_dip = df_dip.groupby(pl.col(COMP_ID)).count()

        print(df.filter(pl.col("count") != 9))
        print(df_dip.filter(pl.col("count") != 9))
        print("Correct!!!")

    def explore_absent_cocomp_edges(self):
        print(self.cyc_comp_pairs.shape[0])
        print(
            self.df_composite.join(
                self.cyc_comp_pairs, on=[PROTEIN_U, PROTEIN_V], how="inner"
            ).shape[0]
        )
        print(
            self.df_dip_composite.join(
                self.cyc_comp_pairs, on=[PROTEIN_U, PROTEIN_V], how="inner"
            ).shape[0]
        )

    def explore_complexes_clusters(self):
        pass

    def explore_l2_pairs(self):
        """
        High l2 scores


        """

    def explore_perfect_l2(self):
        """
        Perfect l2:
        YBR009C,YNL030W
        YBR010W,YNL031C
        YBR118W,YPR080W
        """

        df_ppin = pl.read_csv(
            "../data/preprocessed/dip_ppin.csv", new_columns=[PROTEIN_U, PROTEIN_V]
        )

        perfect_pairs = [
            ("YBR009C", "YNL030W"),
            ("YBR010W", "YNL031C"),
            ("YBR118W", "YPR080W"),
        ]

        for p1, p2 in perfect_pairs:
            srs1 = (
                df_ppin.filter((pl.col(PROTEIN_U) == p1) | (pl.col(PROTEIN_V) == p1))
                .select(
                    pl.when(pl.col(PROTEIN_U) == p1)
                    .then(pl.col(PROTEIN_V))
                    .otherwise(pl.col(PROTEIN_U))
                )
                .to_series()
                .sort()
            )
            srs2 = (
                df_ppin.filter((pl.col(PROTEIN_U) == p2) | (pl.col(PROTEIN_V) == p2))
                .select(
                    pl.when(pl.col(PROTEIN_U) == p2)
                    .then(pl.col(PROTEIN_V))
                    .otherwise(pl.col(PROTEIN_U))
                )
                .to_series()
                .sort()
            )

            print(srs1.series_equal(srs2))

    def num_co_comp_pairs(self):
        n_total_co_comp_pairs = 11_923
        df_co_comp_pairs = get_cyc_comp_pairs()
        n_unique_co_comp_pairs = df_co_comp_pairs.shape[0]
        n_unique_proteins = get_unique_proteins(df_co_comp_pairs).shape[0]
        print(f"n_total_co_comp_pairs = {n_total_co_comp_pairs}")
        print(f"n_unique_co_comp_pairs = {n_unique_co_comp_pairs}")
        print(f"n_unique_proteins = {n_unique_proteins}")
        print(f"Total possible pairs: {n_unique_proteins ** 2}")
        df_labeled = self.model_prep.label_composite(
            self.df_composite, df_co_comp_pairs, IS_CO_COMP, 0, "subset", False
        )
        print(f"Possible pairs in the network: {df_labeled.shape[0]}")
        print(f"Repeated pairs: {n_total_co_comp_pairs - n_unique_co_comp_pairs}")

    def features_heatmap(self):
        df_pd_composite = self.df_composite.to_pandas()
        df_corr_matrix = df_pd_composite[self.features].corr()

        plt.figure()
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
        df_pd_composite = self.df_composite.to_pandas()

        plt.figure()
        ax = sns.histplot(
            data=df_pd_composite[self.features],
            binwidth=0.05,
            stat="percent",
            element="step",
        )
        ax.set_title("Feature values distribution.")

    def explore_co_complexes(self):
        label = IS_CO_COMP

        df_comp_pairs = get_cyc_comp_pairs()
        df_labeled_all = self.model_prep.label_composite(
            self.df_composite, df_comp_pairs, label, mode="all", balanced=False
        )
        df_labeled_subset = self.model_prep.label_composite(
            self.df_composite, df_comp_pairs, label, mode="subset", balanced=False
        )

        self.co_vs_non_co_comp(df_labeled_all, label, "all")
        self.co_vs_non_co_comp(df_labeled_subset, label, "subset")

    def co_vs_non_co_comp(self, df_labeled: pl.DataFrame, label: str, mode: str):
        df_feat_labeled = self.df_composite.join(df_labeled, on=[PROTEIN_U, PROTEIN_V])
        df_pd_display = df_feat_labeled.melt(
            id_vars=[PROTEIN_U, PROTEIN_V, label],
            variable_name="FEATURE",
            value_name="VALUE",
        ).to_pandas()

        plt.figure()
        ax = sns.barplot(data=df_pd_display, x="FEATURE", y="VALUE", hue=label)
        ax.set_title(
            f"({mode}) Mean feature values of co-complex and non-co-complex protein pairs."
        )

    def explore_nip_pairs(self):
        label = IS_NIP

        df_nip_pairs = pl.read_csv(
            "../data/preprocessed/yeast_nips.csv",
            has_header=False,
            new_columns=[PROTEIN_U, PROTEIN_V],
        )
        df_ppin = self.df_composite.filter(pl.col(TOPO) > 0)
        df_labeled = self.model_prep.label_composite(
            df_ppin,
            df_nip_pairs,
            label,
            mode="all",
            balanced=False,
        )

        df_feat_labeled = self.df_composite.join(
            df_labeled, on=[PROTEIN_U, PROTEIN_V], how="inner"
        )
        df_pd_display = df_feat_labeled.melt(
            id_vars=[PROTEIN_U, PROTEIN_V, label],
            variable_name="FEATURE",
            value_name="VALUE",
        ).to_pandas()

        plt.figure()
        ax = sns.barplot(data=df_pd_display, x="FEATURE", y="VALUE", hue=label)
        ax.set_title("Mean feature values of NIP and PPI pairs.")

    def nip_co_comp_intersection(self):
        PAIR_TYPE = "PAIR_TYPE"

        df_pair_types = self.df_composite.with_columns(
            pl.lit("NEITHER").alias(PAIR_TYPE)
        )

        df_nip_pairs = pl.read_csv(
            "../data/preprocessed/yeast_nips.csv",
            has_header=False,
            new_columns=[PROTEIN_U, PROTEIN_V],
        ).with_columns(pl.lit("NIP").alias(PAIR_TYPE))

        df_pair_types = (
            df_pair_types.join(df_nip_pairs, on=[PROTEIN_U, PROTEIN_V], how="left")
            .with_columns(
                pl.when(pl.col(f"{PAIR_TYPE}_right").is_not_null())
                .then(pl.col(f"{PAIR_TYPE}_right"))
                .otherwise(pl.col(PAIR_TYPE))
                .alias(PAIR_TYPE)
            )
            .drop(f"{PAIR_TYPE}_right")
        )

        df_comp_pairs = get_cyc_comp_pairs().with_columns(
            pl.lit("CO_COMP").alias(PAIR_TYPE)
        )

        df_pair_types = (
            df_pair_types.join(df_comp_pairs, on=[PROTEIN_U, PROTEIN_V], how="left")
            .with_columns(
                pl.when(pl.col(f"{PAIR_TYPE}_right").is_not_null())
                .then(pl.col(f"{PAIR_TYPE}_right"))
                .otherwise(pl.col(PAIR_TYPE))
                .alias(PAIR_TYPE)
            )
            .drop(f"{PAIR_TYPE}_right")
        )

        df_nip_co_comp = (
            df_nip_pairs.drop(PAIR_TYPE)
            .join(df_comp_pairs.drop(PAIR_TYPE), on=[PROTEIN_U, PROTEIN_V], how="inner")
            .with_columns(pl.lit("BOTH").alias(PAIR_TYPE))
        )

        df_pair_types = (
            df_pair_types.join(df_nip_co_comp, on=[PROTEIN_U, PROTEIN_V], how="left")
            .with_columns(
                pl.when(pl.col(f"{PAIR_TYPE}_right").is_not_null())
                .then(pl.col(f"{PAIR_TYPE}_right"))
                .otherwise(pl.col(PAIR_TYPE))
                .alias(PAIR_TYPE)
            )
            .drop(f"{PAIR_TYPE}_right")
        )

        df_pd_display = df_pair_types.melt(
            id_vars=[PROTEIN_U, PROTEIN_V, PAIR_TYPE],
            variable_name="FEATURE",
            value_name="VALUE",
        ).to_pandas()

        plt.figure()
        ax = sns.barplot(data=df_pd_display, x="FEATURE", y="VALUE", hue=PAIR_TYPE)
        ax.set_title("Mean feature values of pair type groups.")

    def explore_co_comp_pairs(self):
        INTERACTING = "INTERACTING"
        df_complex_pairs = get_cyc_comp_pairs()
        df_ppin = (
            self.df_composite.filter(pl.col(TOPO) > 0)
            .select([PROTEIN_U, PROTEIN_V])
            .with_columns(pl.lit(True).alias(INTERACTING))
        )

        df_ppi_pairs = df_complex_pairs.join(
            df_ppin, on=[PROTEIN_U, PROTEIN_V], how="left"
        ).fill_null(pl.lit(False))

        n_ppi = df_ppi_pairs.filter(pl.col(INTERACTING) == True).shape[0]
        n_non_ppi = df_ppi_pairs.shape[0] - n_ppi

        df_pd_display = (
            self.df_composite.join(df_ppi_pairs, on=[PROTEIN_U, PROTEIN_V], how="inner")
            .melt(
                id_vars=[PROTEIN_U, PROTEIN_V, INTERACTING],
                variable_name="FEATURE",
                value_name="VALUE",
            )
            .to_pandas()
        )
        plt.figure()
        ax = sns.barplot(data=df_pd_display, x="FEATURE", y="VALUE", hue=INTERACTING)
        ax.set_title(
            f"Mean feature values of CYC2008 complex pairs.\n Non-interacting: {n_ppi} | Interacting: {n_non_ppi}"
        )

    def explore_swc_features(self):
        print(self.df_composite)
        df = self.df_composite.select(SWC_FEATS).filter(
            (pl.col(TOPO_L2) > 0)
            & (pl.col(TOPO) == 0)
            & (pl.col(CO_OCCUR) == 0)
            & (pl.col(STRING) == 0)
        )
        print(df)

    def main(self):
        print(self.df_composite)
        # self.df_composite = self.model_prep.normalize_features(
        #     self.df_composite, self.features
        # )
        # print(self.df_composite.select(self.features).describe())

        # self.num_co_comp_pairs()
        # self.features_heatmap()
        # self.features_dist_hist()
        # self.explore_co_complexes()
        # self.explore_nip_pairs()

        # self.nip_co_comp_intersection()
        # self.explore_co_comp_pairs()
        # plt.show()
        # self.explore_swc_features()
        # self.explore_perfect_l2()
        # self.explore_absent_cocomp_edges()
        self.check_cross_val()
        plt.show()


if __name__ == "__main__":
    pl.Config.set_tbl_cols(40)

    features = FEATURES
    eda = ExploratoryDataAnalysis(features)
    eda.main()
