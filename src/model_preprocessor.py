# pyright: basic

from typing import Dict, List, Literal, Tuple, Union

import polars as pl
from imodels.discretization.mdlp import MDLPDiscretizer
from sklearn.preprocessing import MinMaxScaler

from aliases import PROTEIN_U, PROTEIN_V
from assertions import assert_df_normalized
from utils import get_unique_proteins


class ModelPreprocessor:
    """
    Collection of preprocessing methods for the machine learning models.

    - [X] Normalizing features
    - [X] Discretizing features
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

    def learn_discretization(
        self,
        df_composite: pl.DataFrame,
        df_labeled: pl.DataFrame,
        features: List[str],
        label: str,
        xval_iter: int,
    ):
        df_feat_label = df_labeled.join(
            df_composite, on=[PROTEIN_U, PROTEIN_V], how="left"
        )
        MDLPDiscretizer(
            df_feat_label.to_pandas(),
            class_label=label,
            features=features,
            out_path_data=f"../data/training/discretized_data_iter{xval_iter}.csv",
            out_path_bins=f"../data/training/discretized_bins_iter{xval_iter}.csv",
        )

    def discretize_composite(
        self,
        df_composite: pl.DataFrame,
        df_labeled: pl.DataFrame,
        features: List[str],
        label: str,
        xval_iter: int,
    ) -> pl.DataFrame:
        print()
        print("Discretizing the features (only for models that need discrete values)")
        self.learn_discretization(df_composite, df_labeled, features, label, xval_iter)
        cuts, selected_feats = self.get_cuts(xval_iter)

        df_composite_binned = df_composite.select(
            [PROTEIN_U, PROTEIN_V, *selected_feats]
        )
        for f in selected_feats:
            cut_labels = ["0"] + [str(int(idx + 1)) for idx, _ in enumerate(cuts[f])]

            df_cut = (
                df_composite.select(pl.col(f))
                .to_series()
                .unique()
                .cut(cuts[f], labels=cut_labels)
                .select(
                    [pl.col(f), pl.col("category").cast(pl.UInt64).alias(f"{f}_BINNED")]
                )
            )

            df_composite_binned = df_composite_binned.join(df_cut, on=f, how="left")

        df_composite_binned = df_composite_binned.select(
            pl.exclude(selected_feats)
        ).rename({f"{f}_BINNED": f for f in selected_feats})

        removed_feats = list(filter(lambda f: f not in selected_feats, features))
        print(f"MDLP Discretization done! Removed features: {removed_feats}")
        print()

        df_composite_binned = self.normalize_features(
            df_composite_binned, selected_feats
        )
        assert_df_normalized(df_composite_binned, selected_feats)

        return df_composite_binned

    def get_cuts(self, xval_iter: int) -> Tuple[Dict[str, List[float]], List[str]]:
        bins: Dict[str, List[float]] = {}
        feature = ""
        selected_feats: List[str] = []
        with open(f"../data/training/discretized_bins_iter{xval_iter}.csv") as file:
            lines = file.readlines()[1:]
            for line in lines:
                line = line.strip()
                if line.startswith("attr: "):
                    feature = line.replace("attr: ", "")
                    continue
                elif line.startswith("-inf"):
                    cuts = [
                        float(interval.split("_to_")[1])
                        for interval in line.split(", ")
                    ][:-1]
                    selected_feats.append(feature)
                    bins[feature] = cuts
                elif line.startswith("All"):
                    bins[feature] = []

        return bins, selected_feats

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
