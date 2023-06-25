import polars as pl

from aliases import COMP_ID, COMP_PROTEINS, PROTEIN_U, PROTEIN_V, XVAL_ITER
from utils import construct_composite_network, get_all_cyc_complexes, get_cyc_comp_pairs

df_composite = construct_composite_network()
df_all_complexes = get_all_cyc_complexes()

df_cross_val = pl.read_csv("../data/preprocessed/cross_val_table.csv")

df_iter_9 = df_cross_val.filter(pl.col(f"{XVAL_ITER}_9") == "train").select(COMP_ID)

df_all_pairs = get_cyc_comp_pairs()
df_iter_pairs = get_cyc_comp_pairs(df_iter_9)
df_relevant = df_composite.join(df_iter_pairs, on=[PROTEIN_U, PROTEIN_V], how="inner")

df_all_relevant = df_composite.join(
    df_all_pairs, on=[PROTEIN_U, PROTEIN_V], how="inner"
)
print(df_all_relevant)


# print(f"{self.name}...")
# df_check = df_train_labeled.join(
#     df_w_composite,
#     on=[PROTEIN_U, PROTEIN_V],
#     how="left",
# ).select([PROTEIN_U, PROTEIN_V, PROBA_NON_CO_COMP, PROBA_CO_COMP, IS_CO_COMP])

# print(df_check.sample(fraction=1.0, shuffle=True))
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
        df_comp_pairs = pl.concat(
            [df_train_pairs, df_test_pairs], how="vertical"
        ).unique(maintain_order=True)

        lf = (
            df_w_composite.lazy()
            .select(pl.exclude(self.label))
            .join(
                df_comp_pairs.lazy().with_columns(pl.lit(1).alias(self.label)),
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