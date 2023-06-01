import polars as pl

from aliases import COMPLEX_ID, COMPLEX_PROTEINS, CROSS_VAL_ITER, PROTEIN_U, PROTEIN_V
from utils import (
    construct_composite_network,
    get_all_cyc_complexes,
    get_cyc_complex_pairs,
)

df_composite = construct_composite_network()
df_all_complexes = get_all_cyc_complexes()

df_cross_val = pl.read_csv("../data/preprocessed/cross_val_table.csv")

df_iter_9 = df_cross_val.filter(pl.col(f"{CROSS_VAL_ITER}_9") == "train").select(
    COMPLEX_ID
)

df_all_pairs = get_cyc_complex_pairs()
df_iter_pairs = get_cyc_complex_pairs(df_iter_9)
df_relevant = df_composite.join(df_iter_pairs, on=[PROTEIN_U, PROTEIN_V], how="inner")

df_all_relevant = df_composite.join(
    df_all_pairs, on=[PROTEIN_U, PROTEIN_V], how="inner"
)
print(df_all_relevant)
