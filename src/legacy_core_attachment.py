# pyright: basic

import polars as pl

from aliases import PROTEIN_U, PROTEIN_V, WEIGHT
from utils import get_clusters_list, get_weighted_filename


class CoreAttachment:
    def __init__(self):
        self.sv_methods = ["cnb", "mlp", "rf", "swc"]
        self.inflations = [4]
        self.n_iters = 2

        self.cache_data()

    def cache_data(self):
        # cache weighted networks
        sv_weighted = {}
        for m in self.sv_methods:
            sv_weighted[m] = {}
            for j in range(self.n_iters):
                sv_weighted[m][j] = pl.read_csv(
                    get_weighted_filename(m, True, f"_iter{j}"),
                    new_columns=[PROTEIN_U, PROTEIN_V, WEIGHT],
                    separator="\t",
                    has_header=False,
                )

        # cache clusters

        self.sv_weighted = sv_weighted

    def get_df_w(self, method: str):
        pass
