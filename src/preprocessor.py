import time
from typing import Dict, List, Tuple, TypedDict

import polars as pl

from aliases import (
    CO_OCCUR,
    COMPLEX_ID,
    COMPLEX_PROTEINS,
    PROTEIN_U,
    PROTEIN_V,
    PUBMED,
    STRING,
    TOPO,
    TOPO_L2,
    XVAL_ITER,
)
from assertions import assert_prots_sorted
from utils import get_cyc_complexes, sort_prot_cols


class Preprocessor:
    """
    Preprocessor of data.
    """

    def preprocess_swc_composite(self) -> pl.DataFrame:
        """
        Preprocess the composite PPIN of SWC.

        Returns:
            pl.DataFrame: _description_
        """
        SCORE = "SCORE"
        TYPE = "TYPE"
        df_swc = (
            pl.scan_csv("../data/swc/data_yeast.txt", has_header=False, separator="\t")
            .rename(
                {
                    "column_1": PROTEIN_U,
                    "column_2": PROTEIN_V,
                    "column_3": TYPE,
                    "column_4": SCORE,
                }
            )
            .collect()
            .pivot(
                values=SCORE,
                index=[PROTEIN_U, PROTEIN_V],
                columns=TYPE,
                aggregate_function="first",
            )
            .rename(
                {"PPI": TOPO, "PPIL2": TOPO_L2, "STRING": STRING, "PUBMED": CO_OCCUR}
            )
            .fill_null(0.0)
        )

        assert_prots_sorted(df_swc)
        assert df_swc.select([PROTEIN_U, PROTEIN_V]).is_unique().all()

        return df_swc

    def get_ppin(self, df_swc: pl.DataFrame) -> pl.DataFrame:
        """
        Get the (L1) PPIN from SWC data.

        NOTE: Currently unused. Might delete later.

        Args:
            df_ppin (pl.DataFrame): _description_

        Returns:
            pl.DataFrame: _description_
        """

        df_ppin = (
            df_swc.lazy()
            .filter(pl.col(TOPO).is_not_null())
            .select([PROTEIN_U, PROTEIN_V])
            .collect()
        )

        return df_ppin

    def get_nips(self) -> pl.DataFrame:
        """
        Read Negatome 2.0 and map UniprotIDs to Systematic Name (KEGG).

        Args:
            version (str): _description_

        Returns:
            pl.DataFrame: _description_
        """

        df_mapping = (
            pl.scan_csv("../data/databases/negatome_kegg_mapped.txt", separator="\t")
            .filter(pl.col("To").str.starts_with("sce:"))
            .with_columns(pl.col("To").str.replace("sce:", ""))
        )

        df_nips = (
            (
                pl.scan_csv(
                    f"../data/databases/negatome_combined_stringent.txt",
                    has_header=False,
                    separator="\t",
                ).rename({"column_1": "u", "column_2": "v"})
            )
            .join(df_mapping, left_on="u", right_on="From", how="inner")
            .join(df_mapping, left_on="v", right_on="From", how="inner")
            .drop(["u", "v"])
            .rename({"To": "u", "To_right": "v"})
            .with_columns(sort_prot_cols("u", "v"))
            .select([PROTEIN_U, PROTEIN_V])
        ).collect()

        assert_prots_sorted(df_nips)

        return df_nips

    def generate_kfolds(
        self, k: int, list_large_complexes: List[int]
    ) -> Tuple[str, Dict[int, List[str]]]:
        """
        _summary_

        Args:
            k (int): _description_
            list_large_complexes (List[int]): _description_

        Returns:
            Tuple[str, Dict[int, List[str]]]: _description_
        """

        fold_size = round((1 / k) * len(list_large_complexes))

        output = ""
        xval: Dict[int, List[str]] = {}

        for i in range(k):
            output += f"iter\t{i}\n"
            start = i * fold_size
            end = (i + 1) * fold_size
            fold = list_large_complexes[start:end]
            if end >= len(list_large_complexes):
                fold += list_large_complexes[0 : end - len(list_large_complexes)]

            # testing set for this round...
            testing_set = list(
                filter(lambda complex_id: complex_id not in fold, list_large_complexes)
            )

            for complex_id in testing_set:
                output += f"{complex_id}\n"

                if complex_id in xval:
                    xval[complex_id].append(f"{XVAL_ITER}_{i}")
                else:
                    xval[complex_id] = [f"{XVAL_ITER}_{i}"]
        output = output.strip()

        return output, xval

    def xval_to_df(self, k: int, xval: Dict[int, List[str]]) -> pl.DataFrame:
        XvalDict = TypedDict(
            "XvalDict", {"COMPLEX_ID": List[int], "ITERS": List[List[str]]}
        )
        xval_dict: XvalDict = {COMPLEX_ID: [], "ITERS": []}

        for complex_id in xval:
            xval_dict[COMPLEX_ID].append(complex_id)
            xval_dict["ITERS"].append(xval[complex_id])

        df_xval = (
            pl.LazyFrame(xval_dict)
            .explode("ITERS")
            .with_columns(pl.lit("test").alias("VALUES"))
            .collect()
            .pivot(
                values="VALUES",
                index=COMPLEX_ID,
                columns="ITERS",
                aggregate_function="first",
            )
            .lazy()
            .join(
                df_complexes.lazy().select(pl.col(COMPLEX_ID)),
                on=COMPLEX_ID,
                how="outer",
            )
            .fill_null(pl.lit("train"))
            .sort(pl.col(COMPLEX_ID))
            .select([COMPLEX_ID] + list(sorted([f"{XVAL_ITER}_{i}" for i in range(k)])))
            .collect()
        )
        return df_xval

    def generate_xval_data(
        self, df_complexes: pl.DataFrame
    ) -> Tuple[str, pl.DataFrame]:
        """
        Generate 10 rounds (iterations) of 10-fold cross-validation data.
        In each round, 90% of complexes greater than 3 should be the testing set.
        The rest are the training set.

        Args:
            df_complexes (pl.DataFrame): _description_

        Returns:
            Tuple[str, pl.DataFrame]: _description_
        """

        list_large_complexes: List[int] = (
            df_complexes.filter(pl.col(COMPLEX_PROTEINS).arr.lengths() > 3)
            .select(pl.col(COMPLEX_ID))
            .to_series()
            .shuffle(seed=12345)
            .to_list()
        )

        k = 10  # 10 folds, 10 rounds
        output, xval = self.generate_kfolds(k, list_large_complexes)
        df_xval = self.xval_to_df(k, xval)

        return output, df_xval  # output is for the SWC software

    def read_irefindex(self) -> pl.DataFrame:
        physical_interactions = [
            'psi-mi:"MI:0914"(association)',
            'psi-mi:"MI:0915"(physical association)',
            'psi-mi:"MI:0407"(direct interaction)',
        ]
        df = (
            pl.scan_csv(
                "../data/databases/large/irefindex 559292 mitab26.txt", separator="\t"
            )
            .select(
                [
                    "altA",
                    "altB",
                    "author",
                    "pmids",
                    "taxa",
                    "taxb",
                    "interactionType",
                    "sourcedb",
                    "edgetype",
                ]
            )
            .filter(
                (pl.col("edgetype") == "X")
                & (pl.col("taxa").str.starts_with("taxid:559292"))
                & (pl.col("taxb").str.starts_with("taxid:559292"))
                & (pl.col("altA").str.starts_with("cygd:"))
                & (pl.col("altB").str.starts_with("cygd:"))
                & (pl.col("interactionType").is_in(physical_interactions))
            )
            .with_columns(
                [
                    pl.col("altA").str.extract(r"cygd:([a-zA-Z0-9]+)").alias(PROTEIN_U),
                    pl.col("altB").str.extract(r"cygd:([a-zA-Z0-9]+)").alias(PROTEIN_V),
                    pl.col("pmids").str.extract(r"pubmed:(\d+)$").alias(PUBMED),
                    # pl.col("author").str.extract(r"([12][0-9]{3})").alias("YEAR"),
                ]
            )
            # .filter(pl.col("YEAR") < "2012")
            .select([PROTEIN_U, PROTEIN_V, PUBMED])
            .with_columns(sort_prot_cols(PROTEIN_U, PROTEIN_V))
            .unique(keep="first")
            .collect()
        )

        return df


if __name__ == "__main__":
    start_time = time.time()

    pl.Config.set_tbl_cols(40)
    pl.Config.set_tbl_rows(5)

    preprocessor = Preprocessor()
    df_nips = preprocessor.get_nips()
    df_swc = preprocessor.preprocess_swc_composite()
    df_irefindex = preprocessor.read_irefindex()

    df_complexes = get_cyc_complexes()
    output, df_xval = preprocessor.generate_xval_data(df_complexes)

    print(">>> NIPS")
    print(df_nips)

    print(">>> SWC DATA - COMPOSITE PROTEIN NETWORK")
    print(df_swc)

    print(">>> IREFINDEX")
    print(df_irefindex)

    print(">>> COMPLEXES")
    print(df_complexes)

    print(">>> XVAL DATA")
    print(df_xval)

    # Write files...
    df_nips.write_csv("../data/preprocessed/yeast_nips.csv", has_header=False)

    df_swc.write_csv("../data/preprocessed/swc_composite_data.csv", has_header=True)

    df_swc.select([PROTEIN_U, PROTEIN_V]).write_csv(
        "../data/preprocessed/swc_composite_pairs.csv", has_header=False
    )

    df_irefindex.write_csv("../data/preprocessed/irefindex_pubmed.csv", has_header=True)

    with open("../data/preprocessed/cross_val.csv", "w") as file:
        file.write(output)
    df_xval.write_csv("../data/preprocessed/cross_val_table.csv", has_header=True)

    print(">>> Execution Time")
    print(time.time() - start_time)
