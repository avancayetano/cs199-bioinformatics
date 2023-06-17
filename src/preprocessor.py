import os
import time
from typing import Dict, List, Tuple, TypedDict

import polars as pl

from aliases import (
    CO_OCCUR,
    COMP_ID,
    COMP_PROTEINS,
    PROTEIN,
    PROTEIN_U,
    PROTEIN_V,
    PUBMED,
    STRING,
    TOPO,
    TOPO_L2,
    XVAL_ITER,
)
from assertions import assert_prots_sorted
from utils import get_all_cyc_complexes, sort_prot_cols


class Preprocessor:
    """
    Preprocessor of data.
    """

    def read_swc_composite_network(self) -> pl.DataFrame:
        """
        Preprocess the composite protein network of SWC.

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

    def write_dip_proteins(self, df_dip_ppin_uniprot: pl.DataFrame) -> None:
        if not os.path.exists("../data/databases/dip_uniprot_kegg_mapped.tsv"):
            df_dip_uniprot_ids = (
                df_dip_ppin_uniprot.melt(variable_name="PROTEIN_X", value_name=PROTEIN)
                .select(PROTEIN)
                .unique()
            )
            df_dip_uniprot_ids.write_csv(
                "../data/databases/dip_uniprot_ids.csv", has_header=False
            )

    def read_dip_ppin_uniprot(self) -> pl.DataFrame:
        INTERACTOR_A = "ID interactor A"
        INTERACTOR_B = "ID interactor B"
        df_dip_ppin_uniprot = (
            pl.scan_csv(
                "../data/databases/Scere20170205.txt",
                has_header=True,
                separator="\t",
                null_values="-",
            )
            .filter(
                pl.col("Taxid interactor A").str.contains("taxid:4932")
                & pl.col("Taxid interactor B").str.contains("taxid:4932")
            )
            .select(
                [
                    pl.col(INTERACTOR_A).str.extract(r".+uniprotkb:(.+)"),
                    pl.col(INTERACTOR_B).str.extract(r".+uniprotkb:(.+)"),
                ]
            )
            .filter(
                (pl.col(INTERACTOR_A) != pl.col(INTERACTOR_B))
                & (pl.col(INTERACTOR_A).str.lengths() > 0)
                & (pl.col(INTERACTOR_B).str.lengths() > 0)
            )
            .unique(maintain_order=True)
            .collect()
        )

        self.write_dip_proteins(df_dip_ppin_uniprot)

        return df_dip_ppin_uniprot

    def map_dip_ppin_uniprot_to_kegg(
        self, df_dip_ppin_uniprot: pl.DataFrame
    ) -> pl.DataFrame:
        FROM = "From"
        TO = "To"
        INTERACTOR_A = "ID interactor A"
        INTERACTOR_B = "ID interactor B"

        if os.path.exists("../data/databases/dip_uniprot_kegg_mapped.tsv"):
            lf_mapping = (
                pl.scan_csv(
                    "../data/databases/dip_uniprot_kegg_mapped.tsv",
                    has_header=True,
                    separator="\t",
                )
                .select([pl.col(FROM), pl.col(TO).str.extract(r"sce:(.+)")])
                .filter(pl.col(TO).str.lengths() > 0)
                .unique()
            )

            df_dip_ppin = (
                df_dip_ppin_uniprot.lazy()
                .join(lf_mapping, left_on=INTERACTOR_A, right_on=FROM, how="inner")
                .join(lf_mapping, left_on=INTERACTOR_B, right_on=FROM, how="inner")
                .rename({"To": PROTEIN_U, "To_right": PROTEIN_V})
                .with_columns(sort_prot_cols(PROTEIN_U, PROTEIN_V))
                .select([PROTEIN_U, PROTEIN_V])
                .unique(maintain_order=True)
                .collect()
            )

            return df_dip_ppin
        else:
            print("Please map the DIP uniprot IDs to KEGG ID, then rerun this script")
            return pl.DataFrame()

    def get_nips(self) -> pl.DataFrame:
        """
        Read Negatome 2.0 and map UniprotIDs to Systematic Name (KEGG).

        Args:
            version (str): _description_

        Returns:
            pl.DataFrame: _description_
        """

        lf_mapping = (
            pl.scan_csv("../data/databases/negatome_kegg_mapped.txt", separator="\t")
            .filter(pl.col("To").str.starts_with("sce:"))
            .with_columns(pl.col("To").str.replace("sce:", ""))
        )

        df_nips = (
            pl.scan_csv(
                f"../data/databases/negatome_combined_stringent.txt",
                has_header=False,
                separator="\t",
            )
            .rename({"column_1": "u", "column_2": "v"})
            .join(lf_mapping, left_on="u", right_on="From", how="inner")
            .join(lf_mapping, left_on="v", right_on="From", how="inner")
            .drop(["u", "v"])
            .rename({"To": "u", "To_right": "v"})
            .with_columns(sort_prot_cols("u", "v"))
            .select([PROTEIN_U, PROTEIN_V])
            .unique(maintain_order=True)
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
        cross_val: Dict[int, List[str]] = {}

        for i in range(k):
            output += f"iter\t{i}\n"
            start = i * fold_size
            end = (i + 1) * fold_size
            fold = list_large_complexes[start:end]
            if end >= len(list_large_complexes):
                fold += list_large_complexes[0 : end - len(list_large_complexes)]

            # testing dataset for this round...
            testing_set = list(
                filter(lambda complex_id: complex_id not in fold, list_large_complexes)
            )

            for complex_id in testing_set:
                output += f"{complex_id}\n"

                if complex_id in cross_val:
                    cross_val[complex_id].append(f"{XVAL_ITER}_{i}")
                else:
                    cross_val[complex_id] = [f"{XVAL_ITER}_{i}"]
        output = output.strip()

        return output, cross_val

    def cross_val_to_df(self, k: int, cross_val: Dict[int, List[str]]) -> pl.DataFrame:
        CrossValDict = TypedDict(
            "CrossValDict", {"COMP_ID": List[int], "ITERS": List[List[str]]}
        )
        cross_val_dict: CrossValDict = {COMP_ID: [], "ITERS": []}

        for complex_id in cross_val:
            cross_val_dict[COMP_ID].append(complex_id)
            cross_val_dict["ITERS"].append(cross_val[complex_id])

        df_cross_val = (
            pl.LazyFrame(cross_val_dict)
            .explode("ITERS")
            .with_columns(pl.lit("test").alias("VALUES"))
            .collect()
            .pivot(
                values="VALUES",
                index=COMP_ID,
                columns="ITERS",
                aggregate_function="first",
            )
            .lazy()
            .join(
                df_complexes.lazy().select(pl.col(COMP_ID)),
                on=COMP_ID,
                how="outer",
            )
            .fill_null(pl.lit("train"))
            .sort(pl.col(COMP_ID))
            .select([COMP_ID] + list(sorted([f"{XVAL_ITER}_{i}" for i in range(k)])))
            .collect()
        )
        return df_cross_val

    def generate_cross_val_data(
        self, df_complexes: pl.DataFrame, seed: int = 12345
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
            df_complexes.filter(pl.col(COMP_PROTEINS).list.lengths() > 3)
            .select(pl.col(COMP_ID))
            .to_series()
            .shuffle(seed=seed)
            .to_list()
        )

        k = 10  # 10 folds, 10 rounds
        output, cross_val = self.generate_kfolds(k, list_large_complexes)
        df_cross_val = self.cross_val_to_df(k, cross_val)

        return output, df_cross_val  # output is for the SWC software

    def read_irefindex(self) -> pl.DataFrame:
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
                    "edgetype",
                ]
            )
            .filter(
                (pl.col("edgetype") == "X")
                & (pl.col("taxa").str.starts_with("taxid:559292"))
                & (pl.col("taxb").str.starts_with("taxid:559292"))
                & (pl.col("altA").str.starts_with("cygd:"))
                & (pl.col("altB").str.starts_with("cygd:"))
            )
            .with_columns(
                [
                    pl.col("altA").str.extract(r"cygd:([a-zA-Z0-9]+)").alias(PROTEIN_U),
                    pl.col("altB").str.extract(r"cygd:([a-zA-Z0-9]+)").alias(PROTEIN_V),
                    pl.col("pmids").str.extract(r"pubmed:(\d+)$").alias(PUBMED),
                    pl.col("author").str.extract(r"([12][0-9]{3})").alias("YEAR"),
                ]
            )
            .filter(pl.col("YEAR") < "2012")
            .select([PROTEIN_U, PROTEIN_V, PUBMED])
            .with_columns(sort_prot_cols(PROTEIN_U, PROTEIN_V))
            .unique(maintain_order=True)
            .collect()
        )

        return df


if __name__ == "__main__":
    start_time = time.time()

    pl.Config.set_tbl_cols(40)
    pl.Config.set_tbl_rows(5)

    preprocessor = Preprocessor()
    df_nips = preprocessor.get_nips()

    df_swc = preprocessor.read_swc_composite_network()
    df_dip_ppin_uniprot = preprocessor.read_dip_ppin_uniprot()
    df_dip_ppin = preprocessor.map_dip_ppin_uniprot_to_kegg(df_dip_ppin_uniprot)

    df_irefindex = preprocessor.read_irefindex()

    df_complexes = get_all_cyc_complexes()
    output, df_cross_val = preprocessor.generate_cross_val_data(df_complexes)

    # generate a new cross-val split data for the DIP dataset
    dip_output, df_dip_cross_val = preprocessor.generate_cross_val_data(
        df_complexes, seed=6789
    )

    print(">>> NIPS")
    print(df_nips)

    print(">>> SWC DATA - COMPOSITE PROTEIN NETWORK")
    print(df_swc)

    print(">>> DIP PPIN UNIPROT")
    print(df_dip_ppin_uniprot)

    print(">>> DIP PPIN")
    print(df_dip_ppin)

    print(">>> IREFINDEX")
    print(df_irefindex)

    print(">>> COMPLEXES")
    print(df_complexes)

    print(">>> CROSS VAL DATA")
    print(df_cross_val)

    print(">>> DIP CROSS VAL DATA")
    print(df_dip_cross_val)

    # Write files...
    df_nips.write_csv("../data/preprocessed/yeast_nips.csv", has_header=False)

    df_swc.write_csv("../data/scores/swc_composite_scores.csv", has_header=True)

    df_swc.select([PROTEIN_U, PROTEIN_V]).write_csv(
        "../data/preprocessed/swc_edges.csv", has_header=False
    )

    df_dip_ppin_uniprot.write_csv(
        "../data/databases/dip_ppin_uniprot.csv", has_header=False
    )
    df_dip_ppin.write_csv("../data/preprocessed/dip_ppin.csv", has_header=False)

    df_irefindex.write_csv("../data/preprocessed/irefindex_pubmed.csv", has_header=True)

    with open("../data/preprocessed/cross_val.csv", "w") as file:
        file.write(output)
    df_cross_val.write_csv("../data/preprocessed/cross_val_table.csv", has_header=True)

    # Cross-val split data for DIP dataset
    with open("../data/preprocessed/dip_cross_val.csv", "w") as file:
        file.write(dip_output)
    df_dip_cross_val.write_csv(
        "../data/preprocessed/dip_cross_val_table.csv", has_header=True
    )

    print(">>> Execution Time")
    print(time.time() - start_time)
