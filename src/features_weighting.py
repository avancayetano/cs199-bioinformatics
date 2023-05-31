import polars as pl

from aliases import (
    CO_EXP,
    CO_OCCUR,
    GO_BP,
    GO_CC,
    GO_MF,
    GO_SS,
    PROTEIN_U,
    PROTEIN_V,
    REL,
    STRING,
    TOPO,
    TOPO_L2,
)

if __name__ == "__main__":
    features = [
        REL,
        CO_EXP,
        GO_CC,
        GO_BP,
        GO_MF,
        GO_SS,
        TOPO,
        TOPO_L2,
        STRING,
        CO_OCCUR,
    ]

    swc_features = [
        TOPO,
        TOPO_L2,
        STRING,
        CO_OCCUR,
    ]

    go_features = [GO_CC, GO_BP, GO_MF, GO_SS]

    df_swc = pl.read_csv("../data/preprocessed/swc_composite_data.csv")
    df_go_ss = pl.read_csv(
        "../data/scores/swc_go_ss.csv", null_values="None"
    ).fill_null(0.0)

    df_rel = pl.read_csv("../data/scores/swc_rel.csv")
    df_co_exp = pl.read_csv("../data/scores/swc_co_exp.csv")

    source_df = {
        REL: df_rel,
        CO_EXP: df_co_exp,
        **{F: df_go_ss for F in go_features},
        **{F: df_swc for F in swc_features},
    }

    for F in features:
        if F == GO_SS:
            df = (
                source_df[F]
                .with_columns(
                    (pl.col(GO_CC) + pl.col(GO_BP) + pl.col(GO_MF) / 3).alias(F)
                )
                .select([PROTEIN_U, PROTEIN_V, F])
            )

        else:
            df = source_df[F].select([PROTEIN_U, PROTEIN_V, F])

        df = df.sort(F, descending=True)
        df.write_csv(
            f"../data/weighted/swc_weighted_{F}.csv",
            has_header=False,
            separator="\t",
        )
        df.head(20_000).write_csv(
            f"../data/weighted/swc_weighted_{F}_20k.csv",
            has_header=False,
            separator="\t",
        )
