from typing import List, TypedDict

PROTEIN_U = "PROTEIN_U"
PROTEIN_V = "PROTEIN_V"
PUBMED = "PUBMED"
PROTEIN = "PROTEIN"

COMP_ID = "COMP_ID"
COMP_PROTEINS = "COMP_PROTEINS"
COMP_INFO = "COMP_INFO"
DENSITY = "DENSITY"

XVAL_ITER = "ITER"  # for cross-validation iterations
METHOD = "METHOD"
SCENARIO = "SCENARIO"
VALUE = "VALUE"

# Metrics
METRIC = "METRIC"
PRECISION = "PRECISION"
RECALL = "RECALL"
PR_AUC = "PR_AUC"
F1_SCORE = "F1_SCORE"
RMSE = "RMSE"
BRIER_SCORE = "BRIER_SCORE"
LOG_LOSS = "LOG_LOSS"


# SWC Features
TOPO = "TOPO"  # Topological weighting - Iterative AdjustCD (k=2)
TOPO_L2 = "TOPO_L2"  # Topological weighting - Iterative AdjustCD (k=2) Level-2 PPIs
STRING = "STRING"  # STRING database score
CO_OCCUR = "CO_OCCUR"  # Co-ocurrence in PubMed literature

SWC_FEATS = [TOPO, TOPO_L2, STRING, CO_OCCUR]

# New features
REL = "REL"  # Experiment reliability - MV Scoring (Post-processed)
CO_EXP = "CO_EXP"  # Gene co-expression - Pearson correlation
GO_CC = "GO_CC"  # GO Semantic Similarity : Cellular Component - TCSS
GO_BP = "GO_BP"  # GO Semantic Similarity : Biological Process - TCSS
GO_MF = "GO_MF"  # GO Semantic Similarity : Molecular Function - TCSS

FEATURES = [TOPO, TOPO_L2, STRING, CO_OCCUR, REL, CO_EXP, GO_CC, GO_BP, GO_MF]

# Super features (for unsupervised weighting)
# Simple average of features subset
SuperFeature = TypedDict("SuperFeature", {"name": str, "features": List[str]})
ALL: SuperFeature = {"name": "ALL", "features": FEATURES}
GO_SS: SuperFeature = {"name": "GO_SS", "features": [GO_CC, GO_BP, GO_MF]}
TOPOS: SuperFeature = {"name": "TOPOS", "features": [TOPO, TOPO_L2]}
ASSOC: SuperFeature = {"name": "ASSOC", "features": [STRING, CO_OCCUR, REL, CO_EXP]}
TOPO_GO: SuperFeature = {"name": "TOPO_GO", "features": [TOPO, GO_CC, GO_BP, GO_MF]}
TOPO_CO_EXP: SuperFeature = {"name": "TOPO_CO_EXP", "features": [TOPO, CO_EXP]}
TOPO_GO_CO_EXP: SuperFeature = {
    "name": "TOPO_GO_CO_EXP",
    "features": [TOPO, GO_CC, GO_BP, GO_MF, CO_EXP],
}

SUPER_FEATS = [
    ALL,
    GO_SS,
    TOPOS,
    ASSOC,
    TOPO_GO,
    TOPO_CO_EXP,
    TOPO_GO_CO_EXP,
]

# Labels of protein pairs
IS_CO_COMP = "IS_CO_COMP"


# Predicted classes probability of protein pairs
PROBA_CO_COMP = "PROBA_CO_COMP"  # probability of being a co-complex pair
PROBA_NON_CO_COMP = "PROBA_NON_CO_COMP"

WEIGHT = "WEIGHT"  # alias for PROBA_CO_COMP

# These were not used.
IS_NIP = "IS_NIP"
PROBA_NIP = "PROBA_NIP"  # probability of being a NIP pair
PROBA_NON_NIP = "PROBA_NON_NIP"
