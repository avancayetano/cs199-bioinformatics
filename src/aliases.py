PROTEIN_U = "PROTEIN_U"
PROTEIN_V = "PROTEIN_V"
PUBMED = "PUBMED"
PROTEIN = "PROTEIN"

COMPLEX_ID = "COMPLEX_ID"
COMPLEX_PROTEINS = "COMPLEX_PROTEINS"
COMPLEX_DESCRIPTION = "COMPLEX_DESCRIPTION"

CROSS_VAL_ITER = "ITER"  # for cross-validation iterations

# Features
REL = "REL"  # Experiment reliability - MV Scoring (Post-processed)
CO_EXP = "CO_EXP"  # Gene co-expression - Pearson correlation
GO_CC = "GO_CC"  # GO Semantic Similarity : Cellular Component - TCSS
GO_BP = "GO_BP"  # GO Semantic Similarity : Biological Process - TCSS
GO_MF = "GO_MF"  # GO Semantic Similarity : Molecular Function - TCSS

# SWC Features
TOPO = "TOPO"  # Topological weighting - Iterative AdjustCD (k=2)
TOPO_L2 = "TOPO_L2"  # Topological weighting - Iterative AdjustCD (k=2) Level-2 PPIs
STRING = "STRING"  # STRING database score
CO_OCCUR = "CO_OCCUR"  # Co-ocurrence in PubMed literature

GO_SS = "GO_SS"  # average of GO_CC, GO_BP, and GO_MF

# Actual classes of protein pairs.
IS_CO_COMP = "IS_CO_COMP"
IS_NIP = "IS_NIP"

# Predicted classes probability of protein pairs
PROB_CO_COMP = "PROB_CO_COMP"  # probability of being a co-complex pair
PROB_NON_CO_COMP = "PROB_NON_CO_COMP"
PROB_NIP = "PROB_NIP"  # probability of being a NIP pair; later used as a feature
PROB_NON_NIP = "PROB_NON_NIP"

WEIGHT = "WEIGHT"  # alias for PROB_CO_COMP
