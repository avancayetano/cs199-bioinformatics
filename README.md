# A supervised co-complex probability weighting of yeast composite protein networks using gradient-boosted trees for protein complex detection

## Data

### Databases (`data/databases` and `data/swc`):

These are the raw database files.

- Base yeast composite protein network - SWC Data (late 2011 data) (`data/swc/data_yeast.txt`)
- Gene co-expression data - GSE3431 (2005) (`GSE3431_setA_family.pcl`)
- Gene ontology - `gene_ontology.obo` (2011-10-31) from GO website.
- GO annotations - `sgd.gaf` (2011-10-29) from SGD website.
- iRefIndex - version 19.0 (2022-08-22) - this is for the PUBMED IDs only (for `REL` feature) (this file is located in `data/databases/large`)
- DIP PPIN - `Scere20170205.txt` (2017-02-05)
- CYC2008 (already provided by the SWC software; this is located in `data/swc`)

NOTE: Due to the large size of the iRefIndex database, it is zipped so you need to extract it.

### Preprocessed data (`data/preprocessed`)

Preprocessed data can be found here. Note: the cross-validation splits used by both XGW and SWC are found here.

### Clusters (`data/clusters`)

This is where the MCL algorithm outputs its predicted clusters.

### Evals (`data/evals`)

This is where evaluation data are stored (precision-recall, log loss, etc...)

### Scores (`data/scores`)

This is where the feature scores are stored.

### SWC data (`data/swc`)

This is where the data that are provided and/or needed by SWC are stored.

### Training (`data/training`)

Hyperparameter settings are stored here based on previous training, as well as the computed feature importances.

### Weighted (`data/weighted`)

The weighted protein networks.

## Source code (`src/`)

All the source codes of XGW are stored here. Important codes:

`preprocessor.py` - Preprocesses data in `data/databases` and stored preprocessed data to `data/preprocessed`. This is run only once (if `data/preprocessed`) is empty.

`dip_preprocessor.py` - Preprocesses raw DIP PPIN and constructs the base composite network for this network by topologically weighting it (TOPO and TOPO_L2), and integrating STRING and CO_OCCUR features.

`co_exp_scoring.py` and `rel_scoring.py` - Scores the original and DIP protein network based on CO_EXP and REL.

`weighting.py` - Weights the two composite networks using the 19 weighting methods. Also, outputs the feature importances of XGW to `data/training` (see the notes in the comments of this file)

`evaluate_clusters.py` - To evaluate the predicted clusters (see the notes in the comments of this file)

`evaluate_comp_edges.py` - To evaluate the co-complex pair classification of each method (see the notes in the comments of this file)

## External Software packages/services used

### TCSS

This study uses the Topological Clustering Semantic Similarity (TCSS) software package proposed by Jain & Bader (2010) on their study: _An improved method for scoring protein-protein
interactions using semantic similarity within the gene ontology._

Requirements: Python.

Link: baderlab.org/Software/TCSS

TCSS is licensed under GNU Lesser General Public License v3.0. The said license can be found in the `TCSS\` directory. For more information, the readers are directed to `TCSS\README.md`.

Commands to run TCSS:

`python TCSS/tcss.py -i data/preprocessed/swc_edges.csv -o data/scores/go_ss_scores.csv --drop="IEA" --gene=data/databases/sgd.gaf --go=data/databases/gene_ontology.obo`

to generate the GO-weighted network (`data/scores/go_ss_scores.csv`) used in the study.

For the DIP composite network, the following command was used:

`python TCSS/tcss.py -i data/preprocessed/dip_edges.csv -o data/scores/dip_go_ss_scores.csv --drop="IEA" --gene=data/databases/sgd.gaf --go=data/databases/gene_ontology.obo`

### SWC

This study also uses the SWC software package and source files. The SWC method was proposed by Yong et. al. (2012) on their study: _Supervised maximum-likelihood weighting of composite protein networks for complex prediction_.

Requirements: Perl.

Link: https://www.comp.nus.edu.sg/~wongls/projects/complexprediction/SWC-31oct14/

Commands:

For the original composite network:

`perl score_edges.pl -i data_yeast.txt -c complexes_CYC.txt -m x -x cross_val.csv -e 0 -o "swc"`

For the DIP composite network:

`perl score_edges.pl -i dip_data_yeast.txt -c complexes_CYC.txt -m x -x dip_cross_val.csv -e 0 -o "dip_swc"`

### UniProt ID Mapping

The UniProt Retrieve/ID mapping service was used to map each UniProtKB AC/ID in the DIP PPIN to its corresponding KEGG entry (systematic name).

Link

- https://www.uniprot.org/id-mapping

### MCL

The Markov Cluster (MCL) Algorithm was used to cluster the weighted protein networks.

Requirements: MCL.

Link: https://github.com/micans/mcl

To run MCL (assuming MCL is installed in `~/local/bin/mcl`), navigate to `src`, then run:

`./run_mcl.sh ../data/weighted/ ./ 4`

This command will run the MCL algorithm on the weighted protein networks in `data/weighted` with inflation parameter set to 4. Note that the inflation parameter can be set to arbitrary value, preferrable around the range of [2, 5].

Running the above command will output `out.{file}.csv.I40` files to `data/clusters`.
