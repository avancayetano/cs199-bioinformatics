# CS 199 Bioinformatics

### Data:

- Protein Network - SWC Data (late 2011 data)
- Gene co-expression data - GSE3431 (2005)
- Gene ontology - `go-basic.obo` (2023-03-06) from GO website.
- GO annotations - `gene_association.sgd.20230313.gaf` (2023-03-13) from SGD website.
- iRefIndex - version 19.0 (2022-08-22) - this is for the PUBMED IDs only (for `REL` feature)
- Negatome 2.0 - combined, stringent version (mid-2013)

### Possibility

- DIP network (smaller network)

#### Use of modern data

SWC data is old (2011 data). However, you might have noticed that I used more modern data for the new features I introduced. This is because the goal of this study is to extend SWC to use more modern features (aside from improving the machine learning model).

### Project Structure:

- data/
  - clusters/ - (NOTE: gitignore entire dir except READMEs)
    - all_edges/
      - cross_val/ - includes scores from swc/all_edges/cross_val
        - out.{file}.I40
      - features/
        - out.{file}.I40
    - 20k_edges/
      - cross_val/ - includes scores from swc/20k_edges/cross_val
        - out.{file}.I40
      - features/
        - out.{file}.I40
  - databases/
    - large/ - (NOTE: gitignore entire dir except READMEs)
      - irefindex
    - gene_association_data
    - gene_ontology_data
    - GSE3431_data
    - negatome_combined_stringent.txt
    - negatome_kegg_mapped.txt
  - preprocessed/
    - cross_val_table.csv
    - cross_val.csv
    - irefindex_pubmed.csv
    - swc_composite_pairs.csv
    - yeast_nips.csv
  - scores/
    - swc_composite_scores.csv
    - go_ss_scores.csv
    - rel_scores.csv
    - co_exp_scores.csv
  - swc/
    - raw_weighted/
      - swc_weighted scored_edges iter{n}.txt
    - complexes_CYC.txt
    - data_yeast.txt
  - training/ - (NOTE: gitignore entire dir except READMEs)
    - (training artifacts)
  - weighted/ - (all without headers. NOTE: gitignore entire dir except READMEs)
    - all_edges/
      - cross_val/
        - {model}\_iter{n}.csv
        - {model}\_pca_iter{n}.csv
        - {model}\_{feats}\_iter{n}.csv
      - features/ (TODO: could possibly add combinations of features as well (by averaging))
        - {FEATURE}.csv
    - 20k_edges/
      - cross_val/
        - {model}\_20k_iter{n}.csv
        - {model}\_pca_20k_iter{n}.csv
        - {model}\_{feats}\_20k_iter{n}.csv
      - features/
        - {FEATURE}\_20k.csv
  - src/
  - typings/
  - TCSS/
  - README.md

---

## External Data used

### SWC Data

### Gene Ontology

(to be continued)

## External Software packages/services used

### TCSS

This study uses the Topological Clustering Semantic Similarity (TCSS) software package proposed by Jain & Bader (2010) on their study: _An improved method for scoring protein-protein
interactions using semantic similarity within the gene ontology._

TCSS is licensed under GNU Lesser General Public License v3.0. The said license can be found in the `TCSS\` directory. For more information, the readers are directed to `TCSS\README.md`.

Link:

- `insert link`

Command:
`python TCSS/tcss.py -i data/preprocessed/swc_edges.csv -o data/scores/go_ss_scores.csv --drop="IEA" --gene=data/databases/gene_association.sgd.20230313.gaf --go=data/databases/go-basic.obo`

### SWC

This study also uses the SWC software package and source files. The SWC method was proposed by Yong et. al. (2012) on their study: _Supervised maximum-likelihood weighting of composite protein networks for complex prediction_.

Link:

- `insert link`

Command:
`perl score_edges.pl -i data_yeast.txt -c complexes_CYC.txt -m x -x cross_val.csv -e 0 -o "swc"`

### UniProt ID Mapping

The UniProt Retrieve/ID mapping service was used to map each UniProtKB AC/ID in Negatome 2.0 to its corresponding KEGG entry (systematic name).

Link

- https://www.uniprot.org/id-mapping

### MCL

`~/local/bin/mcl path/to/weighted_network.csv --abc -I 4.0`

(will output out.{file}.csv.I40 to the current path)
