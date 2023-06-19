# TCSS Modified Version\*

_\*Last modified by Anthony Van Cayetano on 2023-06-19._

This `TCSS` library is a modified version of Topological Clustering Semantic Similarity (TCSS) algorithm, originally written by Shobhit Jain and Gary D Bader (2010).

This README file outlines the updated specifications of this modified version of `TCSS`, as well as the changes made by Anthony Van Cayetano and the corresponding reasons for these changes. The reader can also refer to the original README file (renamed `README_orig`) written by Jain & Bader, although some information on the said file may be outdated.

## Updated specifications

Requirements:

- Python 3.11

Gene ontology and gene association files were provided by the original `TCSS` although these data may be outdated.

### Usage

For the usage of `TCSS`, the readers are directed to the original README file (`README_orig`). However, note that unlike in the original version, this version of `TCSS` works with only the _Systematic Names_ of the genes, instead of _SGD gene ids_ (the ones used in the original version as noted in `README_orig`).

## Changes

- The original Python source files of `TCSS` were converted to their corresponding Python 3 versions because these source files were written in Python 2. The conversion process was automatically done via a builtin Python script called `2to3.py`. The command `python <path>\<to>\Tools\scripts\2to3.py -W <file.py>` is executed for every `<file.py>` (i.e. Python source file) in the original `TCSS`. Note that running the said command creates a backup file (the `.bak` files) for each Python file.
- The file `TCSS\main.py` was also modified so that the output files of running `TCSS` match the format of the existing output files of other weighting methods. The said Python file was annotated with comments regarding the changes.
- The file `TCSS\parser.py` was also modified so that the _Systematic Name_ of a gene is used by `TCSS` instead of _SGD gene IDs_ (which were used in the original version). This change is necessary because the other weighting methods use _Systematic Name_. The said Python file was annotated with comments regarding the changes.
- There was an indentation error in `TCSS\tcss.py`, which was fixed.
- Removed the GO files, which are `gene_association.sgd` and `gene_ontology.obo.txt` because different GO files were used.

## How this version was used in the study

The researcher ran the following command:

`python TCSS/tcss.py -i data/preprocessed/swc_edges.csv -o data/scores/go_ss_scores.csv --drop="IEA" --gene=data/databases/sgd.gaf --go=data/databases/gene_ontology.obo`

to generate the GO-weighted network (`data/scores/go_ss_scores.csv`) used in the study.

For the DIP composite network, the following command was used:
`python TCSS/tcss.py -i data/preprocessed/dip_edges.csv -o data/scores/dip_go_ss_scores.csv --drop="IEA" --gene=data/databases/sgd.gaf --go=data/databases/gene_ontology.obo`
