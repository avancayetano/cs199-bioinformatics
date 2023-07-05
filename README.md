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

## References

[2] M Ashburner, C A Ball, J A Blake, D Botstein, H Butler, J M Cherry, A P Davis, K Dolinski, S S
Dwight, J T Eppig, M A Harris, D P Hill, L Issel-Tarver, A Kasarskis, S Lewis, J C Matese, J E
Richardson, M Ringwald, G M Rubin, and G Sherlock. Gene ontology: tool for the unification of
biology. the gene ontology consortium. Nat. Genet., 25(1):25–29, May 2000.

[4] Jerome Beltran, Catalina Montes, John Justine Villar, and Adrian Roy Valdez. A hybrid method
for protein complex prediction in weighted protein-protein interaction networks. Philippine Science Letters, 10, 02 2017.

[8] Tianqi Chen and Carlos Guestrin. Xgboost: A scalable tree boosting system. In Proceedings of the
22nd acm sigkdd international conference on knowledge discovery and data mining, pages 785–794, 2016.

[9] J Michael Cherry, Eurie L Hong, Craig Amundsen, Rama Balakrishnan, Gail Binkley, Esther T
Chan, Karen R Christie, Maria C Costanzo, Selina S Dwight, Stacia R Engel, Dianna G Fisk,
Jodi E Hirschman, Benjamin C Hitz, Kalpana Karra, Cynthia J Krieger, Stuart R Miyasato, Rob S
Nash, Julie Park, Marek S Skrzypek, Matt Simison, Shuai Weng, and Edith D Wong. Saccha-
romyces genome database: the genomics resource of budding yeast. Nucleic Acids Res., 40(Database
issue):D700–5, January 2012.

[11] The Gene Ontology Consortium, Suzi A Aleksander, James Balhoff, Seth Carbon, J Michael Cherry,
Harold J Drabkin, Dustin Ebert, Marc Feuermann, Pascale Gaudet, Nomi L Harris, David P Hill,
Raymond Lee, Huaiyu Mi, Sierra Moxon, Christopher J Mungall, Anushya Muruganugan, Tremayne
Mushayahama, Paul W Sternberg, Paul D Thomas, Kimberly Van Auken, Jolene Ramsey, Deborah A
Siegele, Rex L Chisholm, Petra Fey, Maria Cristina Aspromonte, Maria Victoria Nugnes, Federica
Quaglia, Silvio Tosatto, Michelle Giglio, Suvarna Nadendla, Giulia Antonazzo, Helen Attrill, Gil
dos Santos, Steven Marygold, Victor Strelets, Christopher J Tabone, Jim Thurmond, Pinglei Zhou,
Saadullah H Ahmed, Praoparn Asanitthong, Diana Luna Buitrago, Meltem N Erdol, Matthew C
Gage, Mohamed Ali Kadhum, Kan Yan Chloe Li, Miao Long, Aleksandra Michalak, Angeline Pesala,
Armalya Pritazahra, Shirin C C Saverimuttu, Renzhi Su, Kate E Thurlow, Ruth C Lovering, Colin
Logie, Snezhana Oliferenko, Judith Blake, Karen Christie, Lori Corbani, Mary E Dolan, Harold J
Drabkin, David P Hill, Li Ni, Dmitry Sitnikov, Cynthia Smith, Alayne Cuzick, James Seager, Laurel
Cooper, Justin Elser, Pankaj Jaiswal, Parul Gupta, Pankaj Jaiswal, Sushma Naithani, Manuel Lera-
Ramirez, Kim Rutherford, Valerie Wood, Jeffrey L De Pons, Melinda R Dwinell, G Thomas Hayman,
Mary L Kaldunski, Anne E Kwitek, Stanley J F Laulederkind, Marek A Tutaj, Mahima Vedi, Shur-
Jen Wang, Peter D’Eustachio, Lucila Aimo, Kristian Axelsen, Alan Bridge, Nevila Hyka-Nouspikel,
Anne Morgat, Suzi A Aleksander, J Michael Cherry, Stacia R Engel, Kalpana Karra, Stuart R
Miyasato, Robert S Nash, Marek S Skrzypek, Shuai Weng, Edith D Wong, Erika Bakker, Tanya Z
Berardini, Leonore Reiser, Andrea Auchincloss, Kristian Axelsen, Ghislaine Argoud-Puy, Marie-
Claude Blatter, Emmanuel Boutet, Lionel Breuza, Alan Bridge, Cristina Casals-Casas, Elisabeth
Coudert, Anne Estreicher, Maria Livia Famiglietti, Marc Feuermann, Arnaud Gos, Nadine Gruaz-
Gumowski, Chantal Hulo, Nevila Hyka-Nouspikel, Florence Jungo, Philippe Le Mercier, Damien
Lieberherr, Patrick Masson, Anne Morgat, Ivo Pedruzzi, Lucille Pourcel, Sylvain Poux, Catherine
Rivoire, Shyamala Sundaram, Alex Bateman, Emily Bowler-Barnett, Hema Bye-A-Jee, Paul Denny,
Alexandr Ignatchenko, Rizwan Ishtiaq, Antonia Lock, Yvonne Lussi, Michele Magrane, Maria J
Martin, Sandra Orchard, Pedro Raposo, Elena Speretta, Nidhi Tyagi, Kate Warner, Rossana Zaru,
Alexander D Diehl, Raymond Lee, Juancarlos Chan, Stavros Diamantakis, Daniela Raciti, Mag-
dalena Zarowiecki, Malcolm Fisher, Christina James-Zorn, Virgilio Ponferrada, Aaron Zorn, Sridhar
Ramachandran, Leyla Ruzicka, and Monte Westerfield. The Gene Ontology knowledgebase in 2023.
Genetics, 224(1):iyad031, 03 2023.

[13] Stijn Dongen. Graph clustering by flow simulation. PhD thesis, Center for Math and Computer
Science (CWI), 05 2000.

[14] A. J. Enright, S. Van Dongen, and C. A. Ouzounis. An efficient algorithm for large-scale detection
of protein families. Nucleic Acids Res, 30(7):1575–1584, Apr 2002.
20

[19] Shobhit Jain and Gary D. Bader. An improved method for scoring protein-protein interactions using
semantic similarity within the gene ontology. BMC Bioinformatics, 11(1):562, Nov 2010.

[22] S. Kerrien, B. Aranda, L. Breuza, A. Bridge, F. Broackes-Carter, C. Chen, M. Duesbury, M. Du-
mousseau, M. Feuermann, U. Hinz, C. Jandrasits, R. C. Jimenez, J. Khadake, U. Mahadevan,
P. Masson, I. Pedruzzi, E. Pfeiffenberger, P. Porras, A. Raghunath, B. Roechert, S. Orchard, and
H. Hermjakob. The IntAct molecular interaction database in 2012. Nucleic Acids Res, 40(Database
issue):D841–846, Jan 2012.

[24] George D Kritikos, Charalampos Moschopoulos, Michalis Vazirgiannis, and Sophia Kossida. Noise
reduction in protein-protein interaction graphs by the implementation of a novel weighting scheme.
BMC Bioinformatics, 12(1):239, June 2011.

[26] L. Licata, L. Briganti, D. Peluso, L. Perfetto, M. Iannuccelli, E. Galeota, F. Sacco, A. Palma, A. P.
Nardozza, E. Santonico, L. Castagnoli, and G. Cesareni. MINT, the molecular interaction database:
2012 update. Nucleic Acids Res, 40(Database issue):D857–861, Jan 2012.

[28] Guimei Liu, Jinyan Li, and Limsoon Wong. Assessing and predicting protein interactions using both local and global network topological metrics. Genome Inform., 21:138–149, 2008.

[29] Guimei Liu, Limsoon Wong, and Hon Nian Chua. Complex discovery from weighted PPI networks.
Bioinformatics, 25(15):1891–1897, August 2009.

[34] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Pretten-
hofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and
E. Duchesnay. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research,
12:2825–2830, 2011.

[37] Shuye Pu, Jessica Wong, Brian Turner, Emerson Cho, and Shoshana J Wodak. Up-to-date catalogues
of yeast protein complexes. Nucleic Acids Res., 37(3):825–831, February 2009.

[39] Sabry Razick, George Magklaras, and Ian M Donaldson. iRefIndex: a consolidated protein interac-
tion database with provenance. BMC Bioinformatics, 9:405, September 2008.

[43] Lukasz Salwinski, Christopher S Miller, Adam J Smith, Frank K Pettit, James U Bowie, and David
Eisenberg. The database of interacting proteins: 2004 update. Nucleic Acids Res., 32(Database
issue):D449–51, January 2004.

[50] C. Stark, B. J. Breitkreutz, A. Chatr-Aryamontri, L. Boucher, R. Oughtred, M. S. Livstone, J. Nixon,
K. Van Auken, X. Wang, X. Shi, T. Reguly, J. M. Rust, A. Winter, K. Dolinski, and M. Tyers. The
BioGRID Interaction Database: 2011 update. Nucleic Acids Res, 39(Database issue):698–704, Jan 2011.

[51] Chris Stark, Bobby-Joe Breitkreutz, Teresa Reguly, Lorrie Boucher, Ashton Breitkreutz, and Mike
Tyers. BioGRID: a general repository for interaction datasets. Nucleic Acids Res, 34(Database
issue):D535–9, January 2006.

[52] D. Szklarczyk, A. Franceschini, M. Kuhn, M. Simonovic, A. Roth, P. Minguez, T. Doerks, M. Stark,
J. Muller, P. Bork, L. J. Jensen, and C. von Mering. The STRING database in 2011: functional
interaction networks of proteins, globally integrated and scored. Nucleic Acids Res, 39(Database
issue):D561–568, Jan 2011.

[53] B. P. Tu, A. Kudlicki, M. Rowicka, and S. L. McKnight. Logic of the yeast metabolic cycle: temporal
compartmentalization of cellular processes. Science, 310(5751):1152–1158, Nov 2005.

[54] Stijn Van Dongen. Graph clustering via a discrete uncoupling process. SIAM Journal on Matrix
Analysis and Applications, 30(1):121–141, 2008.

[58] C. H. Yong, G. Liu, H. N. Chua, and L. Wong. Supervised maximum-likelihood weighting of
composite protein networks for complex prediction. BMC Syst Biol, 6 Suppl 2(Suppl 2):S13, 2012.
