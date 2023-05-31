"""
Main Runner file.

Flow:
1) Run preprocesser only ONCE.
    - This will output biogrid and dip ppins (w/ and w/o pubmed)
        and co_complex pairs.

    - Another possibility is instead of using REL as a feature,
        use it as a filtering tool instead. (Only applicable if REL
        is indeed not used, at least wait for the evaluation results.)

    # Output Directory: data/preprocessed
        (All static, unchanging data should be here.)
        
        - PPINs w/ pubmed (x2)
        - PPINs w/o pubmed (x2)
        - Co-complex pairs (CYC2008)

        (!!! NO HEADERS !!!)

-----------------------------------------------------
> INPUT: PPIN (biogrid or dip)

2) Run all the scoring methods
    - All outputs should be prefixed with `{ppin}_*.csv`

    # Output Directory: data/scores
        - 4 score files (x2)

        (!!! WITH HEADERS !!!)

3) Run whatever the model is.
    - All outputs should be prefixed with `{ppin}_*.csv`

    ! FEATURES:
        - REL, TOPO, CO_EXP, GO_CC, GO_BP, GO_MF

    ! MODE:
        - outer

    # Output Directory: data/training
        (Includes params, bins, training data etc)

        ! FILES MAY BE SUFFIXED BY `_NO_REL`

        ? Should I still test my model???

        (!!! WITH HEADERS !!!)      

4) Use learned params to weight the PPIN.
    - All outputs should be prefixed with `{ppin}_weighted.csv`

    # Output Directory: data/weighted
        - Weighted PPINs (x2)

        ! FILES MAY BE SUFFIXED BY `_NO_REL`

        (!!! NO HEADERS !!!)

5) Apply the clustering algorithm on the weighted PPIN.
    - All outputs should be prefixed with `{ppin}_clusters.csv`

    # Output Directory: data/runs
        - Clusters files (x2)

        ! FILES MAY BE SUFFIXED BY `_NO_REL`

        (!!! NO HEADERS, obviously... !!!)

// Repeat steps 2-5 for each PPIN.
-----------------------------------------------------

6) Evaluate the results on a notebook. 

"""
