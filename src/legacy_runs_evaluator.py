import copy
from typing import Dict, List, Set

from aliases import (
    CO_EXP,
    CO_OCCUR,
    GO_BP,
    GO_CC,
    GO_MF,
    GO_SS,
    REL,
    STRING,
    TOPO,
    TOPO_L2,
)


def is_match(cmp1: Set[str], cmp2: Set[str]):
    thresh = 0.5
    jaccard = len(cmp1.intersection(cmp2)) / len(cmp1.union(cmp2))
    if jaccard >= thresh:
        return True
    return False


def get_complexes(path: str, sep: str = " ") -> List[Set[str]]:
    complexes: List[Set[str]] = []

    with open(path) as file:
        lines = file.readlines()
        for line in lines:
            line = set(line.strip().split(sep))
            complexes.append(line)
    complexes = list(filter(lambda cmp: len(cmp) >= 2, complexes))
    return complexes


def compare(ref_complexes: List[Set[str]], pred_complexes: List[Set[str]], name: str):
    ref_complexes = copy.deepcopy(ref_complexes)
    pred_complexes = copy.deepcopy(pred_complexes)
    matches = 0
    match_size: Dict[int, int] = {}
    for cmp1 in pred_complexes:
        for cmp2 in ref_complexes:
            if is_match(cmp1, cmp2):
                size = len(cmp2)
                if size in match_size:
                    match_size[size] += 1
                else:
                    match_size[size] = 1
                matches += 1

    print(f"NAME: {name} | MATCHES: {matches} | TOTAL: {len(pred_complexes)}")

    return matches


# SWC
swc_complexes = get_complexes(
    "../data/runs/mcl/out.swc yeast scored_edges.txt.I40", "\t"
)
swc20k_complexes = get_complexes("../data/runs/mcl/out.swc20k yeast.txt.I40", "\t")


features = [
    REL,
    CO_EXP,
    GO_CC,
    GO_BP,
    GO_MF,
    GO_SS,  # not really a feature...
    TOPO,
    TOPO_L2,
    STRING,
    CO_OCCUR,
]

methods = [
    "unweighted",
    "weighted_cnb",
    "weighted_cnb_all_train",
    "weighted_cnb_all_train_swc",
    "weighted_gnb",
    "weighted_mnb",
    "weighted_mlp",
    "weighted_svm",
]

methods_complexes: Dict[str, List[Set[str]]] = {}
methods_20k_complexes: Dict[str, List[Set[str]]] = {}

# get all complexes
ref_complexes = get_complexes("../data/preprocessed/test_complexes.csv", ",")

for m in methods:
    methods_complexes[m] = get_complexes(f"../data/runs/mcl/out.swc_{m}.csv.I40", "\t")
    if m != "unweighted":
        methods_20k_complexes[m] = get_complexes(
            f"../data/runs/mcl/out.swc_{m}_20k.csv.I40", "\t"
        )

F_complexes: Dict[str, List[Set[str]]] = {}
F_complexes_20k: Dict[str, List[Set[str]]] = {}

for F in features:
    F_complexes[F] = get_complexes(
        f"../data/runs/mcl/out.swc_weighted_{F}.csv.I40", "\t"
    )
    F_complexes_20k[F] = get_complexes(
        f"../data/runs/mcl/out.swc_weighted_{F}_20k.csv.I40", "\t"
    )


print(f"NUMBER OF TEST COMPLEXES: {len(ref_complexes)}")
compare(ref_complexes, swc_complexes, name="SWC")
for m in methods_complexes:
    compare(ref_complexes, methods_complexes[m], name=m)

for f in features:
    compare(ref_complexes, F_complexes[f], name=f)


print("---------- 20K EDGES ONLY -----------------")
print(f"NUMBER OF TEST COMPLEXES: {len(ref_complexes)}")
compare(ref_complexes, swc20k_complexes, name="SWC 20K")
for m in methods_20k_complexes:
    compare(ref_complexes, methods_20k_complexes[m], name=f"{m} 20K")

for f in features:
    compare(ref_complexes, F_complexes_20k[f], name=f"{f} 20K")
