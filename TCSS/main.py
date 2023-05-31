'''
Created on 2010-07-19

@author: Shobhit Jain

@contact: shobhit@cs.toronto.edu

----------------------------------------------------------------------

Modified on 2023-04-14 by Anthony Van Cayetano
Changes:
    - This modified file is the Python 3 version of the original
        file (which was originally written in Python 2). The conversion
        was done via the builtin Python script called '2to3'. 
        The said script also created a backup of the original file,
        with the name 'main.py.bak'.

    - A new function with the name 'return_detail_new' was defined by
        Anthony Van Cayetano.

    - The function 'calculate_semantic_similarity' was modified by
        Anthony Van Cayetano such that it calls the function 
        'return_detail_new' instead of 'return_detail'.

'''

from ontology import GOGraph

def load_semantic_similarity(ontology_file, gene_file, ontology, code):
    '''
    Calls functions for loading and processing data.
    '''
    objs = {}
    ontology = ontology.split(",")
    g = GOGraph()
    print("loading data.............")
    g._obo_parser(ontology_file)
    g._go_annotations(gene_file, code)
    print("processing data..........(takes time)")
    run = {'C':g._cellular_component, 'P':g._biological_process, 'F':g._molecular_function}
    ont = {'C':"Cellular Component", 'P':"Biological Process", 'F':"Molecular Function"}
    for i in ontology:
        i = i.split(":")
        print("working with %s ontology....."%ont[i[0]])
        objs[i[0]] = run[i[0]]()
        objs[i[0]]._species()
        objs[i[0]]._clustering(float(i[1]))
    return objs



def return_detail(result, geneA, geneB, detail):
    '''
    Formats the output for printing on screen or on file.
    '''
    domain_def = {'C':'Cellular Component', 'P':'Biological Process', 'F':'Molecular Function'}
    r = "\nSemantic similarity between " + geneA + " and " + geneB + " is:\n\n"
    for domain in result:
        r += " " + domain_def[domain] + ": " + str(result[domain][0]) + "\n"
        if detail:
            for data in result[domain][1]:
                r += "  GO id assigned to " + geneA + " is: " + data[0] + \
                     "\n  GO id assigned to " + geneB + " is: " + data[1] + \
                     "\n  LCA of assigned GO ids is: " + "|".join(result[domain][1][data]['lca']) + "\n\n"                  
    return r + "\n\n\n"
        
        
def return_detail_new(result, geneA, geneB, detail):
    '''
    This function was created by Anthony Van Cayetano on 2023-04-14.
    This function is a modified version of the 'return_detail' function above.
    '''
    r = f"{geneA},{geneB},{result['C'][0]},{result['P'][0]},{result['F'][0]}\n"
    return r


def calculate_semantic_similarity(objs, geneA, geneB, detail):
    '''
    Calls the function for calculating semantic similarity between
    genesA and genesB.
    '''
    result = {}
    for domain in objs:
        result[domain] = objs[domain]._semantic_similarity(geneA, geneB)
    return return_detail_new(result, geneA, geneB, detail) # This line was modified by Anthony Van Cayetano on 2023-04-14.
        
    #cc._semantic_similarity('S000004065', 'S000001451')
    
