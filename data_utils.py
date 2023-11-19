import pandas as pd
from ast import literal_eval
import networkx as nx
from KeywordExtractor import KeywordExtractor

def load_abstract(name):
    """
    Load text from file based on the filename.
    """
    with open('data/' + name + '.txt', 'r') as file:
        abstract = file.read()
    return abstract

def get_data(path, version):
    """
    Load abstracts and keywords from a csv file. Two datasets are available, CS and inspec.
    """
    if version == "CS":
        df = pd.read_csv(path)
        keywords = df['keywords'].apply(lambda x: literal_eval(x))
        titles = df['Title']
        abstracts = df['abstract']

        joined = [e1 + '. ' + e2 for e1, e2 in zip(titles, abstracts)] 
        return joined, abstracts, titles, keywords 
    elif version == "inspec":
        df = pd.read_csv(path)
        keywords = df['keywords'].apply(lambda x: literal_eval(x))
        abstracts = df['abstracts']
        return abstracts, keywords 
    
def make_metrics_csv(ke: KeywordExtractor, methods: dict):
    """
    Make csv of the graph with the respective ranking algorithm.
    """
    df = pd.DataFrame()
    for key in methods.keys():
        rank_dict = ke.order_nodes(methods[key], to_print=False)
        headers = pd.MultiIndex.from_product([[methods[key]], ["Word", "Ranking"]])
        df = pd.concat([df, pd.DataFrame(list(rank_dict.items()), columns=headers)], axis=1)
    df.to_csv('rankings.csv', index=False)
