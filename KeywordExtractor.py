from utils import *
import networkx as nx
# https://networkx.org/documentation/stable/tutorial.html

class KeywordExtractor:
    def __init__(self, abstract):
        self.abstract = abstract # raw input text
        self.tokens, self.sentences = prune_text(abstract) # list of tokens and sentences 
        self.unique_tokens = list(set(self.tokens))
        self.graph = nx.Graph() # empty graph structure where 
        self.graph.add_nodes_from(self.tokens) # adding list of tokens as graph vertices




    