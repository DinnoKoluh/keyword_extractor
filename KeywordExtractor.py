from utils import *
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# https://networkx.org/documentation/stable/tutorial.html

class KeywordExtractor:
    def __init__(self, abstract):
        self.abstract = abstract # raw input text
        self.tokens, self.sentences = prune_text(abstract) # list of tokens and sentences 
        self.unique_tokens = list(set(self.tokens))
        self.co = {} # co-occurrence representation as a dictionary, it is initialized in the init_graph method, the edges are represented as tuples
        # and they are the keys of the dictionary while the weights are the values
        self.graph = self.init_graph() # graph structure where the relations between tokens are saved

    def init_graph(self):
        """
        Initializes the the graph with the co-occurrence relations where tokens are represented as vertices and edges are the relations between 
        them. The weights are calculated based on the co-occurrence of tokens in a predefined sliding window.
        """
        graph = nx.Graph()
        co, index_dict = get_co(self.sentences, self.unique_tokens)
        self.co = co # initialize the co-occurrence dictionary
        graph.add_nodes_from(index_dict)
        # unpack the dictionary and initialize the graph with edges and weights
        # it can't be done directly as the weights will not be initialized from the dictionary
        # the idea was the networkx represents labels in this format, so they might also initialize the graph as such but it doesn't
        for edge, weight in self.co.items():
            graph.add_edge(edge[0], edge[1], weight=weight)
        #graph.add_edges_from(co)
        return graph
    
    def visualize_graph(self):
        """
        Visualize the graph representation of text.
        """
        labels = nx.get_edge_attributes(self.graph,'weight')
        plt.figure(figsize=(8, 8))
        #pos = nx.spring_layout(G, seed=1)  # Layout algorithm (you can try different algorithms)
        #pos = nx.shell_layout(G)
        pos = nx.circular_layout(self.graph) # layout of the graph
        nx.draw(self.graph, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=10, font_weight='bold')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=labels, font_size=10, font_weight='bold')
        plt.show()