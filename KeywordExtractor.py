from nlp_utils import *
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# https://networkx.org/documentation/stable/tutorial.html

class KeywordExtractor:
    def __init__(self, abstract, window_size):
        self.abstract = abstract # raw input text
        self.tokens, self.sentences = prune_text(abstract) # list of tokens and sentences 
        self.unique_tokens = list(set(self.tokens))
        self.co = {} # co-occurrence representation as a dictionary, it is initialized in the init_graph method, the edges are represented as tuples
        # and they are the keys of the dictionary while the weights are the values
        self.window_size = window_size
        self.graph = self.init_graph() # graph structure where the relations between tokens are saved
        self.added_weights = False # if the embedding weights have already been added to graph

    def init_graph(self):
        """
        Initializes the graph with the co-occurrence relations where tokens are represented as vertices and edges are the relations between 
        them. The weights are calculated based on the co-occurrence of tokens in a predefined sliding window.
        """
        graph = nx.Graph()
        co, index_dict = get_co(sentences = self.sentences, window_size = self.window_size)
        self.co = co # initialize the co-occurrence dictionary
        graph.add_nodes_from(index_dict)
        # unpack the dictionary and initialize the graph with edges and weights
        # it can't be done directly as the weights will not be initialized from the dictionary
        # the idea was the networkx represents labels in this format, so they might also initialize the graph as such but it doesn't
        for edge, weight in self.co.items():
            graph.add_edge(edge[0], edge[1], weight=weight)
        #graph.add_edges_from(co)
        return graph
    
    def add_we_weights(self):
        """
        Reweigh graph by using the word-embeddings of tokens. The new weights are going to be 
        the product of the similarity between two adjacent nodes and the number of co-occurrences
        """
        if self.added_weights:
            print(f"Weights already added!")
            return
        for u, v, data in self.graph.edges(data=True):
            if 'weight' in data:
                data['weight'] *= cosine_similarity(get_word_em(u).reshape(1, -1), get_word_em(v).reshape(1, -1))[0][0]
                data['weight'] = np.round(data['weight'], decimals=3) + 2 # because of negative weights
        #print(f"Added word-embedding weights!")
        self.added_weights = True

    def order_nodes(self, method="degree_centrality", to_print=True):
        """
        Order the nodes of the graph according to some graph centrality algorithm.
        """
        degree_order = None
        if method=="degree_centrality":
            degree_order = nx.degree_centrality(self.graph)
        elif method=="betweenness_centrality":
            degree_order = nx.betweenness_centrality(self.graph, weight="weight")
        elif method=="eigenvector_centrality":
            degree_order = nx.eigenvector_centrality(self.graph, weight="weight")
        elif method=="pagerank":
            degree_order = nx.pagerank(nx.Graph(self.graph), alpha=0.85, weight="weight")
        elif method=="closeness_centrality":
            degree_order = nx.closeness_centrality(self.graph, distance="weight")
        elif method=="katz_centrality":
            degree_order = nx.katz_centrality(self.graph, weight="weight")
        elif method=="hits":
            degree_order, _ = nx.hits(self.graph)
        else:
            raise Exception("Wrong method name!")
        sorted_dict = dict(sorted(degree_order.items(), key=lambda item: item[1], reverse=True))
        if to_print:
            print(f"Method selected: {method}")
            for node, order_value in sorted_dict.items():
                print(f"Node: {node:{20}}   --->    Node Order = {order_value}")
        sorted_dict = {key: round(value, 3) for key, value in sorted_dict.items()}
        return sorted_dict

    def visualize_graph(self):
        """
        Visualize the graph representation of text.
        """
        labels = nx.get_edge_attributes(self.graph,'weight')
        plt.figure(figsize=(8, 8))
        #pos = nx.spring_layout(G, seed=1)  # Layout algorithm (you can try different algorithms)
        #pos = nx.shell_layout(G)
        pos = nx.circular_layout(self.graph) # layout of the graph
        # pos = nx.random_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=10, font_weight='bold')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=labels, font_size=10, font_weight='bold')
        #plt.savefig('paper/figures/graph_example.svg', format='svg')
        plt.show()