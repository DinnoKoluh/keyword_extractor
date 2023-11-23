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
        # n = 20
        # random_subset = df.sample(n=n, random_state=1)
        # random_subset.to_csv(f'data/CS_subset_{n}.csv')
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

def make_keyword_metrics(methods: dict, path_to_file: str):
    """
    Make csv of the graph with the respective ranking algorithm.
    """
    joined, abstracts, titles, keywords = get_data(path_to_file, version="CS")
    df = pd.read_csv(path_to_file)

    predicted_keywords_dict = {key: [] for key in methods.values()}
    for i, abstract in enumerate(joined):
        ke = KeywordExtractor(abstract)
        ke.add_we_weights()
        num_of_true_keywords = len(keywords[i])
        for key in methods.keys():
            try:
                keyword_dict = ke.order_nodes(method=methods[key], to_print=False)
                ke_keywords = list(keyword_dict.keys())[0:num_of_true_keywords]
                predicted_keywords_dict[methods[key]].append(ke_keywords)
            except Exception as e:
                print(e)
                predicted_keywords_dict[methods[key]].append(["ERROR"])
        print(f"Finished {i+1}. abstract")
    new_data = pd.DataFrame(predicted_keywords_dict)

    result_df = pd.concat([df, new_data], axis=1)
    result_df.to_csv('data/metrics_per_abstract.csv', index=False)