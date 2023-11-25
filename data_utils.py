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

def is_predicted_keyword_in_list(predicted_keyword, true_keywords):
    split_predicted_keyword = predicted_keyword.split('_')
    lowercased_true_keywords = [word.lower() for word in true_keywords] # splitting by ' '
    #print(lowercased_true_keywords)
    for keyword in split_predicted_keyword:
        for true_keyword in lowercased_true_keywords:
            if keyword in true_keyword:
                return True
    return False

def get_prf(true_keywords: list, predicted_keywords: list):
    tp, fp, fn = 0, 0, 0

    for predicted_keyword in predicted_keywords:
        if is_predicted_keyword_in_list(predicted_keyword, true_keywords):
        #if predicted_keyword in true_keywords: # for word level implement
            tp += 1
        else:
            fp += 1
    
    fn = len(true_keywords) - tp
    # for true_keyword in true_keywords:
    #     if true_keyword not in predicted_keywords:
    #         fn += 1
    
    return tp/(tp + fp), tp/(tp + fn), tp/(tp + 0.5*(fp+fn))


def make_keyword_metrics(methods: dict, path_to_file: str, window_size: int):
    """
    Make csv of the graph with the respective ranking algorithm.
    """
    joined, abstracts, titles, keywords = get_data(path_to_file, version="CS")
    df = pd.read_csv(path_to_file)

    predicted_keywords_dict = {}
    #predicted_keywords_dict = {key: [] for key in methods.values()}
    for value in methods.values():
        predicted_keywords_dict[value] = []
        predicted_keywords_dict[f'{value}_prf'] = []

    metrics_dict = {}
    for value in methods.values():
        metrics_dict[f'{value}_p'] = []
        metrics_dict[f'{value}_r'] = []
        metrics_dict[f'{value}_f'] = []

    for i, abstract in enumerate(joined):
        ke = KeywordExtractor(abstract, window_size)
        ke.add_we_weights()
        num_of_true_keywords = len(keywords[i])
        for key in methods.keys():
            try:
                keyword_dict = ke.order_nodes(method=methods[key], to_print=False)
                ke_keywords = list(keyword_dict.keys())[0:num_of_true_keywords+1] # add one extra keyword
                predicted_keywords_dict[methods[key]].append(ke_keywords)
                p, r, f = get_prf(keywords[i], ke_keywords) # calculating p, r, f
                #predicted_keywords_dict[f'{methods[key]}_prf'].append(f'p = {p:.3f}, r = {r:.3f}, f = {f:.3f}')
                predicted_keywords_dict[f'{methods[key]}_prf'].append([p, r, f])
                metrics_dict[f'{methods[key]}_p'].append(p)
                metrics_dict[f'{methods[key]}_r'].append(r)
                metrics_dict[f'{methods[key]}_f'].append(f)
            except Exception as e:
                print(f'Failed on abstract: {i}, \n error: {e}')
                predicted_keywords_dict[methods[key]].append(["ERROR"])
                predicted_keywords_dict[f'{methods[key]}_prf'].append(f'error')
        if i % 10 == 0:
            print(f"Finished {i+1}. abstract")
            if i == 100: 
                break
    new_data = pd.DataFrame(predicted_keywords_dict)

    result_df = pd.concat([df, new_data], axis=1)
    result_df.to_csv('data/metrics_per_abstract.csv', index=False)

    return metrics_dict