import nltk
from nltk.tokenize import word_tokenize, MWETokenizer
from nltk.corpus import stopwords
import lexicons
import re
from gensim.models import Word2Vec
from itertools import combinations
import numpy as np

#nltk.download("stopwords")

def prune_text(text):
    """
    Preprocessing input text.
    """
    mwe_tokenizer = MWETokenizer(lexicons.mwe_list) # custom MWE tokenizer with MWE list input
    tokens = word_tokenize(text) # standard nltk toknizer
    tokens = join_MWE(tokens) # joining upper-case tokens into MWE
    tokens = mwe_tokenizer.tokenize(tokens) # tokenizing text
    #pos_tags = nltk.pos_tag(tokens) # PoS tagging (nouns and adjectives)
    tokens = [token.lower() for token in tokens]
    stop_words = set(stopwords.words('english')) # removing stopwords
    tokens = [token for token in tokens if token.lower() not in stop_words]
    # extracting sentences from input text
    sentences = []
    sentence = []
    for token in tokens:
        sentence.append(token)
        if token in ['.', '?', '!']:
            sentences.append(sentence[0:-1])
            sentence = []
            continue
    
    # TODO: custom stopwords
    tokens = [token for token in tokens if re.match(r'^[a-zA-Z0-9_-]+$', token)] # removing non-alphanumeric tokens
    return tokens, sentences

def load_abstract(name):
    with open('data/' + name + '.txt', 'r') as file:
        abstract = file.read()
    return abstract

def join_MWE(tokens):
    """
    Joins consecutive tokens that start with capital letters in MWE.
    Example: "I go to New York to study" -> "I go to New_York to study". 
    """
    out_tokens = []
    i = 0
    while i < len(tokens):
        if tokens[i][0].isupper(): # is the current token uppercase
            mw_token = tokens[i]
            i = i + 1
            flag = False
            while i < len(tokens) and tokens[i][0].isupper(): # while the next token is uppercase and the iterator is less then the length of the number of tokens
                flag = True
                mw_token = mw_token + '_' + tokens[i]
                i = i + 1
            if flag: # if MWE is found, append it and skip all the consequent steps
                out_tokens.append(mw_token)
                continue
            else:  #i if not, go back on step
                i = i - 1
        out_tokens.append(tokens[i])
        i = i + 1
    return out_tokens

def get_co_matrix(sentences, unique_tokens, window_size=3):
    """
    Calculates co-occurrence matrix for a list of sentences. Choose appropriate window-size
    for your relevant needs.
    """
    n = len(unique_tokens)
    # Making a dictionary of indexes corresponding to tokens which are also key values. Needed for fast list access.
    # Dictionary is in the form: {token0: 0, token1: 1, ...} 
    index_dict = {key: index for index, key in enumerate(unique_tokens)} 
    co_matrix = np.zeros(shape=(n,n))
    def populate_co_matrix(co_matrix, index_dict, window_tokens):
        pairs = list(combinations(window_tokens, 2)) # all pair combinations of a list (ex. [1, 2, 3] => (1, 2), (1, 3), (2,3))
        for pair in pairs:
            t1, t2 = pair
            co_matrix[index_dict[t1]][index_dict[t2]] += 1
            co_matrix[index_dict[t2]][index_dict[t1]] += 1
        return co_matrix    
    
    for sentence in sentences:
        short_sentence = True # in case of sentences that are shorter than the window size
        for i, _ in enumerate(sentence):
            if i + window_size > len(sentence):
                break
            short_sentence = False # means that the sentence is longer than the window size
            window_tokens = sentence[i:i+window_size]
            print(window_tokens)
            co_matrix = populate_co_matrix(co_matrix, index_dict, window_tokens)
        if short_sentence:
            print(sentence) 
            co_matrix = populate_co_matrix(co_matrix, index_dict, sentence)
    return co_matrix, index_dict

