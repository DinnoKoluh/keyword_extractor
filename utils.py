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
            # clean sentence from non-alphanumeric entries
            sentence = [t for t in sentence if re.match(r'^[a-zA-Z0-9_-]+$', t)]
            sentences.append(sentence)
            sentence = []
            continue
    
    # TODO: custom stopwords
    tokens = [token for token in tokens if re.match(r'^[a-zA-Z0-9_-]+$', token)] # removing non-alphanumeric tokens
    return tokens, sentences

def load_abstract(name):
    """
    Load text from file based on the filename.
    """
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

def get_co(sentences, unique_tokens, representation='dictionary', window_size=3):
    """
    Calculates co-occurrence representation for a list of sentences. Choose appropriate window-size
    for your relevant needs. The chosen representation can be either 'dictionary' or 'matrix'.
    """
    n = len(unique_tokens)
    # Making a dictionary of indexes corresponding to tokens which are also key values. Needed for fast list access.
    # Dictionary is in the form: {token0: 0, token1: 1, ...} 
    index_dict = {key: index for index, key in enumerate(unique_tokens)} 
    co = {} # more compact representation of co-occurrence matrix as dictionary
    if representation == 'matrix': co = np.zeros(shape=(n,n))

    def populate(co, window_tokens):
        pairs = list(combinations(window_tokens, 2)) # all pair combinations of a list (ex. [1, 2, 3] => (1, 2), (1, 3), (2,3))
        for pair in pairs:
            if representation == 'dictionary':
                pair = tuple(sorted(pair)) # sort tuple alphabetically for undirected graph
                if pair not in co:
                    co[pair] = 0
                co[pair] += 1
            elif representation == 'matrix':
                t1, t2 = pair
                # do it both ways to get symmetric matrix
                co[index_dict[t1]][index_dict[t2]] += 1
                co[index_dict[t2]][index_dict[t1]] += 1
            else:
                Exception("Wrong graph representation name!")
        return co  
            
    for sentence in sentences:
        short_sentence = True # in case of sentences that are shorter than the window size
        for i, _ in enumerate(sentence):
            if i + window_size > len(sentence):
                break
            short_sentence = False # means that the sentence is longer than the window size
            window_tokens = sentence[i:i+window_size]
            #print(window_tokens)
            co = populate(co, window_tokens)
        if short_sentence:
            #print(sentence) 
            co = populate(co, window_tokens)
    return co, index_dict

