import nltk
from nltk.tokenize import word_tokenize, MWETokenizer
from nltk.corpus import stopwords
import lexicons

#nltk.download("stopwords")

def prune_text(text):
    """
    Preprocessing input text.
    """
    tokenizer = MWETokenizer(lexicons.mwe_list)
    tokens = word_tokenize(text)
    tokens = join_MWE(tokens)
    tokens = tokenizer.tokenize(tokens) # tokenizing text
    pos_tags = nltk.pos_tag(tokens) # PoS tagging (nouns and adjectives)
    #stop_words = set(stopwords.words('english')) # removing stopwords
    return tokens

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
            while tokens[i][0].isupper() and i < len(tokens): # while the next token is uppercase and the iterator is less then the length of the number of tokens
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
