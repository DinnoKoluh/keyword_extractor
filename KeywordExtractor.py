from utils import *

class KeywordExtractor:
    def __init__(self, abstract):
        self.tokens = prune_text(abstract)

    