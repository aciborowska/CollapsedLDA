import pandas as pd
import gensim


class Corpus:

    def __init__(self, file):
        df = pd.read_csv(file)
        self.documents = df['headline_text'].map(self.preprocess)
        # lines = []
        # with open(file, 'r') as f:
        #     line = f.readline()
        #     while line:
        #         lines.append(line)
        #         line =f.readline()
        # df = pd.DataFrame({'col': lines})
        # self.documents = df['col'].map(self.preprocess)

    def preprocess(self, text):
        tokens = []
        for token in gensim.utils.simple_preprocess(text, min_len=2, max_len=100):
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 2:
                tokens.append(token)
        return tokens

    def get_docs(self):
        return self.documents
