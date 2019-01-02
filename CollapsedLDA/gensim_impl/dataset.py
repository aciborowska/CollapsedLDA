import gensim
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import numpy as np
import pandas as pd
import nltk
nltk.download('wordnet')

np.random.seed(1234)


class Dataset:

    def __init__(self, files):
        data = pd.concat((pd.read_csv(f) for f in files))
        data_text = data[['paper_text']]
        #data_text['index'] = data_text.index
        self.documents = data_text['paper_text'].map(self.preprocess)
        self.dictionary = gensim.corpora.Dictionary(self.documents)
        self.dictionary.filter_extremes(no_below=2000, no_above=0.7, keep_n=539)
        self.bow_corpus = [self.dictionary.doc2bow(doc) for doc in self.documents]

        #self.tfidf = models.TfidfModel(self.bow_corpus)
        #self.corpus_tfidf = self.tfidf[self.bow_corpus]

    def lemmatize_stemming(self, text):
        # stemmer = SnowballStemmer("english")
        # return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
        return text


    def preprocess(self, text):
        result = []
        for token in gensim.utils.simple_preprocess(text):
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 2:
                result.append(self.lemmatize_stemming(token))
        return result