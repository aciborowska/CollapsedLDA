from gensim_impl import dataset as dt
import gensim
import os
import logging
dirname = os.path.dirname(__file__)

logging.basicConfig(filename='gensim.log',
                    format="%(asctime)s:%(levelname)s:%(message)s",
                    level=logging.INFO)
if __name__ == "__main__":
    dataset_file = '../datasets/abcnews-date-text_short.csv'
    dataset = dt.Dataset([os.path.join(dirname, dataset_file)])

    lda_model = gensim.models.LdaMulticore(dataset.bow_corpus, num_topics=10, id2word=dataset.dictionary, passes=10,
                                           workers=2, alpha=0.2, eta=0.2, random_state=1234, eval_every=1, iterations=1000)

    for idx, topic in lda_model.print_topics(-1):
        print('Topic: {} Word: {}'.format(idx, topic))