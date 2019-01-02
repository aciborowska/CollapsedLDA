import corpus as cor
import vocab as vb
import numpy as np
import os
import lda
import random
import time

random.seed(1234)
np.random.seed(1234)

dirname = os.path.dirname(__file__)

if __name__ == "__main__":
    topicsNo = 5
    alpha = 0.5
    beta = 0.2

    iterations = 1000
    iterations_test = 100
    cvgThreshold = 0.1
    training_ratio = 0.9
    eval_every = 100

    no_below = 2000
    no_above = 1.0

    dataset_dir = 'datasets/'
    dataset_file = 'abcnews-date-text.csv'
    result_dir = 'results/'
    filename_pattern = 'result_dataset={0}_k={1}_V={2}_iter={3}.csv'

    start_time = time.time()

    corpus = cor.Corpus(os.path.join(dirname, dataset_dir + dataset_file))

    vocabulary = vb.Vocabulary(corpus.get_docs())
    print("Number of documents {0}".format(vocabulary.docs_num))
    print("Number of words {0}".format(len(vocabulary.word_id)))

    vocabulary.filter(no_below=no_below, no_above=no_above)
    print("Number of words after filtering {0}".format(len(vocabulary.word_id)))

    docs = []
    word_no = 0
    for i, doc in enumerate(corpus.get_docs()):
        bow = vocabulary.doc_to_bow(doc)
        for tupel in bow:
            word_no += tupel[1]
        docs.append(bow)
    print('number of words {0}'.format(word_no))
    # Train LDA
    print("Running LDA...")
    training_part_end = int(len(docs) * training_ratio)
    model = lda.LDA(topicsNo, len(vocabulary.word_id), len(docs[0:training_part_end]), alpha=alpha, beta=beta,
                    iterations=iterations, convThr=cvgThreshold, eval_every=eval_every, start_time=start_time)
    model.run(docs[0:training_part_end],result_dir + filename_pattern.format(dataset_file, topicsNo,
                                                                    len(vocabulary.word_id), iterations))
    # Test LDA
    model.test(docs[training_part_end:], iterations_test)

    end_time = time.time()

    model.save_execution_time(iterations, end_time)
    model.save_results_to_file(result_dir + filename_pattern.format(dataset_file, topicsNo,
                                                                    len(vocabulary.word_id), iterations))

    # Print topics
    topics_word_dist = model.phi()
    for topic in range(0, topicsNo):
        print("Topic {0}".format(topic))
        word_ids = np.argpartition(topics_word_dist[topic], -4)[-4:]
        for word_id in word_ids:
            print('{0}: {1}'.format(vocabulary.id_to_word(word_id), topics_word_dist[topic, word_id]))


