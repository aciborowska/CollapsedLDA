import numpy as np
import math
import time

class LDA:

    def __init__(self, K, V, M, alpha=0.2, beta=0.2, iterations=100, convThr=0.1, eval_every=None, start_time=None):
        self.K = K  #number of topics
        self.V = V  #number of words
        self.M = M  #number of documents
        self.iterations = iterations
        self.alpha = np.full(K, alpha) # prior for topic-document distribution: ~ Dir(alpha + lam*theta)
        self.beta = np.full(V, beta)   # prior for word-topic distribution: ~ Dir(beta_fun + miu*phi)
        self.z_m_n = np.full((M, V), -1, dtype=int)       # topic to word assignment per each document
        self.c_k_m_x = np.zeros((M, K), dtype=int)     # matrix counting how many words from a document are assigned to a topic (disregard word-dimension)
        self.c_k_x_n = np.zeros((K, V), dtype=int)     # matrix counting how many words are assigned to a topic (disregard document-dimension)
        self.c_k = np.zeros(K, dtype=int)
        self.convThr = convThr
        self.eval_every = eval_every
        self.results = []
        if start_time is None:
            self.start_time = time.time()
        else:
            self.start_time = start_time


    def run(self, docs, filepath=None):
        # initialize z
        for m, doc in enumerate(docs):
            for word_tuple in doc:
                word_id = word_tuple[0]
                word_count = word_tuple[1]
                z = np.random.randint(0, self.K)
                self.z_m_n[m, word_id] = int(z)
                self.c_k_m_x[m, z] += word_count
                self.c_k_x_n[z, word_id] += word_count
                self.c_k[z] += word_count

        perplex = self.perplexity(docs, self.theta(self.c_k_m_x), self.phi())
        self.results.append(('train init', 0, perplex))
        print("Initial perplexity: {0}".format(perplex))

        for iteration in range(0, self.iterations):
            #print('Iteration {0}'.format(iteration))
            for m, doc in enumerate(docs):
                doc_len = 0
                for word_tuple in doc:
                    doc_len += word_tuple[1]

                for word_tuple in doc:
                    word_id = word_tuple[0]
                    word_count = word_tuple[1]
                    z = self.z_m_n[m, word_id]
                    self.c_k_m_x[m, z] -= word_count
                    self.c_k_x_n[z, word_id] -= word_count
                    self.c_k[z] -= word_count

                    left = (self.c_k_x_n[:, word_id] + self.beta[0]) / (self.c_k + (self.beta[0] * self.V))
                    right = (self.c_k_m_x[m, :] + self.alpha[0]) / ((doc_len - word_count) + (self.alpha[0] * self.K))

                    p_z = left * right

                    new_z = np.random.multinomial(1, p_z / np.sum(p_z)).argmax()

                    self.z_m_n[m, word_id] = int(new_z)
                    self.c_k_m_x[m, new_z] += word_count
                    self.c_k_x_n[new_z, word_id] += word_count
                    self.c_k[new_z] += word_count

            if self.eval_every is not None and ((iteration > 100 and iteration % self.eval_every == 0) or iteration<100
                    and iteration % 5 == 0):
                perplex = self.perplexity(docs, self.theta(self.c_k_m_x),self.phi())
                self.results.append(('train', iteration, perplex))
                self.save_execution_time(iteration)
                if filepath is not None:
                    self.save_results_to_file(filepath)
                print("Iteration {0} - perplexity = {1}".format(iteration, perplex))


        perplex = self.perplexity(docs, self.theta(self.c_k_m_x), self.phi())
        self.results.append(('train', self.iterations, perplex))
        print("Final train set perplexity = {0}".format(perplex))

    def phi(self):
        phi = np.zeros((self.K, self.V))
        for k in range(0, self.K):
            for v in range(0, self.V):
                phi[k, v] = (self.c_k_x_n[k, v] + self.beta[v]) / ((self.V * self.beta[v]) + np.sum(self.c_k_x_n[k, :]))
        return phi

    def theta(self, c_k_m_x):
        theta = np.zeros((self.M, self.K))
        for m in range(0, len(c_k_m_x)):
            for k in range(0, self.K):
                theta[m, k] = (c_k_m_x[m, k] + self.alpha[k]) / ((self.K * self.alpha[k]) + np.sum(c_k_m_x[m, :]))
        return theta

    def test(self, docs, iterations_test):
        z_m_n = np.zeros((len(docs), self.V), dtype=int)
        c_k_m_x = np.zeros((len(docs), self.K), dtype=int)
        for m, doc in enumerate(docs):
            for word_tuple in doc:
                word_id = word_tuple[0]
                word_count = word_tuple[1]
                z = np.random.randint(0, self.K)
                z_m_n[m, word_id] = int(z)
                c_k_m_x[m, z] += word_count

        perplex = self.perplexity(docs, self.theta(c_k_m_x), self.phi())
        self.results.append(("test", 0, perplex))
        print("Initial test set perplexity: {0}".format(perplex))

        for iteration in range(0, iterations_test):
            #print("Iteration {0}".format(iteration))
            for m, doc in enumerate(docs):
                doc_len = 0
                for word_tuple in doc:
                    doc_len += word_tuple[1]

                for word_tuple in doc:
                    word_id = word_tuple[0]
                    word_count = word_tuple[1]
                    z = z_m_n[m, word_id]
                    c_k_m_x[m, z] -= word_count

                    left = (self.c_k_x_n[:, word_id] + self.beta[0]) / (self.c_k + (self.beta[0] * self.V))
                    right = (self.c_k_m_x[m, :] + self.alpha[0]) / ((doc_len - word_count) + (self.alpha[0] * self.K))
                    p_z = left * right

                    new_z = np.random.multinomial(1, p_z / np.sum(p_z)).argmax()

                    z_m_n[m, word_id] = int(new_z)
                    c_k_m_x[m, new_z] += word_count

        perplex = self.perplexity(docs, self.theta(c_k_m_x), self.phi())
        self.results.append(('test', iterations_test, perplex))
        print("Test set perplexity = {0}".format(perplex))

    def perplexity(self, docs, theta_m, phi_m):
        sum_nom = 0
        sum_docs_len = 0
        for m, doc in enumerate(docs):
            sum_docs_len += len(doc)
            for word_tuple in doc:
                sum_nom -= np.log(np.inner(phi_m[:, word_tuple[0]], theta_m[m]))

        return math.exp(sum_nom / sum_docs_len)

    def save_execution_time(self, iteration, current_time=None):
        if current_time is None:
            current_time = time.time()
        self.results.append(("time", iteration, current_time - self.start_time))

    def save_results_to_file(self, filepath):
        with open(filepath, 'w') as f:
            for result in self.results:
                f.write('{0},{1},{2}\n'.format(result[0], result[1], result[2]))
