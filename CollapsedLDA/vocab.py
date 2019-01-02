class Vocabulary:

    def __init__(self, docs):
        self.word_id = dict()
        self.id_word = dict()
        self.word_in_docs = dict()
        self.docs_num = len(docs)
        self.build_vocab(docs)

    def build_vocab(self, documents):
        for doc in documents:
            seen_words = set()
            for token in doc:
                if token not in seen_words:
                    if token not in self.word_in_docs:
                        self.word_in_docs[token] = 1
                    else:
                        self.word_in_docs[token] += 1
                    seen_words.add(token)
                if token not in self.word_id:
                    self.word_id[token] = len(self.word_id)
                    self.id_word[len(self.id_word)] = token

    def filter(self, no_below, no_above):
        words_to_remove = set()
        for word in self.word_in_docs:
            if (self.word_in_docs[word] < no_below or self.word_in_docs[word] > (no_above * self.docs_num)):
                words_to_remove.add(word)

        for word in words_to_remove:
            self.word_in_docs.pop(word)
            self.word_id.pop(word)

        #reassign words' ids to keep numeration consistent
        word_idx = 0
        for word in self.word_id:
            self.word_id[word] = word_idx
            self.id_word[word_idx] = word
            word_idx += 1

    def word_to_id(self, word):
        return self.word_id[word]

    def id_to_word(self, id):
        return self.id_word[id]

    def words_num(self):
        return len(self.word_id)

    def doc_to_bow(self, doc):
        bow = dict()
        for word in doc:
            if word in self.word_id:
                word_id = self.word_id[word]
                if word_id in bow:
                    bow[word_id] += 1
                else:
                    bow[word_id] = 1
        return bow.items()