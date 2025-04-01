import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
import numpy as np
from tqdm import tqdm
from os import listdir
import gensim.downloader as api
from gensim.models import Word2Vec, FastText, KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

class Vectorizer:
    
    def __init__(self, type, pre_trained=False, retrain=False, extend_training=False, params={}):
        self.type = type
        self.pre_trained = pre_trained
        self.retrain = retrain
        self.extend_training = extend_training
        self.params = params
        self.vectorizer = None
        self.max_len = None
        self.vectors = None
        self.vocab_length = 0
        self.embedding_dir = os.path.join(os.path.dirname(__file__), "embeddings")

    def word2vec(self, data):
        self.data = data
        if not self.pre_trained:
            if 'word2vec.model' not in listdir(self.embedding_dir) or self.retrain:
                print('\nTraining Word2Vec model...')
                model = self.train_w2v()
            elif self.extend_training:
                print('\nExtending existing Word2Vec model...')
                model = Word2Vec.load(os.path.join(self.embedding_dir, 'word2vec.model'))
                model.train(self.data, total_examples=len(self.data), epochs=5000)
                model.save(os.path.join(self.embedding_dir, 'word2vec.model'))
            else:
                print('\nLoading existing Word2Vec model...')
                model = Word2Vec.load(os.path.join(self.embedding_dir, 'word2vec.model'))
        else:
            model = Word2Vec(self.data, **self.params)

        vectorizer = model.wv
        self.vocab_length = len(vectorizer.key_to_index)
        vectors = [
            np.array([vectorizer[word] for word in tweet if word in vectorizer.key_to_index]).flatten()
            for tweet in tqdm(self.data, 'Vectorizing')
        ]
        self.max_len = max(len(v) for v in vectors)
        self.vectors = [
            np.pad(v, (0, self.max_len - len(v)), 'constant') for v in tqdm(vectors, 'Finalizing')
        ]
        return self.vectors

    def train_w2v(self):
        from gensim.models import Word2Vec
        import logging
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        model = Word2Vec(self.data, sg=1, window=3, vector_size=100, min_count=1, workers=4, epochs=1000, sample=0.01)
        model.save(os.path.join(self.embedding_dir, 'word2vec.model'))
        print("Done training Word2Vec model!")
        return model

    def tfidf(self, data):
        self.data = data
        untokenized = [' '.join(tweet) for tweet in self.data]
        if not self.vectorizer:
            self.vectorizer = TfidfVectorizer(**self.params).fit(untokenized)
        self.vectors = self.vectorizer.transform(untokenized).toarray()
        return self.vectors

    def BoW(self, data):
        self.data = data
        untokenized = [' '.join(tweet) for tweet in self.data]
        if not self.vectorizer:
            self.vectorizer = CountVectorizer(**self.params).fit(untokenized)
        counts = np.array(self.vectorizer.transform(untokenized).toarray()).sum(axis=0)
        mapper = self.vectorizer.vocabulary_
        vectors = [
            np.array([counts[mapper[word]] for word in tweet if word in mapper]).flatten()
            for tweet in tqdm(self.data, 'Vectorizing')
        ]
        self.max_len = max(len(v) for v in vectors)
        self.vectors = [
            np.pad(v, (0, self.max_len - len(v)), 'constant') for v in tqdm(vectors, 'Finalizing')
        ]
        self.vocab_length = len(mapper)
        return self.vectors

    def count(self, data):
        self.data = data
        untokenized = [' '.join(tweet) for tweet in self.data]
        if not self.vectorizer:
            self.vectorizer = CountVectorizer(**self.params).fit(untokenized)
        self.vectors = self.vectorizer.transform(untokenized).toarray()
        self.vocab_length = len(self.vectorizer.vocabulary_)
        return self.vectors

    def glove(self, data):
        self.data = data
        if 'word2vec.model' in listdir(self.embedding_dir):
            print('\n✅ Loading Word2Vec Embeddings from file...')
            model = KeyedVectors.load(os.path.join(self.embedding_dir, 'word2vec.model'))
        elif 'glove-twitter-100.gz' in listdir(self.embedding_dir):
            print('\n✅ Loading Glove Embeddings from file...')
            model = KeyedVectors.load_word2vec_format(os.path.join(self.embedding_dir, 'glove-twitter-100.gz'))
        else:
            print('\n⏬ Loading Glove Embeddings from API...')
            model = api.load('glove-twitter-100')

        vectorizer = model
        self.vocab_length = len(vectorizer.key_to_index)
        vectors = [
            np.array([vectorizer[word] for word in tweet if word in vectorizer.key_to_index]).flatten()
            for tweet in tqdm(self.data, 'Vectorizing')
        ]
        self.max_len = max(len(v) for v in vectors)
        self.vectors = [
            np.pad(v, (0, self.max_len - len(v)), 'constant') for v in tqdm(vectors, 'Finalizing')
        ]
        return self.vectors

    def fasttext(self, data):
        self.data = data
        if not self.pre_trained:
            if 'fasttext.model' not in listdir(self.embedding_dir) or self.retrain:
                print('\nTraining FastText model...')
                model = self.train_ft()
            elif self.extend_training:
                print('\nExtending existing FastText model...')
                model = FastText.load(os.path.join(self.embedding_dir, 'fasttext.model'))
                model.train(self.data, total_examples=len(self.data), epochs=5000)
                model.save(os.path.join(self.embedding_dir, 'fasttext.model'))
            else:
                print('\nLoading existing FastText model...')
                model = FastText.load(os.path.join(self.embedding_dir, 'fasttext.model'))
        else:
            model = FastText(self.data, **self.params)

        vectorizer = model.wv
        self.vocab_length = len(vectorizer.key_to_index)
        vectors = [
            np.array([vectorizer[word] for word in tweet if word in vectorizer.key_to_index]).flatten()
            for tweet in tqdm(self.data, 'Vectorizing')
        ]
        self.max_len = max(len(v) for v in vectors)
        self.vectors = [
            np.pad(v, (0, self.max_len - len(v)), 'constant') for v in tqdm(vectors, 'Finalizing')
        ]
        return self.vectors

    def train_ft(self):
        from gensim.models import FastText
        import logging
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        model = FastText(self.data, sg=1, window=3, vector_size=100, min_count=1, workers=4, epochs=1000, sample=0.01)
        model.save(os.path.join(self.embedding_dir, 'fasttext.model'))
        print("Done training FastText model!")
        return model

    def vectorize(self, data):
        self.data = data
        vectorize_call = getattr(self, self.type, None)
        if vectorize_call:
            return vectorize_call(data)
        else:
            raise Exception(f'{self.type} is not an available function')

    def fit(self, data):
        self.data = data