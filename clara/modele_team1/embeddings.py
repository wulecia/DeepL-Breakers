# offense_classifier/embeddings.py
import numpy as np
import os
import torch
import tensorflow_hub as hub
from sklearn.decomposition import TruncatedSVD

# Chargement de fastText 1M
# Format: texte .vec de fastText (Wikipedia 2017)
def load_fasttext(path="crawl-300d-1M.vec", max_words=50000):
    embeddings = {}
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        next(f)  # skip header
        for i, line in enumerate(f):
            if i >= max_words:
                break
            values = line.rstrip().split(" ")
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# Embedding moyen d'un texte (à partir d'embeddings word2vec/fastText)
def sentence_to_embedding(text, embeddings, dim=300):
    tokens = text.lower().split()
    vectors = [embeddings[word] for word in tokens if word in embeddings]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(dim)

# Universal Sentence Encoder (TF Hub)
USE = None
def load_use():
    global USE
    if USE is None:
        USE = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    return USE

def use_encode(texts):
    model = load_use()
    return model(texts).numpy()

# Combine fastText 1M + custom embedding par concat + SVD
def combine_embeddings(embed_dict1, embed_dict2, dim=300):
    words = set(embed_dict1.keys()).intersection(set(embed_dict2.keys()))
    X = []
    for word in words:
        v1 = embed_dict1[word]
        v2 = embed_dict2[word]
        X.append(np.concatenate([v1, v2]))
    svd = TruncatedSVD(n_components=dim)
    X_svd = svd.fit_transform(X)
    return dict(zip(words, X_svd))