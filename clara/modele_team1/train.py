# offense_classifier/train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessing import normalize_text
from features import extract_features
from embeddings import load_fasttext, sentence_to_embedding
from models import RFModel

# === Load dataset ===
path = "../datasets/trial-data/offenseval-trial.txt"
df = pd.read_csv(path, sep='\t', header=None, names=['text', 'label', 'subtask_b', 'subtask_c'])

# === Clean and preprocess ===
df['text_clean'] = df['text'].apply(normalize_text)

# === Extract features and embeddings ===
ling_features = extract_features(df['text_clean'].tolist())

print("Chargement des embeddings fastText...")
ft = load_fasttext("crawl-300d-1M.vec")  # à adapter si chemin différent
embeddings = np.array([sentence_to_embedding(text, ft) for text in df['text_clean']])

X = np.hstack((embeddings, ling_features))
y = df['label'].values

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train model ===
clf = RFModel()
clf.fit(X_train, y_train)

# === Evaluate ===
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
