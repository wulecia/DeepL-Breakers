# offense_classifier/run.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from preprocessing import normalize_text
from features import extract_features
from embeddings import load_fasttext, sentence_to_embedding
from models import RFModel
from ensemble import SoftVotingEnsemble

# === 1. Chargement du dataset ===
DATA_PATH = "../datasets/trial-data/offenseval-trial.txt"
df = pd.read_csv(DATA_PATH, sep='\t', header=None)
df.columns = ["text", "label_A", "label_B", "label_C"]

# Pour la sous-tâche A
texts = df["text"].values
labels = df["label_A"].map({"NOT": 0, "OFF": 1}).values

# === 2. Prétraitement ===
norm_texts = [normalize_text(t) for t in texts]

# === 3. Extraction des features ===
X_feat = extract_features(norm_texts)

# === 4. Embeddings ===
print("[INFO] Chargement des embeddings fastText...")
embeddings = load_fasttext("../embeddings/crawl-300d-1M.vec")
X_embed = np.vstack([sentence_to_embedding(t, embeddings) for t in norm_texts])

# === 5. Concaténation Features + Embeddings ===
X = np.hstack([X_embed, X_feat])

# === 6. Split Train/Test ===
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# === 7. Modèle (Random Forest) ===
rf = RFModel()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("\n=== Résultats Random Forest ===")
print(classification_report(y_test, y_pred, digits=3))

# === 8. (optionnel) Ensemble avec SVM ou autres ===
# from models import SVMModel
# ensemble = SoftVotingEnsemble([
#     ('rf', rf.model),
#     ('svm', SVMModel().model)
# ])
# ensemble.fit(X_train, y_train)
# y_pred_ens = ensemble.predict(X_test)
# print("\n=== Résultats Ensemble ===")
# print(classification_report(y_test, y_pred_ens))
