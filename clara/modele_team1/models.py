# offense_classifier/models.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# Random Forest
class RFModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

# SVM
class SVMModel:
    def __init__(self):
        self.model = SVC(probability=True, kernel='linear', random_state=42)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

# Simple Transformer (optionnel)
class SimpleTransformer(nn.Module):
    def __init__(self, input_dim=300, n_heads=4, ff_dim=128):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=n_heads, batch_first=True)
        self.fc1 = nn.Linear(input_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, 2)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        x = F.relu(self.fc1(attn_output.mean(dim=1)))
        return self.fc2(x)

# Placeholder GPT
class GPTWrapper:
    def __init__(self):
        self.model = None  # à remplacer par pipeline transformers

    def fit(self, X, y):
        pass

    def predict(self, X):
        return [0]*len(X)

    def predict_proba(self, X):
        return [[0.5, 0.5] for _ in X]


# BERT/Roberta Classifier via Transformers
class TransformerModel:
    def __init__(self, model_name="roberta-large", num_labels=2):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.model.eval()

    def predict(self, texts):
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        return torch.argmax(logits, dim=1).numpy()

    def predict_proba(self, texts):
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=1).numpy()
        return probs
