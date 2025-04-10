# offense_classifier/preprocessing.py
import re
import string
import numpy as np
from sklearn.utils import resample

# Remplace les noms d'utilisateurs, les liens et les caractères spéciaux, et réduit à la forme canonique
def normalize_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)                      # remove URLs
    text = re.sub(r"@\w+", "<USER>", text)                   # anonymize users
    text = re.sub(r"[^a-z0-9\s]", "", text)                  # remove non-alphanumeric chars
    text = re.sub(r"\s+", " ", text).strip()                 # remove multiple spaces
    return text

# Oversample pour équilibrer les classes
def oversample_data(X, y):
    classes = np.unique(y)
    max_count = max([sum(y == c) for c in classes])
    X_resampled, y_resampled = [], []
    for c in classes:
        X_c = X[y == c]
        y_c = y[y == c]
        X_os, y_os = resample(X_c, y_c, replace=True, n_samples=max_count, random_state=42)
        X_resampled.append(X_os)
        y_resampled.append(y_os)
    return np.vstack(X_resampled), np.hstack(y_resampled)
