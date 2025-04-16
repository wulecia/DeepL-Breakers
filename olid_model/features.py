# offense_classifier/features.py
import re
import numpy as np
from typing import List
from collections import Counter

# Expressions offensantes connues (extrait minimal pour test)
BLACKLIST = ["fuck", "shit", "bitch", "nigger", "cunt"]
BLACKLIST_CONTEXT = ["bloody", "pearl necklace"]

# Compter les mots d'une phrase présents dans la blacklist
def count_blacklisted_words(text: str) -> int:
    tokens = text.lower().split()
    return sum(1 for token in tokens if token in BLACKLIST + BLACKLIST_CONTEXT)

# Compter les majuscules, caractères spéciaux, etc.
def count_special_features(text: str) -> List[int]:
    num_uppercase = sum(1 for c in text if c.isupper())
    num_special = sum(1 for c in text if not c.isalnum() and not c.isspace())
    return [num_uppercase, num_special]

# Combiner toutes les features linguistiques
def extract_features(texts: List[str]) -> np.ndarray:
    features = []
    for text in texts:
        f1 = count_blacklisted_words(text)
        f2, f3 = count_special_features(text)
        features.append([f1, f2, f3])
    return np.array(features)
