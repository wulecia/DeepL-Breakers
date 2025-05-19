import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report

# === Device (CPU ou GPU) ===
device = torch.device("cpu")  # ou "cuda" si tu veux tester sur GPU

# === Chargement des modèles ===
model_A = AutoModelForSequenceClassification.from_pretrained("./coco/best_model_A_roberta-base").to(device)
tokenizer_A = AutoTokenizer.from_pretrained("roberta-base")

model_B = AutoModelForSequenceClassification.from_pretrained("./coco/best_model_B_hateBERT").to(device)
tokenizer_B = AutoTokenizer.from_pretrained("GroNLP/hateBERT")

model_C = AutoModelForSequenceClassification.from_pretrained("./coco/best_model_C_hateBERT").to(device)
tokenizer_C = AutoTokenizer.from_pretrained("GroNLP/hateBERT")



# === Chargement du fichier test HASOC ===
df = pd.read_csv("hasoc_dataset/test.tsv", sep="\t")
df.columns = ["id", "text", "label_A_gold", "label_B_gold", "label_C_gold"]
tweets = df["text"].tolist()

# === Fonction de prédiction ===
def predict(texts, model, tokenizer, max_length=128):
    model.eval()
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    return torch.argmax(logits, dim=1).cpu().numpy()

# === Étape 1 : prédiction Task A ===
y_pred_A = predict(tweets, model_A, tokenizer_A)
off_mask = (y_pred_A == 1)

# === Étape 2 : Task B (sur tweets HOF) ===
tweets_B = [t for i, t in enumerate(tweets) if off_mask[i]]
y_pred_B_partial = predict(tweets_B, model_B, tokenizer_B)

# === Étape 3 : Task C (sur tweets B == HATE) ===
tin_mask = (y_pred_B_partial == 0)  # HATE = 0
tweets_C = [t for i, t in enumerate(tweets_B) if tin_mask[i]]
y_pred_C_partial = predict(tweets_C, model_C, tokenizer_C)


# === Reconstruction des prédictions texte ===
pred_A = ["HOF" if x == 1 else "NOT" for x in y_pred_A]
pred_B, pred_C = ["NULL"] * len(tweets), ["NULL"] * len(tweets)

b_idx = 0
for i, is_off in enumerate(off_mask):
    if is_off:
        pred_B[i] = ["HATE", "OFFN", "PRFN"][y_pred_B_partial[b_idx]]
        b_idx += 1

c_idx = 0
for i, is_off in enumerate(off_mask):
    if is_off and pred_B[i] == "HATE":
        pred_C[i] =["UNT", "TIN"][y_pred_C_partial[c_idx]]
        c_idx += 1

# === Ajout au DataFrame ===
df["pred_A"] = pred_A
df["pred_B"] = pred_B
df["pred_C"] = pred_C

# === Convertir les labels gold ===
gold_A = [1 if label == "HOF" else 0 for label in df["label_A_gold"]]
gold_B = [ ["HATE", "OFFN", "PRFN"].index(label) if label in ["HATE", "OFFN", "PRFN"] else -1 for label in df["label_B_gold"] ]
gold_C = [ ["UNT", "TIN"].index(label) if label in ["UNT", "TIN"] else -1 for label in df["label_C_gold"] ]



# === Évaluation Task A ===
print("\n=== Task A ===")
print("Accuracy:", accuracy_score(gold_A, y_pred_A))
print("F1-score:", f1_score(gold_A, y_pred_A, average="macro"))
print(classification_report(gold_A, y_pred_A, target_names=["NOT", "HOF"]))

# === Évaluation Task B ===
gold_B_eval = [g for i, g in enumerate(gold_B) if off_mask[i] and g != -1]
pred_B_eval = [ ["HATE", "OFFN", "PRFN"].index(b) for i, b in enumerate(pred_B) if off_mask[i] and gold_B[i] != -1 ]

print("\n=== Task B ===")
print("Accuracy:", accuracy_score(gold_B_eval, pred_B_eval))
print("F1-score:", f1_score(gold_B_eval, pred_B_eval, average="macro"))
print(classification_report(gold_B_eval, pred_B_eval, target_names=["HATE", "OFFN", "PRFN"]))

# === Évaluation Task C ===
gold_C = [ ["UNT", "TIN"].index(label) if label in ["UNT", "TIN"] else -1 for label in df["label_C_gold"] ]
gold_C_eval = [g for i, g in enumerate(gold_C) if off_mask[i] and pred_B[i] == "HATE" and g != -1]
pred_C_eval = [ ["UNT", "TIN"].index(c) for i, c in enumerate(pred_C) if off_mask[i] and pred_B[i] == "HATE" and gold_C[i] != -1 ]

print("\n=== Task C ===")
print("Accuracy:", accuracy_score(gold_C_eval, pred_C_eval))
print("F1-score:", f1_score(gold_C_eval, pred_C_eval, average="macro"))
print(classification_report(gold_C_eval, pred_C_eval, target_names=["UNT", "TIN"]))
