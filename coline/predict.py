#!/usr/bin/env python
# coding: utf-8

# In[41]:


import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from sklearn.metrics import accuracy_score, f1_score, classification_report
from coline_model import Coline, Paola
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[42]:


# === Device (CPU ou GPU) ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Chargement des modèles ===
model_A = Coline(task="A", model_name="roberta-base", num_labels=2)  # or whatever NUM_LABELS["A"] is
state_dictA = torch.load("best_colinemodel_A_roberta-base.pth")
state_dictA.pop('loss_fn.weight', None)
model_A.load_state_dict(state_dictA, strict=False)
model_A.to(device)
tokenizer_A = AutoTokenizer.from_pretrained("roberta-base")

model_B = Coline(task="B", model_name="GroNLP/hateBERT", num_labels=3)  # or whatever NUM_LABELS["A"] is
state_dictB = torch.load("best_colinemodel_B_hateBERT.pth")
state_dictB.pop('loss_fn.weight', None)
model_B.load_state_dict(state_dictB, strict=False)
model_B.to(device)
tokenizer_B = AutoTokenizer.from_pretrained("GroNLP/hateBERT")

model_C = Coline(task="C", model_name="GroNLP/hateBERT", num_labels=2)  # or whatever NUM_LABELS["A"] is
state_dictC = torch.load("best_colinemodel_C_hateBERT.pth")
state_dictC.pop('loss_fn.weight', None)
model_C.load_state_dict(state_dictC, strict=False)
model_C.to(device)
tokenizer_C = AutoTokenizer.from_pretrained("GroNLP/hateBERT")


# In[40]:


# === Chargement du fichier test HASOC ===
df = pd.read_csv("../hasoc_model/hasoc_dataset/test.tsv", sep="\t")
df.columns = ["id", "text", "label_A_gold", "label_B_gold", "label_C_gold"]

df = df[0:50]

tweets = df["text"].tolist()

# === Fonction de prédiction ===
def predict(texts, extra_features, model, tokenizer, device, max_length=512):
    model.eval()  # Set the model to evaluation mode
    
    # Tokenize the input texts
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    
    # Move the inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Move the model to the device if not already
    model.to(device)
    inputs.pop("token_type_ids", None)

    # Run the prediction with no gradient computation
    with torch.no_grad():
        logits = model(extra_features=extra_features, **inputs)
        
        #logits = outputs["logits"]
        
    # Get predictions from the logits
    predictions = torch.argmax(logits, dim=-1)
    return predictions



# In[36]:


model_paola = Paola().to(device)
model_paola.load_state_dict(torch.load("../paola/model2_loaded.pth", map_location=device, weights_only=True))

print("model2_loaded.pth loaded and ready to use!")

tokenizer_paola = AutoTokenizer.from_pretrained("distilbert-base-uncased")


# In[37]:


# === Étape 1 : prédiction Task A ===
encodings_paolaA = tokenizer_paola(tweets, truncation=True, padding=True, max_length=128, return_tensors="pt")

model_paola.eval()
input_ids_paolaA = encodings_paolaA['input_ids'].to(device)
attention_mask_paolaA = encodings_paolaA['attention_mask'].to(device)

with torch.no_grad():
    preds_numA, preds_binA = model_paola(input_ids=input_ids_paolaA, attention_mask=attention_mask_paolaA)

preds_numA = preds_numA.cpu().numpy()
preds_binA = preds_binA.cpu().numpy()
preds_binA = (preds_binA > 0.5).astype(int)

extra_featuresA = np.concatenate([preds_numA, preds_binA], axis=1)
extra_features_tensorA = torch.tensor(extra_featuresA, dtype=torch.float32).to(device)
y_pred_A = predict(tweets, extra_features_tensorA, model_A, tokenizer_A, device)
off_mask = (y_pred_A == 1)
#print(y_pred_A)


# In[18]:


# === Étape 2 : Task B (sur tweets HOF) ===
tweets_B = [t for i, t in enumerate(tweets) if off_mask[i]]
encodings_paolaB = tokenizer_paola(tweets_B, truncation=True, padding=True, max_length=128, return_tensors="pt")

model_paola.eval()
input_ids_paolaB = encodings_paolaB['input_ids'].to(device)
attention_mask_paolaB = encodings_paolaB['attention_mask'].to(device)

with torch.no_grad():
    preds_numB, preds_binB = model_paola(input_ids=input_ids_paolaB, attention_mask=attention_mask_paolaB)

preds_numB = preds_numB.cpu().numpy()
preds_binB = preds_binB.cpu().numpy()
preds_binB = (preds_binB > 0.5).astype(int)

extra_featuresB = np.concatenate([preds_numB, preds_binB], axis=1)
extra_features_tensorB = torch.tensor(extra_featuresB, dtype=torch.float32).to(device)
y_pred_B_partial = predict(tweets_B, extra_features_tensorB, model_B, tokenizer_B, device)
#print(y_pred_B_partial)


# In[48]:


# === Étape 3 : Task C (sur tweets B == HATE) ===
tin_mask = (y_pred_B_partial == 0)  # HATE = 0
tweets_C = [t for i, t in enumerate(tweets_B) if tin_mask[i]]

encodings_paolaC = tokenizer_paola(tweets_C, truncation=True, padding=True, max_length=128, return_tensors="pt")

model_paola.eval()
input_ids_paolaC = encodings_paolaC['input_ids'].to(device)
attention_mask_paolaC = encodings_paolaC['attention_mask'].to(device)

with torch.no_grad():
    preds_numC, preds_binC = model_paola(input_ids=input_ids_paolaC, attention_mask=attention_mask_paolaC)

preds_numC = preds_numC.cpu().numpy()
preds_binC = preds_binC.cpu().numpy()
preds_binC = (preds_binC > 0.5).astype(int)

extra_featuresC = np.concatenate([preds_numC, preds_binC], axis=1)
extra_features_tensorC = torch.tensor(extra_featuresC, dtype=torch.float32).to(device)
y_pred_C_partial = predict(tweets_C, extra_features_tensorC, model_C, tokenizer_C, device)
#print(y_pred_C_partial)


# In[49]:


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


# In[51]:


# === Évaluation Task A ===
print("\n=== Task A ===")
y_pred_A = y_pred_A.cpu().numpy()
gold_A = gold_A.cpu().numpy() if isinstance(gold_A, torch.Tensor) else gold_A

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


# In[ ]:




