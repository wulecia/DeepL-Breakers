#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch.nn as nn
#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')

#torch.cuda.empty_cache()
#torch.cuda.ipc_collect()
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# In[2]:


class Coline(nn.Module):
    def __init__(self, task, model_name=None, num_labels=None, class_weights=None):
        super().__init__()
        self.task = task
        self.model_name = model_name or MODEL_NAMES[task]
        self.num_labels = num_labels or NUM_LABELS[task]
        self.class_weights = class_weights
        self._keys_to_ignore_on_save = []

        self.transformer = AutoModel.from_pretrained(self.model_name)
        hidden_size = self.transformer.config.hidden_size  # usually 768

        self.extra_feat_size = 13  # 8 numerical + 5 binary

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + self.extra_feat_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, self.num_labels)
        )

        self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights) if self.class_weights is not None else nn.CrossEntropyLoss()

    def freeze_transformer(self):
        for param in self.transformer.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask, extra_features, labels=None):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS token

        # Concatenate CLS embedding with extra features
        combined = torch.cat((pooled_output, extra_features), dim=1)

        logits = self.classifier(combined)

        if labels is not None:
            loss = self.loss_fn(logits, labels)

            return {"logits": logits, "loss": loss, "labels": labels}
        return {"logits": logits}


# In[3]:


# === Device (CPU ou GPU) ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Chargement des modèles ===
model_A = Coline(task="A", model_name="roberta-base", num_labels=2)  # or whatever NUM_LABELS["A"] is
state_dictA = torch.load("../models_trained/best_colinemodel_A_roberta-base.pth")
state_dictA.pop('loss_fn.weight', None)
model_A.load_state_dict(state_dictA, strict=False)
model_A.to(device)
tokenizer_A = AutoTokenizer.from_pretrained("roberta-base")

model_B = Coline(task="B", model_name="GroNLP/hateBERT", num_labels=3)  # or whatever NUM_LABELS["A"] is
state_dictB = torch.load("../models_trained/best_colinemodel_B_hateBERT.pth")
state_dictB.pop('loss_fn.weight', None)
model_B.load_state_dict(state_dictB, strict=False)
model_B.to(device)
tokenizer_B = AutoTokenizer.from_pretrained("GroNLP/hateBERT")

model_C = Coline(task="C", model_name="GroNLP/hateBERT", num_labels=2)  # or whatever NUM_LABELS["A"] is
state_dictC = torch.load("../models_trained/best_colinemodel_C_hateBERT.pth")
state_dictC.pop('loss_fn.weight', None)
model_C.load_state_dict(state_dictC, strict=False)
model_C.to(device)
tokenizer_C = AutoTokenizer.from_pretrained("GroNLP/hateBERT")


# In[4]:


# === Chargement du fichier test HASOC ===
df = pd.read_csv("../hasoc_model/hasoc_dataset/hasoc_dataset_with_features_test.tsv", sep="\t")
df.columns = ["text", "label_A", "label_B", "label_C",
              'label_A_enc',  'label_B_enc',  'label_C_enc',
              'sentiment', 'respect', 'insult', 'humiliate', 'status', 'dehumanize', 'attack_defend', 'hatespeech',
              'target_race', 'target_religion', 'target_origin', 'target_gender', 'target_sexuality'] 

tweets = df["text"].tolist()

# === Fonction de prédiction ===
def predict(texts, extra_features, model, tokenizer, device, max_length=512):
    
    if len(texts) == 0:
        return torch.tensor([], dtype=torch.long)
    
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
        outputs = model(extra_features=extra_features, **inputs)
        
        logits = outputs["logits"]
        
    # Get predictions from the logits
    predictions = torch.argmax(logits, dim=-1)
    return predictions



# In[5]:


# === Étape 1 : prédiction Task A ===
preds_numA = df[['sentiment', 'respect', 'insult', 'humiliate', 'status',
                  'dehumanize', 'attack_defend', 'hatespeech']]
preds_binA = df[['target_race', 'target_religion', 'target_origin', 'target_gender',
                'target_sexuality']]

extra_featuresA = np.concatenate([preds_numA, preds_binA], axis=1)
extra_features_tensorA = torch.tensor(extra_featuresA, dtype=torch.float32).to(device)
y_pred_A = predict(tweets, extra_features_tensorA, model_A, tokenizer_A, device)
off_mask = (y_pred_A == 1).detach().cpu().numpy() 
#print(y_pred_A)


# In[6]:


# === Étape 2 : Task B (sur tweets HOF) ===
tweets_B = [t for i, t in enumerate(tweets) if off_mask[i]]
df_B = df[off_mask]
preds_numB = df_B[['sentiment', 'respect', 'insult', 'humiliate', 'status',
                  'dehumanize', 'attack_defend', 'hatespeech']]
preds_binB = df_B[['target_race', 'target_religion', 'target_origin', 'target_gender',
                'target_sexuality']]

extra_featuresB = np.concatenate([preds_numB, preds_binB], axis=1)
extra_features_tensorB = torch.tensor(extra_featuresB, dtype=torch.float32).to(device)
y_pred_B_partial = predict(tweets_B, extra_features_tensorB, model_B, tokenizer_B, device)
#print(y_pred_B_partial)


# In[7]:


# === Étape 3 : Task C (sur tweets B == HATE) ===
tin_mask = (y_pred_B_partial == 0).detach().cpu().numpy()  # HATE = 0
df_C = df_B[tin_mask]
tweets_C = [t for i, t in enumerate(tweets_B) if tin_mask[i]]
preds_numC = df_C[['sentiment', 'respect', 'insult', 'humiliate', 'status',
                  'dehumanize', 'attack_defend', 'hatespeech']]
preds_binC = df_C[['target_race', 'target_religion', 'target_origin', 'target_gender',
                'target_sexuality']]

extra_featuresC = np.concatenate([preds_numC, preds_binC], axis=1)
extra_features_tensorC = torch.tensor(extra_featuresC, dtype=torch.float32).to(device)
y_pred_C_partial = predict(tweets_C, extra_features_tensorC, model_C, tokenizer_C, device)
#print(y_pred_C_partial)


# In[8]:


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
gold_A = [1 if label == "HOF" else 0 for label in df["label_A"]]
gold_B = [ ["HATE", "OFFN", "PRFN"].index(label) if label in ["HATE", "OFFN", "PRFN"] else -1 for label in df["label_B"] ]
gold_C = [ ["UNT", "TIN"].index(label) if label in ["UNT", "TIN"] else -1 for label in df["label_C"] ]


# In[9]:



# === Convert to encodings ===
df["label_A_enc"] = df["label_A"].map({"NOT": 0, "HOF": 1})
df["label_B_enc"] = df["label_B"].map({"HATE": 0, "OFFN": 1, "PRFN": 2})
df["label_C_enc"] = df["label_C"].map({"UNT": 0, "TIN": 1})

df["pred_A_enc"] = df["pred_A"].map({"NOT": 0, "HOF": 1})
df["pred_B_enc"] = df["pred_B"].map({"HATE": 0, "OFFN": 1, "PRFN": 2})
df["pred_C_enc"] = df["pred_C"].map({"UNT": 0, "TIN": 1})


# === Evaluation Task A ===
print("\n=== Task A ===")
y_true_A = df["label_A_enc"].astype(int)
y_pred_A = df["pred_A_enc"].astype(int)

print("Accuracy:", accuracy_score(y_true_A, y_pred_A))
print("F1-score:", f1_score(y_true_A, y_pred_A, average="macro"))
print(classification_report(y_true_A, y_pred_A, target_names=["NOT", "HOF"]))

# === Task B: ground truth-based inference ===
df_B_gt = df[df["label_A"] == "HOF"]
tweets_B_gt = df_B_gt["text"].tolist()

preds_numB_gt = df_B_gt[['sentiment', 'respect', 'insult', 'humiliate', 'status',
                         'dehumanize', 'attack_defend', 'hatespeech']]
preds_binB_gt = df_B_gt[['target_race', 'target_religion', 'target_origin', 'target_gender',
                         'target_sexuality']]

extra_features_B_gt = np.concatenate([preds_numB_gt, preds_binB_gt], axis=1)
extra_features_tensor_B_gt = torch.tensor(extra_features_B_gt, dtype=torch.float32).to(device)

y_pred_B_gt = predict(tweets_B_gt, extra_features_tensor_B_gt, model_B, tokenizer_B, device)

# === Assign GT-based predictions into new column
df["pred_B_gt"] = "NULL"
df.loc[df["label_A"] == "HOF", "pred_B_gt"] = [ ["HATE", "OFFN", "PRFN"][x] for x in y_pred_B_gt.cpu().numpy() ]
df["pred_B_gt_enc"] = df["pred_B_gt"].map({"HATE": 0, "OFFN": 1, "PRFN": 2})


# === Evaluation Task B ===
def evaluate_task_B(df, mode="gt"):
    print(f"\n=== Task B — Mode: {'Ground Truth' if mode == 'gt' else 'Cascade Prediction'} ===")
    
    if mode == "gt":
        mask = (df["label_A_enc"] == 1) & df["label_B_enc"].notna() & df["pred_B_gt_enc"].notna()
        y_pred = df.loc[mask, "pred_B_gt_enc"].astype(int).tolist()
    else:
        mask = (df["pred_A_enc"] == 1) & df["label_B_enc"].notna() & df["pred_B_enc"].notna()
        y_pred = df.loc[mask, "pred_B_enc"].astype(int).tolist()

    y_true = df.loc[mask, "label_B_enc"].astype(int).tolist()

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("F1-score:", f1_score(y_true, y_pred, average="macro"))
    print(classification_report(y_true, y_pred, target_names=["HATE", "OFFN", "PRFN"]))




# === Task C: ground truth-based inference ===
df_C_gt = df[df["label_A"] == "HOF"]
tweets_C_gt = df_C_gt["text"].tolist()

preds_numC_gt = df_C_gt[['sentiment', 'respect', 'insult', 'humiliate', 'status',
                         'dehumanize', 'attack_defend', 'hatespeech']]
preds_binC_gt = df_C_gt[['target_race', 'target_religion', 'target_origin', 'target_gender',
                         'target_sexuality']]

extra_features_C_gt = np.concatenate([preds_numC_gt, preds_binC_gt], axis=1)
extra_features_tensor_C_gt = torch.tensor(extra_features_C_gt, dtype=torch.float32).to(device)

y_pred_C_gt = predict(tweets_C_gt, extra_features_tensor_C_gt, model_C, tokenizer_C, device)

# === Assign GT-based predictions into new column
df["pred_C_gt"] = "NULL"
df.loc[df["label_A"] == "HOF", "pred_C_gt"] = [ ["UNT", "TIN"][x] for x in y_pred_C_gt.cpu().numpy() ]
df["pred_C_gt_enc"] = df["pred_C_gt"].map({"UNT": 0, "TIN": 1})



# === Evaluation Task C ===
def evaluate_task_C(df, mode="gt"):
    print(f"\n=== Task C — Mode: {'Ground Truth' if mode == 'gt' else 'Cascade Prediction'} ===")

    if mode == "gt":
        mask = (df["label_A_enc"] == 1) & df["label_C_enc"].notna() & df["pred_C_gt_enc"].notna()
        y_pred = df.loc[mask, "pred_C_gt_enc"].astype(int).tolist()
    else:
        mask = (df["pred_A_enc"] == 1) & df["label_C_enc"].notna() & df["pred_C_enc"].notna()
        y_pred = df.loc[mask, "pred_C_enc"].astype(int).tolist()

    y_true = df.loc[mask, "label_C_enc"].astype(int).tolist()

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("F1-score:", f1_score(y_true, y_pred, average="macro"))
    print(classification_report(y_true, y_pred, target_names=["UNT", "TIN"]))


# === Run all evaluations
evaluate_task_B(df, mode="gt")
evaluate_task_B(df, mode="cascade")

evaluate_task_C(df, mode="gt")
evaluate_task_C(df, mode="cascade")


