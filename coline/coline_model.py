#!/usr/bin/env python
# coding: utf-8

# In[40]:


import torch
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from hasoc_model import *
#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')

#torch.cuda.empty_cache()
#torch.cuda.ipc_collect()
print(torch.cuda.is_available())  # False signifie aucun GPU dispo
print(torch.version.cuda)         # Version du runtime CUDA attendu (si dispo)
device = torch.device("cpu")


# In[41]:


df_clara = pd.read_csv("../hasoc_model/hasoc_dataset/train.tsv", sep="\t")
df_clara.columns = ["id", "text", "label_A", "label_B", "label_C"]
df_clara = df_clara[["text", "label_A", "label_B", "label_C"]] 
df_clara = encode_labels(df_clara)


# In[42]:


df_claraA = df_clara.dropna(subset=["label_A_enc"])
labelsA = df_claraA["label_A_enc"].tolist()
df_claraA = df_claraA["text"].tolist()

df_claraB = df_clara[df_clara["label_A"] == "HOF"].dropna(subset=["label_B_enc"])
labelsB = df_claraB["label_B_enc"].tolist()
df_claraB = df_claraB["text"].tolist()

df_claraC = df_clara[(df_clara["label_A"] == "HOF") & (df_clara["label_C"].isin(["UNT", "TIN"]))].dropna(subset=["label_C_enc"])
labelsC = df_claraC["label_C_enc"].tolist()
df_claraC = df_claraC["text"].tolist()


# In[43]:


#len(df_claraA)
'''
n = 100

df_claraA = df_claraA[0:n]
labelsA = labelsA[0:n]

df_claraB = df_claraB[0:n]
labelsB = labelsB[0:n]

df_claraC = df_claraC[0:n]
labelsC = labelsC[0:n]
'''


# In[44]:


MODEL_NAMES = {
    "A": "roberta-base",
    "B": "GroNLP/hateBERT",
    "C": "GroNLP/hateBERT"
}
NUM_LABELS = {"A": 2, "B": 3, "C": 2}


# In[45]:


class Paola(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", num_outputs=8, bin_outputs=5):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_outputs)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, bin_outputs),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        return self.regressor(pooled), self.classifier(pooled)


# In[46]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_paola = Paola().to(device)
model_paola.load_state_dict(torch.load("../paola/model2_loaded.pth", map_location=device, weights_only=True))

print("model2_loaded.pth loaded and ready to use!")

tokenizer_paola = AutoTokenizer.from_pretrained("distilbert-base-uncased")


# In[47]:


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


# In[48]:


# ------- TASK A ------


# In[49]:


task = "A"


# In[50]:


encodings_paolaA = tokenizer_paola(df_claraA, truncation=True, padding=True, max_length=128, return_tensors="pt")

model_paola.eval()
input_ids_paolaA = encodings_paolaA['input_ids'].to(device)
attention_mask_paolaA = encodings_paolaA['attention_mask'].to(device)

with torch.no_grad():
    preds_numA, preds_binA = model_paola(input_ids=input_ids_paolaA, attention_mask=attention_mask_paolaA)

preds_numA = preds_numA.cpu().numpy()
preds_binA = preds_binA.cpu().numpy()
preds_binA = (preds_binA > 0.5).astype(int)

for idx, sentence in enumerate(df_claraA[0:5]):
    print(f"Sentence: {sentence}")
    print(f"Numerical predictions: {preds_numA[idx]}")
    print(f"Binary predictions: {preds_binA[idx]}")
    print()


# numerical_cols = ['sentiment', 'respect', 'insult', 'humiliate', 'status',
#                   'dehumanize', 'attack_defend', 'hatespeech']
#                   
# binary_cols = ['target_race', 'target_religion', 'target_origin', 'target_gender',
#                'target_sexuality']

# In[51]:


tokenizer_claraA = AutoTokenizer.from_pretrained(MODEL_NAMES[task], use_fast=True)
encodings_claraA = tokenizer_claraA(df_claraA, truncation=True, padding=True, return_tensors="pt")
input_ids_claraA = encodings_claraA['input_ids'].to(device)
attention_mask_claraA = encodings_claraA['attention_mask'].to(device)


# In[52]:


class_weightsA = compute_class_weights(labelsA, NUM_LABELS[task], task=task)

model_colineA = Coline(task="A", model_name="roberta-base", class_weights=class_weightsA).to(device)
state_dictA = torch.load("best_model_A_roberta-base.pth", map_location=device, weights_only=True)

# Strip "roberta." from the beginning of keys that belong to the transformer
transformer_state_dictA = {
    k.replace("roberta.", ""): v
    for k, v in state_dictA.items()
    if k.startswith("roberta.")
}

# Load into the RobertaModel (your transformer's structure)
model_colineA.transformer.load_state_dict(transformer_state_dictA, strict=False)


# In[53]:


extra_featuresA = np.concatenate([preds_numA, preds_binA], axis=1)
extra_features_tensorA = torch.tensor(extra_featuresA, dtype=torch.float32)


# In[54]:


datasetA = Dataset.from_dict({
        "input_ids": input_ids_claraA,
        "attention_mask": attention_mask_claraA,
        "labels": torch.tensor(labelsA, dtype=torch.long).tolist(),
        "extra_features": extra_features_tensorA.tolist()
    })

datasetA = datasetA.train_test_split(test_size=0.2, seed=42)

train_model(task, model_colineA, datasetA, tokenizer_claraA, resume=True)
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# In[55]:


# ------- TASK B ------


# In[56]:


task = "B"


# In[57]:


encodings_paolaB = tokenizer_paola(df_claraB, truncation=True, padding=True, max_length=128, return_tensors="pt")

model_paola.eval()
input_ids_paolaB = encodings_paolaB['input_ids'].to(device)
attention_mask_paolaB = encodings_paolaB['attention_mask'].to(device)

with torch.no_grad():
    preds_numB, preds_binB = model_paola(input_ids=input_ids_paolaB, attention_mask=attention_mask_paolaB)

preds_numB = preds_numB.cpu().numpy()
preds_binB = preds_binB.cpu().numpy()
preds_binB = (preds_binB > 0.5).astype(int)

for idx, sentence in enumerate(df_claraB[0:5]):
    print(f"Sentence: {sentence}")
    print(f"Numerical predictions: {preds_numB[idx]}")
    print(f"Binary predictions: {preds_binB[idx]}")
    print()


# In[58]:


tokenizer_claraB = AutoTokenizer.from_pretrained(MODEL_NAMES[task], use_fast=True)
encodings_claraB = tokenizer_claraB(df_claraB, truncation=True, padding=True, return_tensors="pt")
input_ids_claraB = encodings_claraB['input_ids'].to(device)
attention_mask_claraB = encodings_claraB['attention_mask'].to(device)


# In[61]:


class_weightsB = compute_class_weights(labelsB, NUM_LABELS[task], task=task)

model_colineB = Coline(task="B", model_name="GroNLP/hateBERT", class_weights=class_weightsB).to(device)
state_dictB = torch.load("best_model_B_hateBERT.pth", map_location=device, weights_only=True)

# Adjust the layer names if needed, e.g., by stripping out certain prefixes
model_colineB.transformer.load_state_dict({k.replace("bert.", ""): v for k, v in state_dictB.items()}, strict=False)


# In[22]:


extra_featuresB = np.concatenate([preds_numB, preds_binB], axis=1)
extra_features_tensorB = torch.tensor(extra_featuresB, dtype=torch.float32)


# In[23]:


datasetB = Dataset.from_dict({
        "input_ids": input_ids_claraB,
        "attention_mask": attention_mask_claraB,
        "labels": torch.tensor(labelsB, dtype=torch.long).tolist(),
        "extra_features": extra_features_tensorB.tolist()
    })

datasetB = datasetB.train_test_split(test_size=0.2, seed=42)

train_model(task, model_colineB, datasetB, tokenizer_claraB, resume=True)
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# In[24]:


# ------- TASK C ------


# In[25]:


task = "C"


# In[26]:


encodings_paolaC = tokenizer_paola(df_claraB, truncation=True, padding=True, max_length=128, return_tensors="pt")

model_paola.eval()
input_ids_paolaC = encodings_paolaC['input_ids'].to(device)
attention_mask_paolaC = encodings_paolaC['attention_mask'].to(device)

with torch.no_grad():
    preds_numC, preds_binC = model_paola(input_ids=input_ids_paolaC, attention_mask=attention_mask_paolaC)

preds_numC = preds_numC.cpu().numpy()
preds_binC = preds_binC.cpu().numpy()
preds_binC = (preds_binC > 0.5).astype(int)

for idx, sentence in enumerate(df_claraC[0:5]):
    print(f"Sentence: {sentence}")
    print(f"Numerical predictions: {preds_numC[idx]}")
    print(f"Binary predictions: {preds_binC[idx]}")
    print()


# In[27]:


tokenizer_claraC = AutoTokenizer.from_pretrained(MODEL_NAMES[task], use_fast=True)
encodings_claraC = tokenizer_claraC(df_claraC, truncation=True, padding=True, return_tensors="pt")

input_ids_claraC = encodings_claraC['input_ids'].to(device)
attention_mask_claraC = encodings_claraC['attention_mask'].to(device)


# In[28]:


class_weightsC = compute_class_weights(labelsC, NUM_LABELS[task], task=task)

model_colineC = Coline(task="C", model_name="GroNLP/hateBERT", class_weights=class_weightsC).to(device)
state_dictC = torch.load("best_model_C_hateBERT.pth", map_location=device, weights_only=True)

model_colineC.transformer.load_state_dict({k.replace("bert.", ""): v for k, v in state_dictC.items()}, strict=False)


# In[29]:


extra_featuresC = np.concatenate([preds_numC, preds_binC], axis=1)
extra_features_tensorC = torch.tensor(extra_featuresC, dtype=torch.float32)


# In[30]:


datasetC = Dataset.from_dict({
        "input_ids": input_ids_claraC,
        "attention_mask": attention_mask_claraC,
        "labels": torch.tensor(labelsC, dtype=torch.long).tolist(),
        "extra_features": extra_features_tensorC.tolist()
    })

datasetC = datasetC.train_test_split(test_size=0.2, seed=42)

train_model(task, model_colineC, datasetC, tokenizer_claraC, resume=True)
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# In[ ]:





# In[ ]:




