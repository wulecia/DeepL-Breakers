#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
import pandas as pd
import numpy as np
from hasoc_model import *
#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')

#torch.cuda.empty_cache()
#torch.cuda.ipc_collect()


# In[2]:


df_clara = pd.read_csv("hasoc_model/hasoc_dataset/hasoc_dataset_with_features_train.tsv", sep="\t")
df_clara.columns = ["text", "label_A", "label_B", "label_C", 'label_A_enc',  'label_B_enc',  'label_C_enc', 'sentiment', 'respect', 'insult', 'humiliate', 'status',
                  'dehumanize', 'attack_defend', 'hatespeech',
                     'target_race', 'target_religion', 'target_origin', 'target_gender',
                'target_sexuality'] 


# In[3]:


df_claraA = df_clara.dropna(subset=["label_A_enc"])
labelsA = df_claraA["label_A_enc"].tolist()
preds_numA = df_claraA[['sentiment', 'respect', 'insult', 'humiliate', 'status',
                  'dehumanize', 'attack_defend', 'hatespeech']]
preds_binA = df_claraA[['target_race', 'target_religion', 'target_origin', 'target_gender',
                'target_sexuality']]
df_claraA = df_claraA["text"].tolist()

df_claraB = df_clara[df_clara["label_A"] == "HOF"].dropna(subset=["label_B_enc"])
labelsB = df_claraB["label_B_enc"].tolist()
preds_numB = df_claraB[['sentiment', 'respect', 'insult', 'humiliate', 'status',
                  'dehumanize', 'attack_defend', 'hatespeech']]
preds_binB = df_claraB[['target_race', 'target_religion', 'target_origin', 'target_gender',
                'target_sexuality']]
df_claraB = df_claraB["text"].tolist()

df_claraC = df_clara[(df_clara["label_A"] == "HOF") & (df_clara["label_C"].isin(["UNT", "TIN"]))].dropna(subset=["label_C_enc"])
labelsC = df_claraC["label_C_enc"].tolist()
preds_numC = df_claraC[['sentiment', 'respect', 'insult', 'humiliate', 'status',
                  'dehumanize', 'attack_defend', 'hatespeech']]
preds_binC = df_claraC[['target_race', 'target_religion', 'target_origin', 'target_gender',
                'target_sexuality']]
df_claraC = df_claraC["text"].tolist()


# In[4]:


'''
n = 40

df_claraA = df_claraA[0:n]
labelsA = labelsA[0:n]

df_claraB = df_claraB[0:n]
labelsB = labelsB[0:n]

df_claraC = df_claraC[0:n]
labelsC = labelsC[0:n]
'''


# In[5]:


MODEL_NAMES = {
    "A": "roberta-base",
    "B": "GroNLP/hateBERT",
    "C": "GroNLP/hateBERT"
}
NUM_LABELS = {"A": 2, "B": 3, "C": 2}


# In[6]:


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


# In[7]:


# ------- TASK A ------


# In[8]:

"""
task = "A"


# In[9]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer_claraA = AutoTokenizer.from_pretrained(MODEL_NAMES[task], use_fast=True)
encodings_claraA = tokenizer_claraA(df_claraA, truncation=True, padding=True, return_tensors="pt")
input_ids_claraA = encodings_claraA['input_ids'].to(device)
attention_mask_claraA = encodings_claraA['attention_mask'].to(device)


# In[10]:


class_weightsA = compute_class_weights(labelsA, NUM_LABELS[task], task=task)

model_colineA = Coline(task="A", model_name="roberta-base", class_weights=class_weightsA)
state_dictA = torch.load("coline/best_model_A_roberta-base.pth", map_location=device, weights_only=True)
model_colineA.transformer.load_state_dict(
    {k.replace("roberta.", ""): v for k, v in state_dictA.items() if k.startswith("roberta.")},
    strict=False
)
if torch.cuda.device_count() > 1:
    print(f"[A] Using {torch.cuda.device_count()} GPUs with DataParallel")
    model_colineA = nn.DataParallel(model_colineA)
model_colineA = model_colineA.to(device)

# Strip "roberta." from the beginning of keys that belong to the transformer
transformer_state_dictA = {
    k.replace("roberta.", ""): v
    for k, v in state_dictA.items()
    if k.startswith("roberta.")
}



# In[11]:


extra_featuresA = np.concatenate([preds_numA, preds_binA], axis=1)
extra_features_tensorA = torch.tensor(extra_featuresA, dtype=torch.float32)


# In[12]:


datasetA = Dataset.from_dict({
        "input_ids": input_ids_claraA,
        "attention_mask": attention_mask_claraA,
        "labels": torch.tensor(labelsA, dtype=torch.long).tolist(),
        "extra_features": extra_features_tensorA.tolist()
    })

datasetA = datasetA.train_test_split(test_size=0.2, seed=42)

train_model(task, model_colineA, datasetA, tokenizer_claraA, resume=True)


# In[13]:


# ------- TASK B ------


# In[14]:


task = "B"


# In[15]:


tokenizer_claraB = AutoTokenizer.from_pretrained(MODEL_NAMES[task], use_fast=True)
encodings_claraB = tokenizer_claraB(df_claraB, truncation=True, padding=True, return_tensors="pt")
input_ids_claraB = encodings_claraB['input_ids'].to(device)
attention_mask_claraB = encodings_claraB['attention_mask'].to(device)


# In[16]:


class_weightsB = compute_class_weights(labelsB, NUM_LABELS[task], task=task)

model_colineB = Coline(task="B", model_name="GroNLP/hateBERT", class_weights=class_weightsB)
state_dictB = torch.load("coline/best_model_B_hateBERT.pth", map_location=device, weights_only=True)

# Adjust the layer names if needed
model_colineB.transformer.load_state_dict(
    {k.replace("bert.", ""): v for k, v in state_dictB.items()},
    strict=False
)

# Wrap with DataParallel if using multiple GPUs
if torch.cuda.device_count() > 1:
    print(f"[B] Using {torch.cuda.device_count()} GPUs with DataParallel")
    model_colineB = nn.DataParallel(model_colineB)

model_colineB = model_colineB.to(device)


# In[17]:


extra_featuresB = np.concatenate([preds_numB, preds_binB], axis=1)
extra_features_tensorB = torch.tensor(extra_featuresB, dtype=torch.float32)


# In[18]:


datasetB = Dataset.from_dict({
        "input_ids": input_ids_claraB,
        "attention_mask": attention_mask_claraB,
        "labels": torch.tensor(labelsB, dtype=torch.long).tolist(),
        "extra_features": extra_features_tensorB.tolist()
    })

datasetB = datasetB.train_test_split(test_size=0.2, seed=42)

train_model(task, model_colineB, datasetB, tokenizer_claraB, resume=True)


# In[19]:


# ------- TASK C ------


# In[20]:


task = "C"


# In[21]:


tokenizer_claraC = AutoTokenizer.from_pretrained(MODEL_NAMES[task], use_fast=True)
encodings_claraC = tokenizer_claraC(df_claraC, truncation=True, padding=True, return_tensors="pt")

input_ids_claraC = encodings_claraC['input_ids'].to(device)
attention_mask_claraC = encodings_claraC['attention_mask'].to(device)


# In[22]:


class_weightsC = compute_class_weights(labelsC, NUM_LABELS[task], task=task)

model_colineC = Coline(task="C", model_name="GroNLP/hateBERT", class_weights=class_weightsC)
state_dictC = torch.load("coline/best_model_C_hateBERT.pth", map_location=device, weights_only=True)

model_colineC.transformer.load_state_dict(
    {k.replace("bert.", ""): v for k, v in state_dictC.items()},
    strict=False
)

# Wrap with DataParallel if using multiple GPUs
if torch.cuda.device_count() > 1:
    print(f"[C] Using {torch.cuda.device_count()} GPUs with DataParallel")
    model_colineC = nn.DataParallel(model_colineC)

model_colineC = model_colineC.to(device)

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


# In[ ]:


"""


# In[ ]:

# ======== TASK A ========
print("\n=== TRAINING TASK A ===")
task = "A"
tokenizer_A = AutoTokenizer.from_pretrained(MODEL_NAMES[task], use_fast=True)
enc_A = tokenizer_A(df_claraA, truncation=True, padding=True, return_tensors="pt")
input_ids_A = enc_A['input_ids'].to(device)
attention_mask_A = enc_A['attention_mask'].to(device)
extra_A = np.concatenate([preds_numA, preds_binA], axis=1)
extra_tensor_A = torch.tensor(extra_A, dtype=torch.float32)
class_weights_A = compute_class_weights(labelsA, NUM_LABELS[task], task=task)
print(f"[{task}] class weights:", class_weights_A.cpu().numpy())  # ou class_weights_B etc.

model_A = Coline(task=task, model_name=MODEL_NAMES[task], class_weights=class_weights_A).to(device)
if torch.cuda.device_count() > 1:
    print("[A] Using multiple GPUs")
    model_A = nn.DataParallel(model_A)

dataset_A = Dataset.from_dict({
    "input_ids": input_ids_A,
    "attention_mask": attention_mask_A,
    "labels": torch.tensor(labelsA, dtype=torch.long).tolist(),
    "extra_features": extra_tensor_A.tolist()
}).train_test_split(test_size=0.2, seed=42)

train_model(task, model_A, dataset_A, tokenizer_A, resume=True)

# ======== TASK B ========
print("\n=== TRAINING TASK B ===")
task = "B"
tokenizer_B = AutoTokenizer.from_pretrained(MODEL_NAMES[task], use_fast=True)
enc_B = tokenizer_B(df_claraB, truncation=True, padding=True, return_tensors="pt")
input_ids_B = enc_B['input_ids'].to(device)
attention_mask_B = enc_B['attention_mask'].to(device)
extra_B = np.concatenate([preds_numB, preds_binB], axis=1)
extra_tensor_B = torch.tensor(extra_B, dtype=torch.float32)
class_weights_B = compute_class_weights(labelsB, NUM_LABELS[task], task=task)
print(f"[{task}] class weights:", class_weights_B.cpu().numpy())  # ou class_weights_B etc.

model_B = Coline(task=task, model_name=MODEL_NAMES[task], class_weights=class_weights_B).to(device)
if torch.cuda.device_count() > 1:
    print("[B] Using multiple GPUs")
    model_B = nn.DataParallel(model_B)

dataset_B = Dataset.from_dict({
    "input_ids": input_ids_B,
    "attention_mask": attention_mask_B,
    "labels": torch.tensor(labelsB, dtype=torch.long).tolist(),
    "extra_features": extra_tensor_B.tolist()
}).train_test_split(test_size=0.2, seed=42)

train_model(task, model_B, dataset_B, tokenizer_B, resume=True)

# ======== TASK C ========
print("\n=== TRAINING TASK C ===")
task = "C"
tokenizer_C = AutoTokenizer.from_pretrained(MODEL_NAMES[task], use_fast=True)
enc_C = tokenizer_C(df_claraC, truncation=True, padding=True, return_tensors="pt")
input_ids_C = enc_C['input_ids'].to(device)
attention_mask_C = enc_C['attention_mask'].to(device)
extra_C = np.concatenate([preds_numC, preds_binC], axis=1)
extra_tensor_C = torch.tensor(extra_C, dtype=torch.float32)
class_weights_C = compute_class_weights(labelsC, NUM_LABELS[task], task=task)
print(f"[{task}] class weights:", class_weights_C.cpu().numpy())  # ou class_weights_B etc.

model_C = Coline(task=task, model_name=MODEL_NAMES[task], class_weights=class_weights_C).to(device)
if torch.cuda.device_count() > 1:
    print("[C] Using multiple GPUs")
    model_C = nn.DataParallel(model_C)

dataset_C = Dataset.from_dict({
    "input_ids": input_ids_C,
    "attention_mask": attention_mask_C,
    "labels": torch.tensor(labelsC, dtype=torch.long).tolist(),
    "extra_features": extra_tensor_C.tolist()
}).train_test_split(test_size=0.2, seed=42)

train_model(task, model_C, dataset_C, tokenizer_C, resume=True)


