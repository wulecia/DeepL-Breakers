#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
import pandas as pd
import numpy as np
from hasoc_model import encode_labels
#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')

#torch.cuda.empty_cache()
#torch.cuda.ipc_collect()
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# In[2]:


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


# In[3]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_paola = Paola()

# Load weights BEFORE wrapping in DataParallel
state_dict = torch.load("../paola/model2_loaded.pth", map_location=device, weights_only=True)
model_paola.load_state_dict(state_dict)

# Then wrap in DataParallel if multiple GPUs
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
    model_paola = nn.DataParallel(model_paola)

model_paola = model_paola.to(device)
print("model2_loaded.pth loaded and ready to use!")

tokenizer_paola = AutoTokenizer.from_pretrained("distilbert-base-uncased")


# In[4]:


new_feature_names = ['sentiment', 'respect', 'insult', 'humiliate', 'status',
                  'dehumanize', 'attack_defend', 'hatespeech',
                     'target_race', 'target_religion', 'target_origin', 'target_gender',
                'target_sexuality']


# In[5]:


# -----TRAIN-----


# In[6]:


df_clara_train = pd.read_csv("../hasoc_model/hasoc_dataset/train.tsv", sep="\t")
df_clara_train.columns = ["id", "text", "label_A", "label_B", "label_C"]
df_clara_train = df_clara_train[["text", "label_A", "label_B", "label_C"]] 
df_clara_train = encode_labels(df_clara_train)
#df_clara_train = df_clara_train[0:200]
print(df_clara_train.head())


# In[7]:


encodings_paola_train = tokenizer_paola(df_clara_train["text"].tolist(), truncation=True, padding=True, max_length=128, return_tensors="pt")

model_paola.eval()
input_ids_paola_train = encodings_paola_train['input_ids'].to(device)
attention_mask_paola_train = encodings_paola_train['attention_mask'].to(device)

with torch.no_grad():
    preds_num_train, preds_bin_train = model_paola(input_ids=input_ids_paola_train, attention_mask=attention_mask_paola_train)

preds_num_train = preds_num_train.cpu().numpy()
preds_bin_train = preds_bin_train.cpu().numpy()
preds_bin_train = (preds_bin_train > 0.5).astype(int)

for idx in range (3):
    print(f"Sentence: {df_clara_train["text"].tolist()[idx]}")
    print(f"Numerical predictions: {preds_num_train[idx]}")
    print(f"Binary predictions: {preds_bin_train[idx]}")
    print()


# In[8]:


combined_preds_train = np.concatenate([preds_num_train, preds_bin_train], axis=1)
preds_df_train = pd.DataFrame(combined_preds_train, columns=new_feature_names)
df_clara_train = pd.concat([df_clara_train.reset_index(drop=True), preds_df_train], axis=1)
print(df_clara_train.head())
df_clara_train.to_csv("../hasoc_model/hasoc_dataset/hasoc_dataset_with_features_train.tsv", sep='\t', index=False)


# In[9]:


# -----TEST-----


# In[13]:


df_clara_test = pd.read_csv("../hasoc_model/hasoc_dataset/test.tsv", sep="\t")
df_clara_test.columns = ["id", "text", "label_A", "label_B", "label_C"]
df_clara_test = df_clara_test[["text", "label_A", "label_B", "label_C"]] 
df_clara_test = encode_labels(df_clara_test)
#df_clara_test = df_clara_test[0:200]
print(df_clara_test.head())


# In[14]:


encodings_paola_test = tokenizer_paola(df_clara_test["text"].tolist(), truncation=True, padding=True, max_length=128, return_tensors="pt")

model_paola.eval()
input_ids_paola_test = encodings_paola_test['input_ids'].to(device)
attention_mask_paola_test = encodings_paola_test['attention_mask'].to(device)

with torch.no_grad():
    preds_num_test, preds_bin_test = model_paola(input_ids=input_ids_paola_test, attention_mask=attention_mask_paola_test)

preds_num_test = preds_num_test.cpu().numpy()
preds_bin_test = preds_bin_test.cpu().numpy()
preds_bin_test = (preds_bin_test > 0.5).astype(int)

for idx in range (3):
    print(f"Sentence: {df_clara_test["text"].tolist()[idx]}")
    print(f"Numerical predictions: {preds_num_test[idx]}")
    print(f"Binary predictions: {preds_bin_test[idx]}")
    print()


# In[15]:


combined_preds_test = np.concatenate([preds_num_test, preds_bin_test], axis=1)
preds_df_test = pd.DataFrame(combined_preds_test, columns=new_feature_names)
df_clara_test = pd.concat([df_clara_test.reset_index(drop=True), preds_df_test], axis=1)

df_clara_test.to_csv("../hasoc_model/hasoc_dataset/hasoc_dataset_with_features_test.tsv", sep='\t', index=False)


# In[ ]:





# In[ ]:




