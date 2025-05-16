#!/usr/bin/env python
# coding: utf-8

# In[6]:


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


# In[7]:


df_clara = pd.read_csv("hasoc_model/hasoc_dataset/train.tsv", sep="\t")
df_clara.columns = ["id", "text", "label_A", "label_B", "label_C"]
df_clara = df_clara[["text", "label_A", "label_B", "label_C"]] 
df_clara = encode_labels(df_clara)
#df_clara = df_clara[0:100]
print(df_clara.head())


# In[8]:


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


# In[9]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_paola = Paola().to(device)
model_paola.load_state_dict(torch.load("paola/model2_loaded.pth", map_location=device, weights_only=True))

print("model2_loaded.pth loaded and ready to use!")

tokenizer_paola = AutoTokenizer.from_pretrained("distilbert-base-uncased")


# In[10]:


encodings_paola = tokenizer_paola(df_clara["text"].tolist(), truncation=True, padding=True, max_length=128, return_tensors="pt")

model_paola.eval()
input_ids_paola = encodings_paola['input_ids'].to(device)
attention_mask_paola = encodings_paola['attention_mask'].to(device)

with torch.no_grad():
    preds_num, preds_bin = model_paola(input_ids=input_ids_paola, attention_mask=attention_mask_paola)

preds_num = preds_num.cpu().numpy()
preds_bin = preds_bin.cpu().numpy()
preds_bin = (preds_bin > 0.5).astype(int)

for idx in range (3):
    print(f"Sentence: {df_clara["text"].tolist()[idx]}")
    print(f"Numerical predictions: {preds_num[idx]}")
    print(f"Binary predictions: {preds_bin[idx]}")
    print()


# In[11]:


new_feature_names = ['sentiment', 'respect', 'insult', 'humiliate', 'status',
                  'dehumanize', 'attack_defend', 'hatespeech',
                     'target_race', 'target_religion', 'target_origin', 'target_gender',
                'target_sexuality']

combined_preds = np.concatenate([preds_num, preds_bin], axis=1)
preds_df = pd.DataFrame(combined_preds, columns=new_feature_names)
df_clara = pd.concat([df_clara.reset_index(drop=True), preds_df], axis=1)

df_clara.to_csv("../hasoc_model/hasoc_dataset/hasoc_dataset_with_features_train.tsv", sep='\t', index=False)


# In[ ]:





# In[ ]:




