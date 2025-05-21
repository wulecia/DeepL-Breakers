# model_utils.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from sklearn.model_selection import train_test_split

# Define your model
class BERTMultiTaskModel(nn.Module):
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
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, bin_outputs)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        return self.regressor(pooled_output), self.classifier(pooled_output)


# Dataset class
class HateSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, texts, targets_num, targets_bin, tokenizer, max_len=128):
        self.texts = list(texts)
        self.targets_num = torch.tensor(targets_num.values, dtype=torch.float)
        self.targets_bin = torch.tensor(targets_bin.values, dtype=torch.float)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'num_targets': self.targets_num[idx],
            'bin_targets': self.targets_bin[idx]
        }


# Loader functions
def get_model():
    return BERTMultiTaskModel()

def get_dataloaders(batch_size=16, test_only=False):
    df = pd.read_parquet("data/measuring-hate-speech.parquet")
    df = df.dropna(subset=['text'])

    numerical_cols = ['sentiment', 'respect', 'insult', 'humiliate', 'status',
                      'dehumanize', 'attack_defend', 'hatespeech']
    binary_cols = ['target_race', 'target_religion', 'target_origin', 'target_gender',
                   'target_sexuality']
    df[binary_cols] = df[binary_cols].astype(int)

    train_texts, temp_texts, train_y_num, temp_y_num, train_y_bin, temp_y_bin = train_test_split(
        df['text'], df[numerical_cols], df[binary_cols], test_size=0.3, random_state=42
    )
    val_texts, test_texts, val_y_num, test_y_num, val_y_bin, test_y_bin = train_test_split(
        temp_texts, temp_y_num, temp_y_bin, test_size=1/3, random_state=42
    )

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    if test_only:
        test_dataset = HateSpeechDataset(test_texts, test_y_num, test_y_bin, tokenizer)
        return DataLoader(test_dataset, batch_size=batch_size)
    
    train_dataset = HateSpeechDataset(train_texts, train_y_num, train_y_bin, tokenizer)
    val_dataset = HateSpeechDataset(val_texts, val_y_num, val_y_bin, tokenizer)
    test_dataset = HateSpeechDataset(test_texts, test_y_num, test_y_bin, tokenizer)

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size),
        DataLoader(test_dataset, batch_size=batch_size)
    )


def get_loss_and_optimizer(model, lr=2e-5):
    loss_fn_num = nn.MSELoss()
    loss_fn_bin = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True)
    return loss_fn_num, loss_fn_bin, optimizer, scheduler
