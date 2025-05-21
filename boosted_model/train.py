# === train.py ===
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from hasoc_model import *

MODEL_NAMES = {
    "A": "roberta-base",
    "B": "GroNLP/hateBERT",
    "C": "GroNLP/hateBERT"
}
NUM_LABELS = {"A": 2, "B": 3, "C": 2}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

freeze = False

# === TASK A ===
print("\n=== TRAINING TASK A ===")
task = "A"
tokenizer_A = AutoTokenizer.from_pretrained(MODEL_NAMES[task], use_fast=True)
dataset_A, labels_A = prepare_dataset(task)
model_A = CombinedModel(MODEL_NAMES[task], NUM_LABELS[task], extra_feature_dim=13).to(device)
class_weights_A = compute_class_weights(labels_A, NUM_LABELS[task], task=task)
print(f"[A] Class weights: {class_weights_A.tolist()}")
model_A.class_weights = class_weights_A
if torch.cuda.device_count() > 1:
    print("[A] Using multiple GPUs")
    model_A = nn.DataParallel(model_A)
train_model(task, model_A, dataset_A, tokenizer_A, freeze=freeze)

# === TASK B ===
print("\n=== TRAINING TASK B ===")
task = "B"
tokenizer_B = AutoTokenizer.from_pretrained(MODEL_NAMES[task], use_fast=True)
dataset_B, labels_B = prepare_dataset(task)
model_B = CombinedModel(MODEL_NAMES[task], NUM_LABELS[task], extra_feature_dim=13).to(device)
class_weights_B = compute_class_weights(labels_B, NUM_LABELS[task], task=task)
print(f"[B] Class weights: {class_weights_B.tolist()}")
model_B.class_weights = class_weights_B
if torch.cuda.device_count() > 1:
    print("[B] Using multiple GPUs")
    model_B = nn.DataParallel(model_B)
train_model(task, model_B, dataset_B, tokenizer_B, freeze=freeze)

# === TASK C ===
print("\n=== TRAINING TASK C ===")
task = "C"
tokenizer_C = AutoTokenizer.from_pretrained(MODEL_NAMES[task], use_fast=True)
dataset_C, labels_C = prepare_dataset(task)
model_C = CombinedModel(MODEL_NAMES[task], NUM_LABELS[task], extra_feature_dim=13).to(device)
class_weights_C = compute_class_weights(labels_C, NUM_LABELS[task], task=task)
print(f"[C] Class weights: {class_weights_C.tolist()}")
model_C.class_weights = class_weights_C
if torch.cuda.device_count() > 1:
    print("[C] Using multiple GPUs")
    model_C = nn.DataParallel(model_C)
train_model(task, model_C, dataset_C, tokenizer_C, freeze=freeze)
