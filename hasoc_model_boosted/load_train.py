# === load_train.py ===
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from hasoc_model_boosted import *
from collections import OrderedDict

MODEL_NAMES = {
    "A": "roberta-base",
    "B": "GroNLP/hateBERT",
    "C": "GroNLP/hateBERT"
}
NUM_LABELS = {"A": 2, "B": 3, "C": 2}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

freeze = False
experiment = "load"

# === TASK A ===
print("\n=== TRAINING TASK A ===")
task = "A"
tokenizer_A = AutoTokenizer.from_pretrained(MODEL_NAMES[task], use_fast=True)
dataset_A, labels_A = prepare_dataset(task)
model_A = CombinedModel(MODEL_NAMES[task], NUM_LABELS[task], extra_feature_dim=13).to(device)

state_dictA = torch.load("../hasoc_model_base/best_model_A_roberta-base.pth", map_location=device, weights_only=True)
new_state_dictA = OrderedDict()
for k, v in state_dictA.items():
    new_key = k
    if k.startswith('roberta.'):
        new_key = k.replace('roberta.', 'text_model.')
    new_state_dictA[new_key] = v
model_A.load_state_dict(new_state_dictA, strict=False)  # strict=False to ignore unexpected keys (e.g. classifier, etc.)


class_weights_A = compute_class_weights(labels_A, NUM_LABELS[task], task=task)
print(f"[A] Class weights: {class_weights_A.tolist()}")
model_A.class_weights = class_weights_A
if torch.cuda.device_count() > 1:
    print("[A] Using multiple GPUs")
    model_A = nn.DataParallel(model_A)
train_model(task, experiment, model_A, dataset_A, tokenizer_A, freeze=freeze)

# === TASK B ===
print("\n=== TRAINING TASK B ===")
task = "B"
tokenizer_B = AutoTokenizer.from_pretrained(MODEL_NAMES[task], use_fast=True)
dataset_B, labels_B = prepare_dataset(task)
model_B = CombinedModel(MODEL_NAMES[task], NUM_LABELS[task], extra_feature_dim=13).to(device)

state_dictB = torch.load("../hasoc_model_base/best_model_B_hateBERT.pth", map_location=device, weights_only=True)
new_state_dictB = OrderedDict()
for k, v in state_dictB.items():
    new_key = k
    if k.startswith('bert.'):
        new_key = k.replace('bert.', 'text_model.')
    new_state_dictB[new_key] = v
model_B.load_state_dict(new_state_dictB, strict=False)

class_weights_B = compute_class_weights(labels_B, NUM_LABELS[task], task=task)
print(f"[B] Class weights: {class_weights_B.tolist()}")
model_B.class_weights = class_weights_B
if torch.cuda.device_count() > 1:
    print("[B] Using multiple GPUs")
    model_B = nn.DataParallel(model_B)
train_model(task, experiment, model_B, dataset_B, tokenizer_B, freeze=freeze)

# === TASK C ===
print("\n=== TRAINING TASK C ===")
task = "C"
tokenizer_C = AutoTokenizer.from_pretrained(MODEL_NAMES[task], use_fast=True)
dataset_C, labels_C = prepare_dataset(task)
model_C = CombinedModel(MODEL_NAMES[task], NUM_LABELS[task], extra_feature_dim=13).to(device)

state_dictC = torch.load("../hasoc_model_base/best_model_C_hateBERT.pth", map_location=device, weights_only=True)
new_state_dictC = OrderedDict()
for k, v in state_dictC.items():
    new_key = k
    if k.startswith('bert.'):
        new_key = k.replace('bert.', 'text_model.')
    new_state_dictC[new_key] = v
model_C.load_state_dict(new_state_dictC, strict=False)

class_weights_C = compute_class_weights(labels_C, NUM_LABELS[task], task=task)
print(f"[C] Class weights: {class_weights_C.tolist()}")
model_C.class_weights = class_weights_C
if torch.cuda.device_count() > 1:
    print("[C] Using multiple GPUs")
    model_C = nn.DataParallel(model_C)
train_model(task, experiment, model_C, dataset_C, tokenizer_C, freeze=freeze)