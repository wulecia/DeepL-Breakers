# === predict.py ===
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from hasoc_model import CombinedModel, encode_labels, compute_metrics
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

MODEL_NAMES = {
    "A": "roberta-base",
    "B": "GroNLP/hateBERT",
    "C": "GroNLP/hateBERT"
}
NUM_LABELS = {"A": 2, "B": 3, "C": 2}

EXTRA_FEATURES = [
    'sentiment', 'respect', 'insult', 'humiliate', 'status',
    'dehumanize', 'attack_defend', 'hatespeech',
    'target_race', 'target_religion', 'target_origin', 'target_gender',
    'target_sexuality'
]

LABEL_MAPS = {
    "A": {0: "NOT", 1: "HOF"},
    "B": {0: "HATE", 1: "OFFN", 2: "PRFN"},
    "C": {0: "UNT", 1: "TIN"}
}

def load_model(task, device):
    model_path = f"./results/best_boostedmodel_{task}_{MODEL_NAMES[task].split('/')[-1]}"
    model = CombinedModel(MODEL_NAMES[task], NUM_LABELS[task], extra_feature_dim=13)
    state_dict = torch.load(model_path + ".pth", map_location=device)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    return model.to(device).eval()

def get_encoded_inputs(df, tokenizer):
    encodings = tokenizer(df["text"].tolist(), truncation=True, padding=True, max_length=128, return_tensors="pt")
    extra_feats = torch.tensor(df[EXTRA_FEATURES].values, dtype=torch.float32)
    return encodings, extra_feats

def predict_task(task, model, df, tokenizer, device):
    encodings, extra_feats = get_encoded_inputs(df, tokenizer)
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)
    extra_feats = extra_feats.to(device)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            extra_features=extra_feats
        )
        preds = torch.argmax(outputs["logits"], dim=1).cpu().numpy()
    return preds

def evaluate_predictions(preds, labels, task):
    print(f"\n=== Evaluation for task {task} ===")
    print(classification_report(labels, preds, zero_division=0, digits=4))
    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"F1-score (weighted): {report['weighted avg']['f1-score']:.4f}")

    # Print confusion matrix
    cm = confusion_matrix(labels, preds)
    print("Confusion Matrix:")
    print(cm)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix for task {task}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

def main():
    df = pd.read_csv("../hasoc_model/hasoc_dataset/hasoc_dataset_with_features_test.tsv", sep="\t")
    df = encode_labels(df)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Task A ===
    print("\n===== PREDICTING TASK A =====")
    model_A = load_model("A", device)
    tokenizer_A = AutoTokenizer.from_pretrained(MODEL_NAMES["A"], use_fast=True)
    preds_A = predict_task("A", model_A, df, tokenizer_A, device)
    df["pred_A"] = preds_A
    evaluate_predictions(preds_A, df["label_A_enc"].values, "A")

    # === Task B ===
    print("\n===== PREDICTING TASK B =====")
    model_B = load_model("B", device)
    tokenizer_B = AutoTokenizer.from_pretrained(MODEL_NAMES["B"], use_fast=True)
    df_B_oracle = df[df["label_A"] == "HOF"].copy()
    if len(df_B_oracle) > 0:
        preds_B_oracle = predict_task("B", model_B, df_B_oracle, tokenizer_B, device)
        evaluate_predictions(preds_B_oracle, df_B_oracle["label_B_enc"].values, "B - Oracle")
    df_B_pred = df[df["pred_A"] == 1].copy()
    df_B_pred = df_B_pred.dropna(subset=["label_B_enc"])
    if len(df_B_pred) > 0:
        preds_B_cascade = predict_task("B", model_B, df_B_pred, tokenizer_B, device)
        evaluate_predictions(preds_B_cascade, df_B_pred["label_B_enc"].values.astype(int), "B - Cascade")

    # === Task C ===
    print("\n===== PREDICTING TASK C =====")
    model_C = load_model("C", device)
    tokenizer_C = AutoTokenizer.from_pretrained(MODEL_NAMES["C"], use_fast=True)
    df_C_oracle = df[df["label_A"] == "HOF"].copy()
    if len(df_C_oracle) > 0:
        preds_C_oracle = predict_task("C", model_C, df_C_oracle, tokenizer_C, device)
        evaluate_predictions(preds_C_oracle, df_C_oracle["label_C_enc"].values, "C - Oracle")
    df_C_pred = df[df["pred_A"] == 1].copy()
    df_C_pred = df_C_pred.dropna(subset=["label_C_enc"])
    if len(df_C_pred) > 0:
        preds_C_cascade = predict_task("C", model_C, df_C_pred, tokenizer_C, device)
        evaluate_predictions(preds_C_cascade, df_C_pred["label_C_enc"].values.astype(int), "C - Cascade")

if __name__ == "__main__":
    main()
