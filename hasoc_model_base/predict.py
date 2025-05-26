import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

MODEL_NAMES = {
    "A": "roberta-base",
    "B": "GroNLP/hateBERT",
    "C": "GroNLP/hateBERT"
}
LABEL_MAPS = {
    "A": {0: "NOT", 1: "HOF"},
    "B": {0: "HATE", 1: "OFFN", 2: "PRFN"},
    "C": {0: "UNT", 1: "TIN"}
}
NUM_LABELS = {"A": 2, "B": 3, "C": 2}


def load_model(task, device):
    model_path = f"./results/models/best_no_features_{task}_{MODEL_NAMES[task].split('/')[-1]}_full.pt"
    model = torch.load(model_path, map_location=device)
    return model.eval().to(device)

def tokenize_batch(df, tokenizer):
    return tokenizer(df["text"].tolist(), truncation=True, padding=True, max_length=128, return_tensors="pt")


def predict_batch(task, model, df, tokenizer, device):
    inputs = tokenize_batch(df, tokenizer)
    inputs = {k: v.to(device) for k, v in inputs.items() if k in ['input_ids', 'attention_mask']}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs["logits"]
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    return preds


def evaluate_predictions(preds, labels, task_name, label_map):
    print(f"\n=== Evaluation for task {task_name} ===")
    print(classification_report(labels, preds, target_names=label_map.values(), digits=4, zero_division=0))
    print(f"Accuracy: {accuracy_score(labels, preds):.4f}")
    print(f"F1-score (weighted): {f1_score(labels, preds, average='weighted'):.4f}")
    print(f"F1-score (macro): {f1_score(labels, preds, average='macro'):.4f}")  # ✅ this line

    cm = confusion_matrix(labels, preds)
    print("Confusion Matrix:")
    print(cm)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_map.values(), yticklabels=label_map.values())
    plt.title(f"Confusion Matrix for task {task_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv("../hasoc_dataset/test.tsv", sep="\t", names=["id", "text", "label_A", "label_B", "label_C"])
    df = df.dropna(subset=["text", "label_A"])

    df["label_A_enc"] = df["label_A"].map({"NOT": 0, "HOF": 1})
    df["label_B_enc"] = df["label_B"].map({"HATE": 0, "OFFN": 1, "PRFN": 2})
    df["label_C_enc"] = df["label_C"].map({"UNT": 0, "TIN": 1})

    # === TASK A ===
    print("\n===== TASK A =====")
    df_A = df.dropna(subset=["text", "label_A", "label_A_enc"]).copy()
    tokenizer_A = AutoTokenizer.from_pretrained(MODEL_NAMES["A"], use_fast=True)
    model_A = load_model("A", device)
    preds_A = predict_batch("A", model_A, df_A, tokenizer_A, device)
    assert len(preds_A) == len(df_A), f"preds: {len(preds_A)}, labels: {len(df_A)}"
    df_A["pred_A"] = preds_A
    evaluate_predictions(preds_A, df_A["label_A_enc"].astype(int).values, "A", LABEL_MAPS["A"])
    df.loc[df_A.index, "pred_A"] = df_A["pred_A"]

    # === TASK B ===
    print("\n===== TASK B =====")
    df_B_oracle = df[df["label_A"] == "HOF"].dropna(subset=["label_B_enc"]).copy()
    df_B_pred = df[df["pred_A"] == 1].dropna(subset=["label_B_enc"]).copy()
    model_B = load_model("B", device)
    tokenizer_B = AutoTokenizer.from_pretrained(MODEL_NAMES["B"], use_fast=True)

    if not df_B_oracle.empty:
        preds_B_oracle = predict_batch("B", model_B, df_B_oracle, tokenizer_B, device)
        evaluate_predictions(preds_B_oracle, df_B_oracle["label_B_enc"].astype(int).values, "B - Oracle", LABEL_MAPS["B"])

    if not df_B_pred.empty:
        preds_B_cascade = predict_batch("B", model_B, df_B_pred, tokenizer_B, device)
        evaluate_predictions(preds_B_cascade, df_B_pred["label_B_enc"].astype(int).values, "B - Cascade", LABEL_MAPS["B"])
    df.loc[df_B_pred.index, "pred_B"] = preds_B_cascade

    # === TASK C ===
    print("\n===== TASK C =====")
    df_C_oracle = df[df["label_A"] == "HOF"].dropna(subset=["label_C_enc"]).copy()
    df_C_pred = df[(df["pred_A"] == 1) & (df["pred_B"] == 0)].dropna(subset=["label_C_enc"]).copy()
    model_C = load_model("C", device)
    tokenizer_C = AutoTokenizer.from_pretrained(MODEL_NAMES["C"], use_fast=True)

    if not df_C_oracle.empty:
        preds_C_oracle = predict_batch("C", model_C, df_C_oracle, tokenizer_C, device)
        evaluate_predictions(preds_C_oracle, df_C_oracle["label_C_enc"].astype(int).values, "C - Oracle", LABEL_MAPS["C"])

    if not df_C_pred.empty:
        preds_C_cascade = predict_batch("C", model_C, df_C_pred, tokenizer_C, device)
        evaluate_predictions(preds_C_cascade, df_C_pred["label_C_enc"].astype(int).values, "C - Cascade", LABEL_MAPS["C"])


if __name__ == "__main__":
    main()
