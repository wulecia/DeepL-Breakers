# === predict.py ===
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from hasoc_model import CombinedModel, encode_labels, compute_metrics, compute_class_weights, prepare_dataset
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
#---------------------
import os
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
import csv

def get_next_results_filename(prefix="results", suffix=".csv", folder="results"):
    os.makedirs(folder, exist_ok=True)
    existing = [f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith(suffix)]
    indexes = [int(f[len(prefix):-len(suffix)]) for f in existing if f[len(prefix):-len(suffix)].isdigit()]
    next_index = max(indexes) + 1 if indexes else 1
    return os.path.join(folder, f"{prefix}{next_index}{suffix}")

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


def load_model(task, experiment, device):
    model_path = f"./results/{experiment}/best_boostedmodel_{task}_{MODEL_NAMES[task].split('/')[-1]}"
    model = CombinedModel(MODEL_NAMES[task], NUM_LABELS[task], extra_feature_dim=13)

    # Load saved weights
    state_dict = torch.load(model_path + ".pth", map_location=device)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    # Recompute class weights
    _, labels = prepare_dataset(task)  # Only need labels
    class_weights = compute_class_weights(labels, NUM_LABELS[task], task=task)
    model.class_weights = class_weights

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



all_metrics = []
def evaluate_predictions(preds, experiment, labels, task, class_weights=None):
    from sklearn.metrics import classification_report, confusion_matrix

    report_dict = classification_report(labels, preds, output_dict=True, zero_division=0)
    accuracy = report_dict["accuracy"]
    f1_score = report_dict["weighted avg"]["f1-score"]
    cm = confusion_matrix(labels, preds)

    print(f"\n=== Evaluation for task {task} ===")
    print(classification_report(labels, preds, digits=4))
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score (weighted): {f1_score:.4f}")
    print("Confusion Matrix:")
    print(cm)

    all_metrics.append({
        "Task": task,
        "Experiment": experiment,
        "Accuracy": accuracy,
        "Weighted F1": f1_score,
        "Class Weights": class_weights.tolist() if class_weights is not None else "None",
        "Confusion Matrix": cm.tolist()
    })


def main():
    experiment = "random_init"

    df = pd.read_csv("../hasoc_model/hasoc_dataset/hasoc_dataset_with_features_test.tsv", sep="\t")
    df = encode_labels(df)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Task A ===
    print("\n===== PREDICTING TASK A =====")
    model_A = load_model("A", experiment, device)
    tokenizer_A = AutoTokenizer.from_pretrained(MODEL_NAMES["A"], use_fast=True)
    preds_A = predict_task("A", model_A, df, tokenizer_A, device)
    df["pred_A"] = preds_A
    evaluate_predictions(preds_A, experiment, df["label_A_enc"].values, "A", model_A.class_weights)


    # === Task B ===
    print("\n===== PREDICTING TASK B =====")
    model_B = load_model("B", experiment, device)
    tokenizer_B = AutoTokenizer.from_pretrained(MODEL_NAMES["B"], use_fast=True)
    df_B = df[df["label_A"] == "HOF"].copy()
    preds_B = predict_task("B", model_B, df_B, tokenizer_B, device)
    evaluate_predictions(preds_B, experiment, df_B["label_B_enc"].values, "B", model_B.class_weights)

    # === Task C ===
    print("\n===== PREDICTING TASK C =====")
    model_C = load_model("C", experiment, device)
    tokenizer_C = AutoTokenizer.from_pretrained(MODEL_NAMES["C"], use_fast=True)
    df_C = df[df["label_A"] == "HOF"].copy()
    preds_C = predict_task("C", model_C, df_C, tokenizer_C, device)
    evaluate_predictions(preds_C, experiment, df_C["label_C_enc"].values, "C", model_C.class_weights)


    # Save all metrics into one file after all tasks are processed
    os.makedirs(f"results/{experiment}", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results/{experiment}/run_{timestamp}.csv"

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Task", "Experiment", "Accuracy", "Weighted F1", "Class Weights", "Confusion Matrix"])
        for row in all_metrics:
            writer.writerow([
                row["Task"],
                row["Experiment"],
                row["Accuracy"],
                row["Weighted F1"],
                row["Class Weights"],
                row["Confusion Matrix"]
            ])
    print(f"\n✅ Metrics saved to {output_file}")



if __name__ == "__main__":
    main()
