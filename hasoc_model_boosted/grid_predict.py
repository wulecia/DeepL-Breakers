# === grid_predict.py ===
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from hasoc_model import CombinedModel, encode_labels
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

WEIGHTS = {
    "A": [
        (1.0, 1.0), (1.5, 1.0), (1.0, 1.5), (1.3, 1.3),
        (1.6, 1.4), (1.4, 1.6), (1.8, 1.2), (1.2, 1.8),
        (2.0, 1.0), (1.0, 2.0), (2.5, 1.0), (1.0, 2.5)
    ],
    "B": [
        (1.0, 1.0, 1.0),
        (1.5, 1.0, 1.0), (1.0, 1.5, 1.0), (1.0, 1.0, 1.5),
        (1.8, 1.4, 1.8), (2.0, 1.5, 1.8), (2.0, 3.0, 2.0),
        (1.5, 4.0, 1.5), (2.0, 4.0, 2.0), (2.5, 5.0, 2.5)
    ],
    "C": [
        (1.0, 1.0), (1.5, 1.0), (1.0, 1.5), (2.0, 1.0), (1.0, 2.0),
        (2.5, 1.0), (1.0, 2.5), (3.0, 1.0), (1.0, 3.0),
        (4.0, 1.0), (1.0, 4.0)
    ]
}

LABELS = {
    "A": {0: "NOT", 1: "HOF"},
    "B": {0: "HATE", 1: "OFFN", 2: "PRFN"},
    "C": {0: "UNT", 1: "TIN"}
}

def weight_id(task, weights):
    return f"{task}_{'-'.join([str(w).replace('.', '') for w in weights])}"

def load_model(task, experiment, model_id, device):
    model_path = f"results/models/{experiment}/best_boostedmodel_{model_id}.pth"
    model = CombinedModel(MODEL_NAMES[task], NUM_LABELS[task], extra_feature_dim=13)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    return model.to(device).eval()

def get_encoded_inputs(df, tokenizer):
    encodings = tokenizer(df["text"].tolist(), truncation=True, padding=True, max_length=128, return_tensors="pt")
    extra_feats = torch.tensor(df[EXTRA_FEATURES].values, dtype=torch.float32)
    return encodings, extra_feats

def predict_and_eval(task, experiment, model, df, tokenizer, label_col, prefix):
    encodings, extra_feats = get_encoded_inputs(df, tokenizer)
    device = next(model.parameters()).device
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)
    extra_feats = extra_feats.to(device)
    labels = df[label_col].values.astype(int)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, extra_features=extra_feats)
        preds = torch.argmax(outputs["logits"], dim=1).cpu().numpy()

    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(labels, preds)

    # Save results
    os.makedirs(f"results/{experiment}/grid_metrics", exist_ok=True)
    pd.DataFrame(cm).to_csv(f"results/{experiment}/grid_metrics/confmat_{prefix}.csv", index=False)
    with open(f"results/{experiment}/grid_metrics/report_{prefix}.csv", "w") as f:
        f.write(pd.DataFrame(report).to_csv())

    # Optional: Save confusion matrix image
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {prefix}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"results/{experiment}/grid_metrics/confmat_{prefix}.png")
    plt.close()
    return report["weighted avg"]["f1-score"], report["macro avg"]["f1-score"], report["accuracy"]

def run_all_predictions(experiment):
    df = pd.read_csv("../hasoc_model/hasoc_dataset/hasoc_dataset_with_features_test.tsv", sep="\t")
    df = encode_labels(df)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary = []

    for task, weight_list in WEIGHTS.items():
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAMES[task], use_fast=True)

        for weights in weight_list:
            model_id = weight_id(task, weights)
            print(f"\n=== EVALUATING {model_id} ===")
            model = load_model(task, experiment, model_id, device)

            if task == "A":
                df_task = df.copy()
            else:
                df_task = df[df["label_A"] == "HOF"].copy() if "label_A" in df else df.copy()

            if task == "B":
                label_col = "label_B_enc"
            elif task == "C":
                label_col = "label_C_enc"
            else:
                label_col = "label_A_enc"

            df_task = df_task.dropna(subset=[label_col])
            f1_weighted, f1_macro, acc = predict_and_eval(task, experiment, model, df_task, tokenizer, label_col, model_id)

            summary.append({
                "task": task,
                "experiment": experiment,
                "weights": weights,
                "model_id": model_id,
                "f1_weighted": f1_weighted,
                "f1_macro": f1_macro,
                "accuracy": acc
            })

    pd.DataFrame(summary).to_csv(f"results/{experiment}/grid_metrics/summary_f1_accuracy.csv", index=False)

if __name__ == "__main__":
    experiment = "load_grid"
    run_all_predictions(experiment)
