# === hasoc_model_boosted.py ===
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

MODEL_NAMES = {
    "A": "roberta-base",
    "B": "GroNLP/hateBERT",
    "C": "GroNLP/hateBERT"
}
NUM_LABELS = {"A": 2, "B": 3, "C": 2}


def encode_labels(df):
    df = df.copy()
    df["label_A_enc"] = df["label_A"].map({"NOT": 0, "HOF": 1})
    df["label_B_enc"] = df["label_B"].map({"HATE": 0, "OFFN": 1, "PRFN": 2})
    df["label_C_enc"] = df["label_C"].map({"UNT": 0, "TIN": 1})
    return df.dropna(subset=["label_A_enc"])


def prepare_dataset(task, split="train"):
    file_path = f"../hasoc_dataset/{split}_extra_features.tsv"
    df = pd.read_csv(file_path, sep="\t")
    df = encode_labels(df)

    if task in ["B", "C"]:
        df = df[df["label_A"] == "HOF"]

    label_col = f"label_{task}_enc"
    df_task = df[["text", label_col] + [
        'sentiment', 'respect', 'insult', 'humiliate', 'status',
        'dehumanize', 'attack_defend', 'hatespeech',
        'target_race', 'target_religion', 'target_origin', 'target_gender',
        'target_sexuality'
    ]].dropna()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAMES[task], use_fast=True)
    encodings = tokenizer(df_task["text"].tolist(), truncation=True, padding=True, max_length=128)

    extra_feats = df_task.iloc[:, 2:].astype(np.float32).values.tolist()
    labels = df_task[label_col].astype(int).tolist()

    dataset = Dataset.from_dict({
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "extra_features": extra_feats,
        "labels": labels
    })

    return dataset.train_test_split(test_size=0.2, seed=42), labels


def compute_class_weights(labels, num_labels, task=None, weight_factors=None):
    class_weights = compute_class_weight(class_weight='balanced', classes=np.arange(num_labels), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    
    if weight_factors:
        for i, factor in enumerate(weight_factors):
            class_weights[i] *= factor

    else : 
        if task == "A":
            class_weights[0] *= 1.7
            class_weights[1] *= 1.2
    
        if task == "B":
            class_weights[0] *= 1.8
            class_weights[1] *= 1.4
            class_weights[2] *= 1.7
    
        if task == "C":
            class_weights[0] *= 1
            class_weights[1] *= 2

    return class_weights


def compute_metrics(eval_pred):
    predictions, labels = eval_pred.predictions, eval_pred.label_ids
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    preds = np.argmax(predictions, axis=1)
    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    return {
        "f1": report["weighted avg"]["f1-score"],
        "accuracy": report["accuracy"]
    }


def log_metrics_to_csv(log_history, task):
    import csv
    from collections import defaultdict
    os.makedirs("visu", exist_ok=True)
    path = f"visu/metrics_{task}.csv"
    epochs = defaultdict(dict)
    for entry in log_history:
        if "epoch" in entry:
            epoch = round(entry["epoch"], 2)
            for key in ["loss", "eval_loss", "eval_f1", "eval_accuracy"]:
                if key in entry:
                    epochs[epoch][key] = entry[key]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "eval_loss", "eval_f1", "eval_accuracy"])
        writer.writeheader()
        for epoch in sorted(epochs.keys()):
            row = {
                "epoch": epoch,
                "train_loss": epochs[epoch].get("loss", 0),
                "eval_loss": epochs[epoch].get("eval_loss", 0),
                "eval_f1": epochs[epoch].get("eval_f1", 0),
                "eval_accuracy": epochs[epoch].get("eval_accuracy", 0)
            }
            writer.writerow(row)


class CombinedModel(nn.Module):
    def __init__(self, model_name, num_labels, extra_feature_dim):
        super().__init__()
        self.model_name = model_name
        self.text_model = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.extra_layer = nn.Linear(extra_feature_dim, 64)
        self.classifier = nn.Sequential(
            nn.Linear(self.text_model.config.hidden_size + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_labels)
        )
        self.class_weights = None

    def freeze_transformer(self):
        for param in self.text_model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask, extra_features, labels=None):
        text_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        extra_output = torch.relu(self.extra_layer(extra_features))
        combined = torch.cat([text_output, extra_output], dim=1)
        logits = self.classifier(combined)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}


class WeightedFocalLossTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.args.device) if class_weights is not None else None
        self.gamma = 2.0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").long()
        extra_features = inputs.pop("extra_features")
        outputs = model(extra_features=extra_features, labels=labels, **inputs)
        logits = outputs["logits"]
        ce_loss = torch.nn.functional.cross_entropy(logits, labels, weight=self.class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return (focal_loss, outputs) if return_outputs else focal_loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        extra_features = inputs.pop("extra_features")
        labels = inputs.get("labels")
        with torch.no_grad():
            outputs = model(extra_features=extra_features, **inputs)
            loss = None
            if labels is not None:
                loss = self.compute_loss(model, {**inputs, "extra_features": extra_features, "labels": labels})
        logits = outputs["logits"]
        return (loss, logits, labels)


def train_model(task, experiment, model_wrapper, dataset, tokenizer, resume=False, freeze=False):
    model = model_wrapper.module if isinstance(model_wrapper, nn.DataParallel) else model_wrapper
    output_dir = f"./results/{experiment}/results_{task}_{model.model_name.split('/')[-1]}"
    logging_dir = f"./results/{experiment}/logs_{task}_{model.model_name.split('/')[-1]}"

    os.makedirs(output_dir, exist_ok=True)

    checkpoint_path = None
    if resume and os.path.isdir(output_dir):
        checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith("checkpoint")]
        if checkpoints:
            checkpoint_path = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]
            print(f"Resuming from checkpoint: {checkpoint_path}")
        else:
            print("No checkpoint found — starting from scratch.")
    else:
        print("Starting training from scratch.")

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        num_train_epochs=12,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        save_strategy="steps" if task == "B" else "epoch",
        save_steps=200 if task == "B" else None,
        eval_strategy="steps" if task == "B" else "epoch",
        eval_steps=200 if task == "B" else None,
        logging_steps=100,
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        label_smoothing_factor=0.05 if task == "A" else 0.1 if task == "B" else 0.0,
        save_total_limit=2,
        report_to="none",
        logging_first_step=True,
        disable_tqdm=False,
        greater_is_better=True,
        seed=42,
        do_train=True,
        remove_unused_columns=False
    )

    if freeze==True:
        model.freeze_transformer()

    trainer = WeightedFocalLossTrainer(
        class_weights=model.class_weights,
        model=model_wrapper,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    trainer.train(resume_from_checkpoint=checkpoint_path if checkpoint_path else None)

    trainer.save_model(f"./results/models/{experiment}/best_boostedmodel_{task}_{model.model_name.split('/')[-1]}")
    torch.save(
        model_wrapper.state_dict(),
        f"./results/models/{experiment}/best_boostedmodel_{task}_{model.model_name.split('/')[-1]}.pth"
    )
    print(f"Task {task} training complete.")
    log_metrics_to_csv(trainer.state.log_history, task)
