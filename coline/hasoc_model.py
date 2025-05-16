import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

# === 1. Constantes ===
MODEL_NAMES = {
    "A": "roberta-base",
    "B": "GroNLP/hateBERT",
    "C": "GroNLP/hateBERT"
}
NUM_LABELS = {"A": 2, "B": 3, "C": 2}

'''
# === 2. Charger les données HASOC ===
df = pd.read_csv("../hasoc_model/hasoc_dataset/train.tsv", sep="\t")
df.columns = ["id", "text", "label_A", "label_B", "label_C"]
df = df[["text", "label_A", "label_B", "label_C"]] 
'''
def encode_labels(df):
    df = df.copy()
    df["label_A_enc"] = df["label_A"].map({"NOT": 0, "HOF": 1})
    df["label_B_enc"] = df["label_B"].map({"HATE": 0, "OFFN": 1, "PRFN": 2})
    df["label_C_enc"] = df["label_C"].map({"UNT": 0, "TIN": 1}) 
    return df.dropna(subset=["label_A_enc"])

'''
df = encode_labels(df)
'''


# === 3. Préparer les datasets ===
def prepare_dataset(df, task):
    if task == "A":
        df_task = df.dropna(subset=["label_A_enc"])
        labels = df_task["label_A_enc"].tolist()

    elif task == "B":
        df_task = df[df["label_A"] == "HOF"].dropna(subset=["label_B_enc"])
        labels = df_task["label_B_enc"].tolist()

    elif task == "C":
        df_task = df[(df["label_A"] == "HOF") & (df["label_C"].isin(["UNT", "TIN"]))].dropna(subset=["label_C_enc"])
        labels = df_task["label_C_enc"].tolist()

    if df_task.empty:
        raise ValueError(f"Aucune donnée trouvée pour la tâche {task}. Vérifie les filtres.")

    texts = df_task["text"].tolist()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAMES[task], use_fast=True)
    encodings = tokenizer(texts, truncation=True, padding=True)

    dataset = Dataset.from_dict({
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": torch.tensor(labels, dtype=torch.long).tolist()
    })

    return dataset.train_test_split(test_size=0.2, seed=42), labels


# === 4. Pondération des classes ===
def compute_class_weights(labels, num_labels, task=None):
    class_weights = compute_class_weight(class_weight='balanced', classes=np.arange(num_labels), y=labels)

    if task == "C":
        class_weights[1] *= 1.0
        class_weights[0] *= 2.0

    return torch.tensor(class_weights, dtype=torch.float)



# === 5. Métriques ===
def compute_metrics(eval_pred):
    if isinstance(eval_pred, tuple):
        predictions, loss, labels = eval_pred
    else:
        predictions, labels = eval_pred.predictions, eval_pred.label_ids

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    preds = np.argmax(predictions, axis=1)
    report = classification_report(labels, preds, output_dict=True, zero_division=0)

    return {
    "f1": report["weighted avg"]["f1-score"],
    "accuracy": report["accuracy"]
}

    
# === 6. Trainer personnalisé pour tâche B ===
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
        """
        Custom prediction step to handle extra_features in evaluation.
        """
        extra_features = inputs.pop("extra_features")
        labels = inputs.get("labels")
        
        with torch.no_grad():
            outputs = model(extra_features=extra_features, **inputs)
            loss = None
            if labels is not None:
                loss = self.compute_loss(model, {**inputs, "extra_features": extra_features, "labels": labels})
        
        logits = outputs["logits"]
        return (loss, logits, labels)



# === 7. Classe ModelHASOC ===
class ModelHASOC(nn.Module):
    def __init__(self, task, model_name=None, num_labels=None, class_weights=None):
        super(ModelHASOC, self).__init__()
        self.task = task
        self.model_name = model_name or MODEL_NAMES[task]
        self.num_labels = num_labels or NUM_LABELS[task]
        self.class_weights = class_weights

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        )

    def forward(self, **inputs):
        return self.model(**inputs)

        

# === 8. Training ===
def train_model(task, model_wrapper, dataset, tokenizer, resume=True):
    model = model_wrapper.module if isinstance(model_wrapper, nn.DataParallel) else model_wrapper
    output_dir = f"./results_{task}_{model.model_name.split('/')[-1]}"
    logging_dir = f"./logs_{task}_{model.model_name.split('/')[-1]}"

    checkpoint_path = None
    if resume and os.path.isdir(output_dir):
        checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith("checkpoint")]
        if checkpoints:
            checkpoint_path = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]
            print(f"🔁 Resuming from checkpoint: {checkpoint_path}")
        else:
            print("⚠️ No checkpoint found — starting from scratch.")
    else:
        print("🆕 Starting training from scratch.")

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        num_train_epochs=4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        save_strategy="steps" if task == "B" else "epoch",
        save_steps=500 if task == "B" else None,
        eval_strategy="epoch",
        eval_steps=500 if task == "B" else None,
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
    
    (model_wrapper.module if isinstance(model_wrapper, nn.DataParallel) else model_wrapper).freeze_transformer()

    if task in ["B", "A"]:
        trainer = WeightedFocalLossTrainer(
            class_weights=model.class_weights,
            model=model_wrapper,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
    else:
        trainer = Trainer(
            model=model_wrapper,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )
    

    trainer.train(resume_from_checkpoint=checkpoint_path if checkpoint_path else None)

    model_name_str = (model_wrapper.module if isinstance(model_wrapper, nn.DataParallel) else model_wrapper).model_name

    trainer.save_model(f"./best_colinemodel_{task}_{model_name_str.split('/')[-1]}")
    torch.save(
        model_wrapper.state_dict(),
        f"./best_colinemodel_{task}_{model_name_str.split('/')[-1]}.pth"
    )
    print(f"Task {task} training complete.")
    
'''
# === 9. Main Loop ===
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for task in ["A", "B", "C"]:
        print(f"\n🚀 Training Task {task}")
        dataset, labels = prepare_dataset(df, task)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAMES[task], use_fast=True)

        class_weights = compute_class_weights(labels, NUM_LABELS[task], task=task) if task in ["A", "B", "C"] else None
        model_wrapper = ModelHASOC(task=task, class_weights=class_weights).to(device)

        train_model(task, model_wrapper, dataset, tokenizer, resume=True)
'''
