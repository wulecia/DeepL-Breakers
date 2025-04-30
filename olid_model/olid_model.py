import torch.nn as nn
from transformers import AutoModel

class ModelOlid(nn.Module):
    def __init__(self, task, model_name=None, num_labels=None, class_weights=None):
        super(ModelOlid, self).__init__()
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


def train_model(task, model_wrapper, dataset, labels, tokenizer, resume=True):
    output_dir = f"./results_{task}_{model_wrapper.model_name.split('/')[-1]}"
    logging_dir = f"./logs_{task}_{model_wrapper.model_name.split('/')[-1]}"

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
        eval_strategy="steps" if task == "B" else "epoch",
        eval_steps=500 if task == "B" else None,
        logging_steps=100,
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        label_smoothing_factor=0.1 if task == "B" else 0.0,
        save_total_limit=2,
        report_to="none",
        logging_first_step=True,
        disable_tqdm=False,
        greater_is_better=True,
        seed=42
    )

    if task == "B":
        trainer = WeightedFocalLossTrainer(
            class_weights=model_wrapper.class_weights,
            model=model_wrapper.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
    else:
        trainer = Trainer(
            model=model_wrapper.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

    trainer.train(resume_from_checkpoint=checkpoint_path if checkpoint_path else None)
    trainer.save_model(f"./best_model_{task}_{model_wrapper.model_name.split('/')[-1]}")
    torch.save(
        model_wrapper.model.state_dict(),
        f"./best_model_{task}_{model_wrapper.model_name.split('/')[-1]}.pth"
    )
    print(f"✅ Task {task} training complete.")


# MAIN LOOP : TO RUN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for task in ["A", "B", "C"]:
    print(f"\n🚀 Training Task {task}")
    dataset, labels = prepare_dataset(df, task)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAMES[task], use_fast=True)

    class_weights = compute_class_weights(labels, NUM_LABELS[task], task=task) if task == "B" else None
    model_wrapper = ModelOlid(task=task, class_weights=class_weights).to(device)

    train_model(task, model_wrapper, dataset, labels, tokenizer, resume=False)
