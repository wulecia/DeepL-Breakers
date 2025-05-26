# === load_grid_train.py ===
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from hasoc_model_boosted import *
from collections import OrderedDict


ALL_WEIGHTS = {
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

LOAD_DICT_KEY = {
    "A": "roberta.",
    "B": "bert.",
    "C": "bert."
}

LOAD_DICT_PATH = {
    "A": "best_no_features_A_roberta-base_full.pth",
    "B": "best_no_features_B_hateBERT_full.pth",
    "C": "best_no_features_C_hateBERT_full.pth"
}

def weight_id(task, weights):
    return f"{task}_{'-'.join([str(w).replace('.', '') for w in weights])}"

def train_all(freeze, experiment):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for task in ["A", "B", "C"]:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAMES[task], use_fast=True)
        dataset, labels = prepare_dataset(task)

        for weights in ALL_WEIGHTS[task]:
            model_id = weight_id(task, weights)
            print(f"\n=== TRAINING {task} with weights {weights} ===")
            model = CombinedModelNoFeatures(MODEL_NAMES[task], NUM_LABELS[task], extra_feature_dim=13).to(device)

            try:
                state_dict = torch.load(f"../hasoc_model_base/{LOAD_DICT_PATH[task]}", map_location=device, weights_only=True)
            except TypeError:
                model_wrapper = torch.load(f"../hasoc_model_base/{LOAD_DICT_PATH[task]}", map_location=device)
                state_dict = model_wrapper.state_dict()
                
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_key = k
                if k.startswith(LOAD_DICT_KEY[task]):
                    new_key = k.replace(LOAD_DICT_KEY[task], 'text_model.')
                new_state_dict[new_key] = v
            model.load_state_dict(new_state_dict, strict=False)

            
            class_weights = compute_class_weights(labels, NUM_LABELS[task], task=task, weight_factors=weights)
            model.class_weights = class_weights

            if torch.cuda.device_count() > 1:
                print(f"[{task}] Using multiple GPUs")
                model = nn.DataParallel(model)

            train_model(
                task=task,
                experiment=experiment,
                model_wrapper=model,
                dataset=dataset,
                tokenizer=tokenizer,
                resume=False,
                freeze=freeze
            )

            path = f"results/models/{experiment}/best_boostedmodel_{task}_{MODEL_NAMES[task].split('/')[-1]}"
            new_path = f"results/models/{experiment}/best_boostedmodel_{model_id}.pth"
            torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), new_path)
            print(f"Model saved as {new_path}")

if __name__ == "__main__":
    freeze = False
    experiment = "load_grid"
    train_all(freeze, experiment)
