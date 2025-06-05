from functions2 import compute_scores
from model_utils import get_model, get_dataloaders 
from datetime import datetime
import torch
import pandas as pd
import os
import sys
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error, r2_score

# command to use :
# python3 test.py best_model.pth

def main(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model().to(device)

    test_loader = get_dataloaders(test_only=True)
    
    # === Load model from .pth file ===
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # === Compute Scores ===
    y_true_bin, y_pred_bin = [], []
    y_true_num, y_pred_num = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets_bin = batch['bin_targets'].to(device)
            targets_num = batch['num_targets'].to(device)

            pred_num, pred_bin = model(input_ids, attention_mask)

            y_true_bin.extend(targets_bin.cpu().numpy())
            y_pred_bin.extend((pred_bin > 0.5).cpu().numpy())
            y_true_num.extend(targets_num.cpu().numpy())
            y_pred_num.extend(pred_num.cpu().numpy())

    # === Compute Metrics ===
    metrics = {
        "R2": r2_score(y_true_num, y_pred_num),
        "MSE": mean_squared_error(y_true_num, y_pred_num),
        "Accuracy": accuracy_score(y_true_bin, y_pred_bin),
        "F1_Macro": f1_score(y_true_bin, y_pred_bin, average="macro"),
        "F1_Micro": f1_score(y_true_bin, y_pred_bin, average="micro"),
    }

    # === Save to CSV ===
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame([metrics])
    df["timestamp"] = timestamp
    df["epochs"] = 10  # still hardcoded
    df["model_path"] = model_path
    df.to_csv(f"results/test_results_{timestamp}.csv", index=False)

    print("Evaluation complete. Metrics saved to results/test_results_{timestamp}.csv")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test.py path_to_model.pth")
    else:
        main(sys.argv[1])
