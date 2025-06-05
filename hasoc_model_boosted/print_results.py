import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

experiment = "load_grid"

summary = pd.read_csv(f"./results/{experiment}/grid_metrics/summary_f1_accuracy.csv") 
confmat_dir = Path(f"./results/{experiment}/grid_metrics")
save_dir = Path(f"./results/{experiment}/grid_metrics_final")
save_dir.mkdir(parents=True, exist_ok=True)

top_models = summary.groupby("task").apply(lambda g: g.nlargest(1, "f1_macro")).reset_index(drop=True) #1 fot Top 1

for task in top_models["task"].unique():
    row = top_models[top_models["task"] == task].iloc[0]
    model_id = row["model_id"]
    confmat_path = confmat_dir / f"confmat_{model_id}.csv"
    
    if confmat_path.exists():
        confmat = pd.read_csv(confmat_path, header=None, skiprows=1).values
        plt.figure(figsize=(6, 5))
        ax = sns.heatmap(confmat, annot=True, fmt="d", cmap="Greys", cbar=False, annot_kws={"size": 40})
        
        if task == "A":
            coefs = [0.8, 1.3]
        elif task == "B":
            coefs = [0.65, 1.675, 1.15]
        elif task == "C":
            coefs = [5.1, 0.56]
        task_name, weights = model_id.split("_")
        weight_parts = [float(w) for w in weights.split("-")]
        weighted_parts = [w * c for w, c in zip(weight_parts, coefs)]

        formatted_weights = "-".join(f"{float(w)/10:.1f}" for w in weighted_parts)
        
        title_line1 = f"Task {task_name} (weights: {formatted_weights})"

        title_line2 = (
            f"Weighted F1: {row['f1_weighted']:.3f} | "
            f"Macro F1: {row['f1_macro']:.3f} | "
            f"Acc: {row['accuracy']:.3f}"
        )
        plt.suptitle(title_line1, fontsize=24, fontweight='bold', x=0.55)
        ax.set_title(title_line2, fontsize=18)

        ax.set_xlabel("Predicted", fontsize=21)
        ax.set_ylabel("True", fontsize=21)

        if task == "A":
            ax.set_xticklabels(["NOT", "HOF"], fontsize=21)
            ax.set_yticklabels(["NOT", "HOF"], fontsize=21)
        elif task == "B":
            ax.set_xticklabels(["HATE", "OFFN", "PRFN"], fontsize=21)
            ax.set_yticklabels(["HATE", "OFFN", "PRFN"], fontsize=21)
        elif task == "C":
            ax.set_xticklabels(["UNT", "TIN"], fontsize=21)
            ax.set_yticklabels(["UNT", "TIN"], fontsize=21)

        plt.tight_layout()
        save_path = save_dir / f"Top_1_Model_Task_{task}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved: {save_path}")
    else:
        print(f"Missing confusion matrix for model: {model_id}")
