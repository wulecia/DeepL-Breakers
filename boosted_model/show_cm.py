import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

experiment = "load_freeze_grid"

summary = pd.read_csv(f"./results/{experiment}/grid_metrics/summary_f1_accuracy.csv") 
confmat_dir = Path(f"./results/{experiment}/grid_metrics")
save_dir = Path(f"./results/{experiment}/grid_metrics_final")
save_dir.mkdir(parents=True, exist_ok=True)

'''
top_models = summary.groupby("task").apply(lambda g: g.nlargest(8, "f1_weighted")).reset_index(drop=True)

for task in top_models["task"].unique():
    task_models = top_models[top_models["task"] == task]
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, (_, row) in enumerate(task_models.iterrows()):
        model_id = row["model_id"]
        confmat_path = confmat_dir / f"confmat_{model_id}.csv"
        
        ax = axes[idx]
        if confmat_path.exists():
            confmat = pd.read_csv(confmat_path, header=None, skiprows=1).values
            sns.heatmap(confmat, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
            
            title_line1 = model_id
            title_line2 = (
                f"Weighted F1: {row['f1_weighted']:.3f} | "
                f"Macro F1: {row['f1_macro']:.3f} | "
                f"Acc: {row['accuracy']:.3f}"
            )
            ax.set_title(f"{title_line1}\n{title_line2}", fontsize=10, pad=12)
        else:
            ax.set_title(f"{model_id}\nMissing File", fontsize=10)
            ax.axis("off")
        
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    
    fig.suptitle(f"Top 8 Models for Task {task}", fontsize=18, y=1.05)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = save_dir / f"Top_8_Models_Task_{task}.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {save_path}")
'''
# Only get top 1 model per task
top_models = summary.groupby("task").apply(lambda g: g.nlargest(1, "f1_weighted")).reset_index(drop=True)

for task in top_models["task"].unique():
    row = top_models[top_models["task"] == task].iloc[0]
    model_id = row["model_id"]
    confmat_path = confmat_dir / f"confmat_{model_id}.csv"
    
    if confmat_path.exists():
        confmat = pd.read_csv(confmat_path, header=None, skiprows=1).values
        plt.figure(figsize=(6, 5))
        sns.heatmap(confmat, annot=True, fmt="d", cmap="Greys", cbar=False, annot_kws={"size": 40})
        
        task_name, weights = model_id.split("_")
        weight_parts = weights.split("-")
        formatted_weights = "-".join(f"{int(w)/10:.1f}" for w in weight_parts)
        title_line1 = f"Task {task_name} (weights: {formatted_weights})"


        #title_line1 = model_id
        title_line2 = (
            f"Weighted F1: {row['f1_weighted']:.3f} | "
            f"Macro F1: {row['f1_macro']:.3f} | "
            f"Acc: {row['accuracy']:.3f}"
        )
        plt.title(f"{title_line1}\n{title_line2}", fontsize=16, pad=12)
        plt.xlabel("Predicted", fontsize=16)
        plt.ylabel("True", fontsize=16)
        if "task" = "A":
            plt.set_xticklabels(["NOT", "HOF"], fontsize=14)
        elif "task" = "B":
            plt.set_yticklabels(["HATE", "OFFN", "PRFN"], fontsize=14)
        elif "task" = "C":
            plt.set_yticklabels(["UNT", "TIN"], fontsize=14)

        plt.tight_layout()
        save_path = save_dir / f"Top_1_Model_Task_{task}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved: {save_path}")
    else:
        print(f"Missing confusion matrix for model: {model_id}")