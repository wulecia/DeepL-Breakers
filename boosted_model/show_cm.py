import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

experiment = "load_freeze_grid"
summary = pd.read_csv(f"./results/{experiment}/grid_metrics/summary_f1_accuracy.csv") 
confmat_dir = Path(f"./results/{experiment}/grid_metrics")

top_models = summary.groupby("task").apply(lambda g: g.nlargest(8, "f1_weighted")).reset_index(drop=True)

for task in top_models["task"].unique():
    task_models = top_models[top_models["task"] == task]
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, (_, row) in enumerate(task_models.iterrows()):
        model_id = row["model_id"]
        confmat_path = confmat_dir / f"confmat_{model_id}.csv"
        
        if confmat_path.exists():
            confmat = pd.read_csv(confmat_path, header=None).values
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
    plt.show()
