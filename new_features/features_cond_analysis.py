import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic
from hasoc_model import encode_labels
from pathlib import Path
from matplotlib.text import Text
from matplotlib import colors as mcolors
import colorsys

def shade_color(base_rgb, factor):
    h, l, s = colorsys.rgb_to_hls(*base_rgb)
    l = max(0, min(1, l * factor))
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (r, g, b)

def plot_mosaic_for_features():
    features = ['target_race', 'target_religion', 'target_origin', 'target_gender', 'target_sexuality']
    tasks = ["A", "B", "C"]
    file_path_train = "../hasoc_dataset/train_extra_features.tsv"
    file_path_test = "../hasoc_dataset/test_extra_features.tsv"

    df_train = pd.read_csv(file_path_train, sep="\t")
    df_test = pd.read_csv(file_path_test, sep="\t")
    df_combined = pd.concat([df_train, df_test], ignore_index=True)

    save_dir = Path("./features_probs")
    save_dir.mkdir(parents=True, exist_ok=True)

    df = encode_labels(df_combined) 

    for feature in features:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Mosaic plots for feature: {feature}", fontsize=25, fontweight='bold')

        for idx, task in enumerate(tasks):
            df_task = df.copy()
            if task in ["B", "C"]:
                df_task = df_task[df_task["label_A"] == "HOF"]

            label_col = f"label_{task}_enc"
            df_task = df_task[[label_col, feature]].dropna()
            df_task[feature] = df_task[feature].astype(int)

            label_names = {
                "A": {0: "NOT", 1: "HOF"},
                "B": {0: "HATE", 1: "OFFN", 2: "PRFN"},
                "C": {0: "UNT", 1: "TIN"},
            }

            df_task["label_str"] = df_task[label_col].map(label_names[task])
            df_task["feature_str"] = df_task[feature].astype(str)

            cross_tab = df_task.groupby(["feature_str", "label_str"]).size().reset_index(name="count")

            total_feature_counts = df_task.groupby("feature_str").size().to_dict()
            probs_dict = {}
            for _, row in cross_tab.iterrows():
                fstr = row["feature_str"]
                lstr = row["label_str"]
                count = row["count"]
                feat_val = int(fstr)
                total_feat_count = total_feature_counts.get(fstr, 1)
                prob = count / total_feat_count if total_feat_count > 0 else 0
                probs_dict[(fstr, lstr)] = {
                    "count": count,
                    "label": f"{prob:.2f}"
                }
            mosaic_data = {k: v["count"] for k, v in probs_dict.items()}
            label_texts = {k: v["label"] for k, v in probs_dict.items()}
            
            def custom_labelizer(key):
                return label_texts.get(key, '')

            ax = axs[idx]
            label_map = {"A": "HOF/NOT", "B": "HATE/OFFN/PRFN", "C": "TIN/UNT"}

            def custom_props(key):
                base_red = (244 / 255, 204 / 255, 204 / 255)   
                base_green = (0.78, 0.90, 0.78)               

                feature_str, label_str = key
                count = mosaic_data[key]

                group_counts = {
                    fs: max(v for (fs_key, _), v in mosaic_data.items() if fs_key == fs)
                    for fs in set(fs_key for (fs_key, _) in mosaic_data)
                }

                max_count = group_counts[feature_str]
                fraction = 0.7 + 0.3 * (count / max_count)  

                if "0" in feature_str:
                    color = shade_color(base_red, fraction)
                else:
                    color = shade_color(base_green, fraction)

                return {"color": color}

            mosaic(mosaic_data, ax=ax, labelizer=custom_labelizer, properties=custom_props,
            axes_label=True, gap=0.01)
            ax.set_title(f"Task {task}", fontsize=19, fontweight="bold")  
            ax.set_ylabel("")
            ax.set_xlabel("")
            ax.tick_params(labelsize=19)
            for text in ax.texts:
                text.set_fontsize(19)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = save_dir / f"Prob_analysis_{feature}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

plot_mosaic_for_features()
