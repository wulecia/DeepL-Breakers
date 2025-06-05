# Hate Speech Detection & Classification using BERTMultiTaskModel, RoBERTa and HateBERT

## Description
This project explores hate speech and offensive language detection using transformer-based models. It proceeds in two main stages:

### Feature Extraction Model:
1. Train a Multitask DistilBERT on the Berkeley dataset: We first train a BERTMultiTaskModel model on a labeled [dataset from Berkeley](#ref1) containing 13 distinct categories (8 numerical: *sentiment*, *respect*, *insult*, *humiliate*, *status*, *dehumanize*, *attack_defend*, *hatespeech*; 5 binary: *target_race*, *target_religion*, *target_origin*, *target_gender*, *target_sexuality*).
2. Infer 13 features for each HASOC tweet: After achieving satisfying performance, we apply this model to a second dataset — [HASOC 2019](#ref2) — to annotate tweets with these 13 inferred features. 

### Hate Speech Classification Models:
3. Train RoBERTa and HateBERT models on the Hasoc dataset: We use the Hasoc dataset to train a RoBERTa model for Subtask A, and HateBERT models for Subtasks B and C (V1 models). To address class imbalances, we apply subtask-specific class weights, using the WeightedFocalLossTrainer to emphasize harder-to-classify examples and improve performance on underrepresented classes.

4. Train several models on the Hasoc dataset with 13 features to boost performance: We coded 3 models (V2 models) with same architecture, integrating the 13 inferred features from the first stage into the classification models, with different class weighting strategies. The V2 models are obtained by fine-tuning each V1 model with 3 strategies:  
- V2.1: Using pretrained RoBERTa and HateBERT transformer parts, with no freezing.  
- V2.2: Loading fine-tuned parameters (weights and biases) from the best V1 models to transformer layers, with no freezing.  
- V2.3: Loading fine-tuned parameters (weights and biases) from the best V1 models to transformer layers and freezing them.

## Subtasks
The HASOC 2019 dataset contains tweets in English, German, and Hindi. We focus on English tweets across three subtasks:
- **Sub-task A: Offensive Language Identification**  
  Classify tweets as either:

HOF – Hate and Offensive: Contains hate speech, aggression, or profanity.

NOT – Non Hate-Offensive: Contains acceptable content without any offensive language.  

- **Sub-task B: Fine-Grained Offensive Type Classification**  
  Only tweets labeled as HOF in sub-task A are used. To disambiguate the type of offense expressed in HOF tweets, each HOF tweet is further categorized into:

HATE – Hate speech targeting groups based on race, gender, religion, etc.

OFFN – Offensive language targeting an individual or group, including insults or threats.

PRFN – Profane language without specific targeting (e.g., general swearing or cursing).  

- **Sub-task C: Offense Target Identification**  
  Only tweets labeled HOF in sub-task A are included. The goal is to determine the target of the offensive language:

TIN – Targeted Insult: Direct insult/threat against individuals, groups, or other entities.

UNT – Untargeted: Profane or aggressive language without a specific target.

## Repository Structure

```plaintext
.
├── berkeley_model                  # Model for feature generation with the Berkeley dataset
│   ├── data
│   │   └── measuring-hate-speech.parquet  # Parquet file with Berkeley data
│   ├── results                     # Plots and test results
│   │   ├── loss_plot_*.png         # Loss plots over training
│   │   └── test_results_*.csv      # Test set evaluation results
│   ├── best_Berkeley_model.pth     # Best model weights
│   ├── functions2.py               # Supporting functions
│   ├── model_utils.py              # Utility functions for the model
│   ├── test.py                     # Script to test the model
│   ├── train.py                    # Script to train the model
│   ├── try.py                      # Experimentation script to see output for example comments
│   └── visualize_data.ipynb        # Notebook to visualize data
│
├── hasoc_dataset                   # Dataset for HASOC task
│   ├── .ipynb_checkpoints/         # Jupyter auto-saves
│   ├── test-checkpoint.tsv         # Checkpoint test data
│   ├── train-checkpoint.tsv        # Checkpoint train data
│   ├── test_extra_features.tsv     # Extra features for test data
│   ├── test.tsv                    # Test data
│   ├── train_extra_features.tsv    # Extra features for train data
│   └── train.tsv                   # Train data
│
├── hasoc_model_base                # Baseline model for HASOC
│   ├── hasoc_model_base.py         # Model architecture
│   ├── predict.py                  # Prediction script
│   └── train.py                    # Training script
│
├── hasoc_model_boosted             # Boosted model for HASOC
│   ├── results
│   │   ├── grid_metrics/           # Grid search results
│   │   │   ├── run_*.csv
│   ├── grid_predict.py             # Grid search prediction
│   ├── grid_train.py               # Grid search training
│   ├── hasoc_model_boosted.py      # Boosted model architecture
│   ├── load_grid_train.py          # Load training data for grid search
│   ├── load_train.py               # Load standard training data
│   ├── predict.py                  # Prediction script
│   ├── print_results.ipynb         # Notebook for result visualization
│   ├── show_cm.py                  # Show confusion matrices
│   └── train.py                    # Training script
│
├── new_features                    # Additional features
│   ├── add_extra_features.ipynb    # Notebook to add extra features
│   ├── add_extra_features.py       # Script to add extra features
│   └── features_cond_analysis.py   # Conditional analysis of tweets given the features
│
├── LICENSE_berkeley                # License for the Berkeley dataset
├── LICENSE_hasoc                   # License for the Hasoc dataset
├── README.md                       # This file
└── requirements.txt                # Required Python packages
```

## Running
First, download all pretrained models:  
- Move this to the folder "berkeley_model":  
`python3 -m gdown https://drive.google.com/uc?id=1ADAqy8HwLNSsep-wMnzAcGTyULZWHRyW`  
- Move these to the folder "hasoc_model_base":  
`python3 -m gdown https://drive.google.com/uc?id=1o1b4vRKceIVEPSfXUrcrlSFvD6nQYc-9`  
`python3 -m gdown https://drive.google.com/uc?id=1Wem7CyJh8T-gb-ct8fLIejsPV_5frJls`  
`python3 -m gdown https://drive.google.com/uc?id=1_65mHOetpg9Z1S3X5zoAtplSZyyWlly3`  

Then, install the requirements:  
- If you are using a GPU, run `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126` (replace cu126 with your CUDA version).  
- If you are using a CPU-only, run `pip install torch torchvision torchaudio`.
- Install requirements using `pip3 install -r requirements.txt`

Finally, run the whole project:  
- `berkeley_model/...py` to train [BertMultiTasks model](#feature-extraction-model) on the Berkeley Dataset (1).  
- `new_features/add_extra_features.py` to infer the 13 features to the Hasoc dataset (2).  
- `hasoc_model_base/train.py` then `hasoc_model_base/predict.py` to train and evaluate [baseline RoBERTa and HateBERT models](#hate-speech-classification-models) on the Hasoc dataset (3).  

  Depending on the experiment, you have to run different files for training the fine-tuned models on augmented the Hasoc dataset (4):  
- `hasoc_model_boosted/train.py` then `hasoc_model_boosted/predict.py` if you have specific class weights to try with [V2.1 model](#hate-speech-classification-models).  
- `hasoc_model_boosted/grid_train.py` then `hasoc_model_boosted/grid_predict.py` if you have a grid of specific class weights to try with [V2.1 model](#hate-speech-classification-models).  
- `hasoc_model_boosted/load_train.py` then `hasoc_model_boosted/predict.py` if you have specific class weights to try with [V2.2 or V2.3 model](#hate-speech-classification-models).
- `hasoc_model_boosted/load_grid_train.py` then `hasoc_model_boosted/grid_predict.py` if you have a grid of specific class weights to try with [V2.2 or V2.3 model](#hate-speech-classification-models).  

  For the last two, change `freeze` and `experiment` variables to choose between [V2.2 or V2.3 model](#hate-speech-classification-models).  

Optional:  
- `new_features/features_cond_analysis.py` to perform a conditional analysis of tweets given the 13 features.  
- `hasoc_model_boosted/show_cm.py` to show the Top 1 confusion matrices of the results.  
- `hasoc_model_boosted/print_results.py` to print results.

## Ideas of improvement

Train on complementary datasets  
Extend to multilingual models  
Perform feature ablation studies  
Test robustness across other tweet platforms (TikTok, YouTube, Instagram...)  
Preprocess tweets (stemming, lemmatization...)

## References

<a name="ref1"></a>
**[1]** Kennedy, C. J., Bacon, G., Sahn, A., & von Vacano, C. (2020). Constructing interval variables via faceted Rasch measurement and multitask deep learning: A hate speech application. D-Lab, University of California, Berkeley; Department of Biomedical Informatics, Harvard Medical School; Department of Linguistics, UC Berkeley; Travers Department of Political Science, UC Berkeley; Digital Humanities, UC Berkeley. [Version: September 23, 2020]. Corresponding author: ck37@berkeley.edu.

<a name="ref2"></a>
**[2]** Mandl, T., Modha, S., Majumder, P., & Patel, D. H. (2019). Overview of the HASOC track at FIRE 2019: Hate Speech and Offensive Content Identification in Indo-European languages. In Proceedings of the 11th annual meeting of the Forum for Information Retrieval Evaluation (FIRE 2019). CEUR-WS.org. Available at: https://ceur-ws.org/Vol-2517/T3-1.pdf
