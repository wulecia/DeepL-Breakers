# Hate Speech Detection & Classification using BERTMultiTaskModel, RoBERTa and HateBERT

## Description
This project explores hate speech and offensive language detection using transformer-based models. It proceeds in two main stages:

### Feature Extraction Model:
1. Train 13-label model on base dataset: We first train a BERTMultiTaskModel model on a labeled [dataset from Berkeley](#ref1) containing 13 distinct categories (8 numerical: *sentiment*, *respect*, *insult*, *humiliate*, *status*, *dehumanize*, *attack_defend*, *hatespeech*; 5 binary: *target_race*, *target_religion*, *target_origin*, *target_gender*, *target_sexuality*).
2. Infer 13 features for each HASOC tweet: After achieving satisfying performance, we apply this model to a second dataset — [HASOC 2019](#ref2) — to annotate tweets with these 13 inferred features.

### Hate Speech Classification Models:
3. Train RoBERTa & HateBERT models on HASOC Sub-tasks A, B, and C: Using the HASOC 2019 dataset, we train classification models (RoBERTa and HateBERT) to solve the three HASOC subtasks (A, B, and C).
4. Fine-tune with 13 additional features to boost performance: Then we fine-tune these models, by integrating the 13 inferred features from the first stage into the classification models to improve performance on the 3 tasks.

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

## Running
- Move this to the folder "berkeley_model":  
python3 -m gdown https://drive.google.com/uc?id=1ADAqy8HwLNSsep-wMnzAcGTyULZWHRyW  
- Move these to the folder "hasoc_model_base":  
python3 -m gdown https://drive.google.com/uc?id=1o1b4vRKceIVEPSfXUrcrlSFvD6nQYc-9
python3 -m gdown https://drive.google.com/uc?id=1Wem7CyJh8T-gb-ct8fLIejsPV_5frJls
python3 -m gdown https://drive.google.com/uc?id=1_65mHOetpg9Z1S3X5zoAtplSZyyWlly3

- Install requirements using `pip3 install -r requirements.txt`  
- `berkeley_model/...py` to train BertMultiTasks model on Berkeley Dataset (1).  
- `new_features/add_extra_features.py` to infer the 13 features to Hasoc Dataset (2).  
- `hasoc_model_base/train.py` then `hasoc_model_base/predict.py` to train and evaluate baseline RoBERTa and HateBERT models on Hasoc Dataset (3).  

Depending on the experiment, you have to run different files for training the fine-tuned models on Hasoc Dataset (4):  
- `hasoc_model_boosted/train.py` then `hasoc_model_boosted/predict.py` if you have specific class weights to try with V2.1 model.  
- `hasoc_model_boosted/grid_train.py` then `hasoc_model_boosted/grid_predict.py` if you have a grid of specific class weights to try V2.1 model.  
- `hasoc_model_boosted/load_train.py` then `hasoc_model_boosted/predict.py` if you have specific class weights to try with V2.2 or V2.3 model.
- `hasoc_model_boosted/load_grid_train.py` then `hasoc_model_boosted/grid_predict.py` if you have a grid of specific class weights to try V2.2 or V2.3 model.  
For the last two, change `freeze` and `experiment` arguments to choose between V2.2 or V2.3 models.  

Optional:  
- `new_features/features_descr_analysis.py` to perform a descriptive analysis of the 13 features.  
- `hasoc_model_boosted/show_cm.py` to show the Top 1 confusion matrices of the results.  
- `hasoc_model_boosted/print_results.py` to print results.

## Ideas of improvement

Train on complementary datasets  
Extend to multilingual models  
Perform feature ablation studies  
Test robustness across tweet platforms  
Preprocess tweets (stemming, lemmatization...)

## References

<a name="ref1"></a>
**[1]** Kennedy, C. J., Bacon, G., Sahn, A., & von Vacano, C. (2020). Constructing interval variables via faceted Rasch measurement and multitask deep learning: A hate speech application. D-Lab, University of California, Berkeley; Department of Biomedical Informatics, Harvard Medical School; Department of Linguistics, UC Berkeley; Travers Department of Political Science, UC Berkeley; Digital Humanities, UC Berkeley. [Version: September 23, 2020]. Corresponding author: ck37@berkeley.edu.

<a name="ref2"></a>
**[2]** Mandl, T., Modha, S., Majumder, P., & Patel, D. H. (2019). Overview of the HASOC track at FIRE 2019: Hate Speech and Offensive Content Identification in Indo-European languages. In Proceedings of the 11th annual meeting of the Forum for Information Retrieval Evaluation (FIRE 2019). CEUR-WS.org. Available at: https://ceur-ws.org/Vol-2517/T3-1.pdf
