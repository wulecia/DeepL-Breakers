# Hate Speech Detection & Classification using BERTMultiTaskModel, RoBERTa and HateBERT

## Description
This project explores hate speech and offensive language detection using transformer-based models. It proceeds in two main stages:

### Feature Extraction Model:
1. Train 13-label model on base dataset: We first train a BERTMultiTaskModel model on a labeled [dataset from Berkeley](#ref1) containing 13 distinct categories (8 numerical: *sentiment*, *respect*, *insult*, *humiliate*, *status*, *dehumanize*, *attack_defend*, *hatespeech*; 5 binary: *target_race*, *target_religion*, *target_origin*, *target_gender*, *target_sexuality*).
2. Infer 13 features for each HASOC tweet: After achieving satisfying performance, we apply this model to a second dataset — [HASOC 2019](#ref2) — to annotate tweets with these 13 inferred features.

### Hate Speech Classification Models:
3. Train RoBERTa & HateBERT models on HASOC Sub-tasks A, B, and C: Using the HASOC 2019 dataset, we train classification models (RoBERTa and HateBERT) to solve the three HASOC subtasks (A, B, and C).
4. Fine-tune with 13 additional features to boost performance: Then we fine-tune these models, by integrating the inferred features from the first stage into the classification models to improve performance.

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


## Implementation

### Preprocessing
Tokenization

### Deeplearning
BERTMultiTaskModel, RoBERTa, HateBERT

## Running
- Move these to the folder "paola":  
python3 -m gdown https://drive.google.com/uc?id=10hcRYw1qxgabsqiXMVbO-qF4LmaRJM4l  
python3 -m gdown https://drive.google.com/uc?id=1ADAqy8HwLNSsep-wMnzAcGTyULZWHRyW  
- Move this to the folder "paola/results":  
python3 -m gdown https://drive.google.com/uc?1NYlv9REz6C9Ubsx-i-vaXgBPPhuVGfps=1AbcD3FgHxyz12345678  
unzip best_model_2025-05-21_14-23-53.zip  
- Move these to the folder "coline":  
python3 -m gdown https://drive.google.com/uc?id=1PCzrY1vg6NcG_T9ogi46swyufyEOtOg9  
python3 -m gdown https://drive.google.com/uc?id=1qj0GclRw8y3Y6jSOnyBS8rVd23K-Ebqr  
python3 -m gdown https://drive.google.com/uc?id=1eCx2Lmbhh_eVAbyy-lpO8QSl3jSI6Fci  
- Install requirements using `pip3 install -r requirements.txt`
- `paola/...py` to train the BertMultiTasks model on Berkeley Dataset (1).
- `coline/coline_features.py` to infer the 13 features to Hasoc Dataset (2).
- `clara/...py` to train the RoBERTa and HateBERT models on Hasoc Dataset (3).
- `boosted_model/train.py` to train the fine-tuned model on Hasoc Dataset (4).
- `boosted_model/predict.py` to evaluate the fine-tuned model on Hasoc Dataset (4).

- Move these to the folder "hasoc_model/results/models":  
python3 -m gdown https://drive.google.com/uc?id=17o32Sa6Mb7Soz21KEziMcPgfybHAtLmt  
python3 -m gdown https://drive.google.com/uc?id=10tkbyTi3DXT8bDe5A2EJuJotJVxa-Qgn  
python3 -m gdown https://drive.google.com/uc?id=1Q2oC3KpcyuTujbn15ibl41k6OvKzyB6S  
unzip best_no_features_A_roberta-base_full.pt.zip  
unzip best_no_features_B_hateBERT_full.pt.zip  
unzip best_no_features_C_hateBERT_full.pt.zip

- Move these to the folder "coline":  
python3 -m gdown https://drive.google.com/uc?id=1Uxwq9xVr1UHNXrXvd1_Rewz1Fe-QHevs  
python3 -m gdown https://drive.google.com/uc?id=1INMtTKjdX6LPENm-ktBuO5NPGEuKJyph  
python3 -m gdown https://drive.google.com/uc?id=1G6ZK1NXU_0FwuRYvZFy_tyj5vqdjMjmP

## Ideas of improvement

### Preprocessing
Stopwords Removal, Lemmatizaion, Stemming

### Vectorization
TFIDF, Count, Word2Vec, GloVe, fastText

### Classification
KNN, Naïve Bayes, SVM, Decision Trees, Random Forest, Logistic Regression, MLP, Adaboost, Bagging

## References

<a name="ref1"></a>
**[1]** Kennedy, C. J., Bacon, G., Sahn, A., & von Vacano, C. (2020). Constructing interval variables via faceted Rasch measurement and multitask deep learning: A hate speech application. D-Lab, University of California, Berkeley; Department of Biomedical Informatics, Harvard Medical School; Department of Linguistics, UC Berkeley; Travers Department of Political Science, UC Berkeley; Digital Humanities, UC Berkeley. [Version: September 23, 2020]. Corresponding author: ck37@berkeley.edu.

<a name="ref2"></a>
**[2]** Mandl, T., Modha, S., Majumder, P., & Patel, D. H. (2019). Overview of the HASOC track at FIRE 2019: Hate Speech and Offensive Content Identification in Indo-European languages. In Proceedings of the 11th annual meeting of the Forum for Information Retrieval Evaluation (FIRE 2019). CEUR-WS.org. Available at: https://ceur-ws.org/Vol-2517/T3-1.pdf
