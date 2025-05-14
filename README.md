

Move this to the folder "paola"

gdown https://drive.google.com/uc?id=10hcRYw1qxgabsqiXMVbO-qF4LmaRJM4l   
gdown https://drive.google.com/uc?id=1ADAqy8HwLNSsep-wMnzAcGTyULZWHRyW  


Move this to the folder "coline"

gdown https://drive.google.com/uc?id=1PCzrY1vg6NcG_T9ogi46swyufyEOtOg9  
gdown https://drive.google.com/uc?id=1qj0GclRw8y3Y6jSOnyBS8rVd23K-Ebqr  
gdown https://drive.google.com/uc?id=1eCx2Lmbhh_eVAbyy-lpO8QSl3jSI6Fci  



# Hate Speech Detection & Classification using Roberta and HateBERT
SemEval 2019 - Task 6 - Identifying and Categorizing Offensive Language in Social Media 

## Description
This project explores hate speech and offensive language detection using transformer-based models. It proceeds in two main stages:

### Feature Extraction Model:
We first train a model on a labeled dataset containing 12 distinct categories. After achieving satisfying performance, we apply this model to a second dataset—HASOC 2019—to annotate tweets with these 12 inferred features.

### Hate Speech Classification Models:
Using the HASOC 2019 dataset, we train then fine-tune classification models (RoBERTa and HateBERT) to solve the three HASOC subtasks (A, B, and C). When fine-tuning, we also integrate the inferred features from the first stage into the classification models to improve performance.

## Subtasks
The HASOC 2019 dataset contains tweets in English, German, and Hindi. We focus on English tweets across three subtasks:
- Sub-task A: Offensive Language Identification (HOF vs. NOT)
Classify tweets as either:
HOF – Hate and Offensive: Contains hate speech, aggression, or profanity.
NOT – Non Hate-Offensive: Contains acceptable content without any offensive language.

- Sub-task B: Fine-Grained Offensive Type Classification
Only tweets labeled as HOF in sub-task A are used. Each HOF tweet is further categorized into:
HATE – Hate speech targeting groups based on race, gender, religion, etc.
OFFN – Offensive language targeting an individual or group, including insults or threats.
PRFN – Profane language without specific targeting (e.g., general swearing or cursing).
This task disambiguates the type of offense expressed in HOF tweets.

- Sub-task C: Offense Target Identification
Only tweets labeled HOF in sub-task A are included. The goal is to determine the target of the offensive language:
TIN – Targeted Insult: Direct insult/threat against individuals, groups, or other entities.
UNT – Untargeted: Profane or aggressive language without a specific target.


## Implementation

### Preprocessing
Tokenization

#### Deeplearning
BertMultiTask, RoBERTa, HateBERT

## Running
- Install requiremetns using `pip3 install -r requirements.txt`

- `coline/coline_model.py` to train the fine-tuned model on Hasoc Dataset.

- `coline/predict.py` to evaluate the fine-tuned model on Hasoc Dataset.

## Ideas of improvement

### Preprocessing
Stopwords Removal, Lemmatizaion, Stemming

### Vectorization
TFIDF, Count, Word2Vec, GloVe, fastText

### Classification
KNN, Naïve Bayes, SVM, Decision Trees, Random Forest, Logistic Regression, MLP, Adaboost, Bagging
