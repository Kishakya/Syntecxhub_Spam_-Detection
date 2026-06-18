# 📩 Spam Detection — Naive Bayes & Logistic Regression

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Project Overview

This project builds a text classification system to detect spam SMS messages using Natural Language Processing (NLP) techniques. Three models are trained and compared — Naive Bayes with CountVectorizer, Naive Bayes with TF-IDF, and Logistic Regression with TF-IDF — and the best-performing pipeline is saved for reuse on new messages.

---

## 📂 Dataset Description

| Property | Details |
|---|---|
| **Source** | [SMS Spam Collection Dataset — Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) |
| **Rows** | 5,572 |
| **Columns** | 2 (after cleanup) |
| **Target** | `label` (ham = 0, spam = 1) |
| **Class Distribution** | ~87% Ham, ~13% Spam |

### Features

| Feature | Description |
|---|---|
| `message` | Raw SMS text |
| `clean_message` | Preprocessed text (lowercased, punctuation removed, numbers removed) |
| `label_encoded` | Binary target — Ham (0), Spam (1) |

---

## 🛠️ Technologies Used

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` & `seaborn` | Data visualization |
| `scikit-learn` | Vectorization, model training, evaluation |
| `joblib` | Pipeline serialization |
| `re`, `string` | Text cleaning and preprocessing |

---

## 🤖 Models Used

Three model-vectorizer combinations were trained and compared:

| # | Model | Vectorizer |
|---|---|---|
| 1 | Naive Bayes (MultinomialNB) | CountVectorizer |
| 2 | Naive Bayes (MultinomialNB) | TF-IDF |
| 3 | Logistic Regression | TF-IDF |

**Text Preprocessing Pipeline:**
1. Convert to lowercase
2. Remove punctuation
3. Remove numbers
4. Strip extra whitespace
5. Vectorize using CountVectorizer or TF-IDF (max 5,000 features, English stop words removed)

---

## 📊 Evaluation Metrics

| Metric | Description |
|---|---|
| **Accuracy** | Overall correct predictions |
| **Precision** | Of all predicted spam, how many were actually spam |
| **Recall** | Of all actual spam, how many were correctly caught |
| **F1 Score** | Harmonic mean of Precision and Recall |

> For spam detection, **Recall** is the most critical metric — missing a spam message is worse than occasionally filtering a legitimate one.

---

## 📈 Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| **NB + CountVectorizer** | **0.9794** | 0.9437 | **0.8993** | **0.9209** |
| NB + TF-IDF | 0.9659 | 0.9912 | 0.7517 | 0.8550 |
| LR + TF-IDF | 0.9632 | 0.9909 | 0.7315 | 0.8417 |

**Key observations:**
- **Naive Bayes + CountVectorizer** was selected as the best model based on highest Recall and F1 Score
- TF-IDF models achieved near-perfect Precision (~0.991) but significantly lower Recall, meaning they missed more actual spam
- CountVectorizer's raw term frequency representation suits Naive Bayes better on this short-text dataset than weighted TF-IDF scores
- The saved pipeline correctly classified all 4 custom test messages

---

## 📁 Project Structure

```
spam-detection/
│
├── spam.csv                      # Raw dataset
├── Spam_ham_Analysis.ipynb       # Main notebook
├── spam_model.pkl                # Saved best model (NB + CountVectorizer)
├── spam_vectorizer.pkl           # Saved CountVectorizer
├── requirements.txt              # Dependencies
└── README.md                     # Project documentation
```

---

## 👤 Author

**Kishkaya Gayathri**
Bachelor of Science in Information Technology specialized in Artificial Intelligence
[SLIIT] — [2026]
