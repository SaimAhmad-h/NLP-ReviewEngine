# 🛍️ Customer Reviews Intelligence System
### NLP & Machine Learning Pipeline for E-Commerce Feedback Analysis

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-NLP-4CAF50?style=for-the-badge)
![Gradio](https://img.shields.io/badge/Gradio-Interface-FF7C00?style=for-the-badge)

> **Course:** AI4001 – Fundamentals of Natural Language Processing
> **Institution:** NUCES Chiniot-Faisalabad Campus
> **Department:** Artificial Intelligence and Data Science

---

## 📌 Project Overview

An end-to-end NLP pipeline that analyzes customer reviews from an e-commerce platform. The system handles mixed English and Roman Urdu text, performs sentiment analysis, classifies customer intent, and discovers hidden topics using classical NLP and machine learning techniques.

> **Note:** If the real dataset (`Womens Clothing E-Commerce Reviews.csv`) is not found, the notebook automatically generates a synthetic dataset of 240 samples (40 unique reviews × 6 duplicates) for demonstration purposes. All reported metrics below are from this synthetic dataset.

---

## 🎯 Features

| Feature | Method |
|---|---|
| Text Preprocessing | Tokenization, Lemmatization, Stopword Removal |
| Feature Extraction | Bag of Words + TF-IDF (CountVectorizer / TfidfVectorizer) |
| Sentiment Analysis | VADER (Rule-Based) + Logistic Regression (ML) |
| Intent Classification | Logistic Regression — 4 intent classes |
| Topic Modelling | NMF (Non-negative Matrix Factorization) — 5 topics |
| Evaluation | Accuracy, Precision, Recall, F1 Score, Confusion Matrix |
| Live Interface | Gradio Web App |

---

## 🔧 Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/customer-reviews-nlp.git
cd customer-reviews-nlp
```

### 2. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn vaderSentiment gradio datasets
```

### 3. (Optional) Use Real Dataset
Place the CSV file in the project root:
- [Women's E-Commerce Clothing Reviews — Kaggle](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)

If no file is found, the notebook auto-generates a synthetic dataset with English and Roman Urdu reviews across positive, negative, neutral, and complaint categories.

### 4. Run the Notebook
```bash
jupyter notebook LT1_24F0001.ipynb
```

Run all cells top to bottom. The final cell launches the Gradio interface.

---

## 📓 Notebook Walkthrough

### Cell 1 — Library Installation
Installs all required packages via `subprocess` at runtime.

### Cell 2 — Imports & NLTK Downloads
Loads all libraries and downloads NLTK resources: `punkt`, `stopwords`, `wordnet`.

### Cell 3 — Dataset Loading & Labelling
Attempts to load the real CSV. Falls back to a synthetic dataset of 240 samples if not found. Sentiment labels are derived from ratings:

| Rating | Sentiment Label |
|---|---|
| 4–5 | Positive |
| 3 | Neutral |
| 1–2 | Negative |

**Synthetic dataset class distribution:**
```
Negative    120
Neutral      60
Positive     60
```

### Cell 4 — Text Preprocessing Pipeline
Six-step cleaning pipeline applied to every review:
```
Raw Text → Lowercase → Remove URLs → Remove Punctuation
         → Tokenize → Remove Stopwords → Lemmatize → Clean Text
```

Average word count drops from **6.95** (original) to **5.0** (cleaned).

Sample:
```
BEFORE: "It's fine, does the job."
AFTER : "fine job"

BEFORE: "Bahut achha product hai, zaroor kharidein."
AFTER : "bahut achha product hai zaroor kharidein"
```

> Note: Roman Urdu tokens pass through unchanged since NLTK stopwords are English-only.

### Cell 5 — Preprocessing Visualisation
Histograms comparing word count distributions before and after preprocessing.

### Cell 6 — Feature Extraction: BoW & TF-IDF
Converts cleaned text to numerical matrices (192 training samples, 142 vocabulary size).

**Top 15 tokens by frequency:**
```
product, quality, hai, wrong, delivery, refund,
size, average, fit, please, great, item, dress, bahut, material
```

### Cell 7 — BoW vs TF-IDF Comparison (Naive Bayes)
Naive Bayes trained on both feature sets.

| Method | Accuracy | F1 Score |
|---|---|---|
| Bag of Words | 1.0000 | 1.0000 |
| TF-IDF | 1.0000 | 1.0000 |

> ⚠️ Both methods achieve perfect scores because the synthetic dataset is built by repeating the same 40 sentences 6 times. The model simply memorizes the repeated patterns. These scores do NOT reflect real-world performance.

### Cell 8 — VADER Rule-Based Sentiment
Applies the VADER lexicon to the full dataset without any training.

| Metric | Score |
|---|---|
| Accuracy | 0.6250 |
| F1 Score (Weighted) | 0.6303 |

```
              precision    recall    f1-score
Negative        0.81        0.65      0.72
Neutral         0.36        0.40      0.38
Positive        0.62        0.80      0.70
```

> VADER struggles most with **Neutral** reviews (F1: 0.38), which is expected — VADER is a lexicon tool designed for clearly positive/negative text, not borderline cases.

### Cell 9 — VADER Confusion Matrix
Heatmap of VADER predictions vs actual labels.

### Cell 10 — Logistic Regression Sentiment
Supervised ML model trained on TF-IDF features (192 train, 48 test).

| Metric | Score |
|---|---|
| Accuracy | 1.0000 |
| F1 Score (Weighted) | 1.0000 |

> ⚠️ Perfect score is a direct consequence of the duplicated synthetic data. Train and test sets contain near-identical reviews, so the model is effectively memorizing, not generalizing.

### Cell 11 — LR Confusion Matrix
All 48 test samples correctly classified (zero misclassifications).

### Cell 12 — Intent Label Assignment
Intent labels assigned via keyword matching rules:

| Intent | Trigger Keywords |
|---|---|
| Refund Request | refund, money back, paisa wapas |
| Delivery Issue | delivery, shipping, late, kab ayega, not received |
| Complaint | broken, damaged, worst, ghatia, kharab |
| General Query | everything else |

**Intent distribution in synthetic data:**
```
General Query     120
Complaint          66
Refund Request     30
Delivery Issue     24
```

### Cell 13 — Intent Classifier
Logistic Regression trained on TF-IDF features for 4-class intent prediction.

| Metric | Score |
|---|---|
| Accuracy | 1.0000 |
| F1 Score (Weighted) | 1.0000 |

> ⚠️ Same caveat applies — perfect score due to duplicated synthetic data.

### Cell 14 — Intent Confusion Matrix
4×4 heatmap — all test samples predicted correctly.

### Cell 15 — NMF Topic Modelling
Unsupervised discovery of 5 topics from the TF-IDF matrix (1000 features, min_df=2, max_df=0.90).

**Topic assignments (as labelled in code):**

| Topic | Label in Code | Actual Top Keywords |
|---|---|---|
| 1 | Product Quality | quality, average, price, great, bad, buy |
| 2 | Sizing & Fit | product, refund, damaged, okay, normal |
| 3 | Delivery & Shipping | wrong, item, size, replacement, shipped |
| 4 | Customer Service | hai, kharidein, bohat, ghatia (Urdu cluster) |
| 5 | Returns & Refunds | material, fit, purchase, stitching, worst |

> ⚠️ Topic labels are manually assigned and do NOT accurately reflect the actual keywords. For example, Topic 4 ("Customer Service") contains purely Roman Urdu words, and Topic 2 ("Sizing & Fit") contains refund/damage words. This mismatch is a known limitation of running NMF on a small, duplicate-heavy synthetic dataset.

**Document-topic distribution:**
```
Sizing & Fit        60
Delivery & Shipping 54
Returns & Refunds   48
Product Quality     42
Customer Service    36
```

### Cell 16 — Topic Distribution Chart
Bar chart of how reviews are distributed across the 5 NMF topics.

### Cell 17 — Comprehensive Evaluation Summary

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| VADER (Rule-Based) | 0.6250 | 0.6510 | 0.6250 | 0.6303 |
| LR Sentiment (TF-IDF) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| LR Intent (TF-IDF) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

> **Most reliable metric:** F1 Score (Weighted) — chosen because the dataset is class-imbalanced, making raw accuracy misleading. For VADER (the only model evaluated on real conditions), weighted F1 of 0.63 is the honest performance indicator.

### Cell 18 — Model Comparison Chart
Grouped bar chart visualising all four metrics across all three models.

### Cell 19 — Gradio Interface
Live web app that runs all three models on any typed review.

**Example:**
```
Input:  "My order never arrived, I want a refund!"

Output:
  Sentiment → 😞 Negative  (VADER: -0.71)
              🤖 ML Prediction: Negative (confidence: 0.93)
  Intent    → 🏷️ Refund Request (confidence: 0.88)
  Topic     → 📌 Returns & Refunds
              🔑 Keywords: refund, return, material, fit...
```

Launched with `share=True` — generates a public Gradio URL valid for 7 days.

---

## 📊 Honest Results Summary

| Model | Accuracy | F1 Score | Notes |
|---|---|---|---|
| VADER (Rule-Based) | **0.6250** | **0.6303** | Evaluated on full 240-sample synthetic set. Realistic score. |
| LR Sentiment (TF-IDF) | 1.0000 | 1.0000 | Inflated — dataset contains duplicate reviews in train/test. |
| LR Intent (TF-IDF) | 1.0000 | 1.0000 | Inflated — same reason. |

**On the real Women's Clothing dataset, expect approximately:**
- LR Sentiment: 0.85–0.91 F1
- LR Intent: 0.80–0.87 F1 (keyword-based labels are noisy)
- VADER: 0.60–0.70 F1 (consistent with synthetic result)

---

## ⚠️ Known Limitations

1. **Duplicate synthetic data** — The 240-sample dataset is 40 unique sentences repeated 6 times. This causes data leakage between train and test splits, producing artificially perfect ML scores (1.0).

2. **Roman Urdu handling** — NLTK stopwords and VADER are English-only. Roman Urdu tokens are not cleaned or understood by VADER, reducing its accuracy on mixed-language reviews.

3. **NMF topic mislabelling** — Topic labels are hardcoded and do not match the actual top keywords discovered by NMF on this dataset.

4. **Intent labels from rules** — Intent ground truth is generated by the same keyword rules used at inference time, meaning the intent classifier is essentially learning to replicate a rule-based system rather than true intent understanding.

---

## 🧠 Key Concepts

**Why TF-IDF over Bag of Words?**
BoW treats all words equally. TF-IDF penalises words that appear in nearly every document (like "product") and rewards rare but discriminative terms — giving the classifier stronger signal.

**Why Logistic Regression?**
Fast, interpretable, and strong on text classification. Word-level coefficients directly reveal which tokens drive each prediction.

**Why F1 Score as primary metric?**
The dataset is class-imbalanced. Plain accuracy would be misleadingly high if the model ignored minority classes. Weighted F1 balances precision and recall across all classes fairly.

**What is NMF?**
Non-negative Matrix Factorization factorizes the TF-IDF matrix into topic-word and document-topic components. It is unsupervised — it discovers latent themes without needing labelled data. On small or homogeneous datasets, topics may not be cleanly separable.

---

## 📁 Output Files

| File | Description |
|---|---|
| `preprocessing_lengths.png` | Word count histograms before/after cleaning |
| `vader_cm.png` | VADER confusion matrix heatmap |
| `lr_cm.png` | Logistic Regression sentiment confusion matrix |
| `intent_cm.png` | Intent classifier confusion matrix |
| `intent_dist.png` | Intent class distribution bar chart |
| `topic_dist.png` | NMF topic distribution bar chart |
| `model_comparison.png` | Multi-metric model comparison bar chart |
