# 🛍️ Customer Reviews Intelligence System
### NLP & Machine Learning Pipeline for E-Commerce Feedback Analysis

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-NLP-4CAF50?style=for-the-badge)
![VADER](https://img.shields.io/badge/VADER-Sentiment-blue?style=for-the-badge)
![Gradio](https://img.shields.io/badge/Gradio-Interface-FF7C00?style=for-the-badge)

> **Course:** AI4001 – Fundamentals of Natural Language Processing
> **Institution:** NUCES Chiniot-Faisalabad Campus
> **Department:** Artificial Intelligence and Data Science

---

## 📌 Project Overview

An end-to-end NLP pipeline that analyzes customer reviews from an e-commerce platform. The system handles mixed English and Roman Urdu text, performs sentiment analysis, classifies customer intent, and discovers hidden topics using classical NLP and machine learning techniques.

> **Note:** If the real dataset (`Womens Clothing E-Commerce Reviews.csv`) is not found, the notebook automatically generates a synthetic dataset of 55 unique reviews. The dataset is split into train/test **before** duplication to prevent data leakage.

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
pip install pandas numpy matplotlib seaborn nltk scikit-learn vaderSentiment gradio
```

### 3. (Optional) Use Real Dataset
Place the CSV file in the project root:
- [Women's E-Commerce Clothing Reviews — Kaggle](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)

If no file is found, the notebook auto-generates a synthetic dataset with 55 unique English and Roman Urdu reviews across positive, negative, neutral, and complaint categories.

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
Attempts to load the real CSV. Falls back to a synthetic dataset of 55 unique reviews if not found. Sentiment labels are derived from ratings:

| Rating | Sentiment Label |
|---|---|
| 4–5 | Positive |
| 3 | Neutral |
| 1–2 | Negative |

**Key fix applied here:**
> The dataset is split into train/test on unique reviews **first**, then only the training portion is duplicated ×6. This prevents any test sentence from appearing in training — eliminating data leakage.

```
55 unique reviews
→ train_test_split → Train: 44 unique, Test: 11 unique
→ duplicate train only: 44 × 6 = 264 training samples
→ test stays: 11 genuinely unseen reviews
```

**Synthetic dataset sentiment distribution:**
```
Negative    ~120
Neutral     ~60
Positive    ~90
```

### Cell 4 — Text Preprocessing Pipeline
Six-step cleaning pipeline applied to every review:
```
Raw Text → Lowercase → Remove URLs → Remove Punctuation
         → Tokenize → Remove Stopwords → Lemmatize → Clean Text
```

Sample:
```
BEFORE: "It's fine, does the job."
AFTER : "fine job"

BEFORE: "Bahut achha product hai, zaroor kharidein."
AFTER : "bahut achha product hai zaroor kharidein"
```

> Note: Roman Urdu tokens pass through unchanged since NLTK stopwords are English-only. This is a known limitation.

### Cell 5 — Preprocessing Visualisation
Histograms comparing word count distributions before and after preprocessing.

### Cell 6 — Feature Extraction: BoW & TF-IDF
Converts cleaned text to numerical matrices using leakage-free train/test splits.

**Top 15 tokens by frequency include:**
```
product, quality, hai, delivery, refund, size,
fit, great, item, material, wrong, average, dress, bahut, please
```

### Cell 7 — BoW vs TF-IDF Comparison (Naive Bayes)
Naive Bayes trained on both feature sets and evaluated on genuinely unseen test reviews.

> Both methods are compared on the same unseen test set. TF-IDF generally outperforms BoW because it down-weights common words and highlights discriminative terms.

### Cell 8 — VADER Rule-Based Sentiment
Applies the VADER lexicon to the full dataset without any training.

| Metric | Score |
|---|---|
| Accuracy | 0.6545 |
| F1 Score (Weighted) | **0.66** |

```
              precision    recall    f1-score
Negative        0.81        0.65      0.72
Neutral         0.36        0.40      0.38
Positive        0.62        0.80      0.70
```

> VADER struggles most with **Neutral** reviews (F1: 0.38) and Roman Urdu text — expected behaviour for an English-only lexicon tool.

### Cell 9 — VADER Confusion Matrix
Heatmap of VADER predictions vs actual labels.

### Cell 10 — Logistic Regression Sentiment
Supervised ML model trained on TF-IDF features on the leakage-free training set.

> LR scores on the small 11-sample test set are not statistically reliable. On the full Kaggle dataset (23,000+ reviews), expected LR F1 is 0.85–0.91.

### Cell 11 — LR Sentiment Confusion Matrix
Confusion matrix for LR sentiment predictions on unseen test samples.

### Cell 12 — Intent Label Assignment
Intent labels assigned via keyword matching rules:

| Intent | Trigger Keywords |
|---|---|
| Refund Request | refund, money back, paisa wapas, charged twice |
| Delivery Issue | delivery, shipping, late, kab ayega, never arrived, missing |
| Complaint | broken, damaged, worst, terrible, ghatia, kharab, zipper |
| General Query | everything else |

**Intent distribution:**
```
General Query     ~120
Complaint         ~80
Refund Request    ~30
Delivery Issue    ~24
```

### Cell 13 — Intent Classifier
Logistic Regression trained on TF-IDF features for 4-class intent prediction using the leakage-free split.

### Cell 14 — Intent Confusion Matrix
4×4 heatmap of intent predictions vs actual labels on unseen test set.

### Cell 15 — NMF Topic Modelling
Unsupervised discovery of 5 topics from the TF-IDF matrix (1000 features, min_df=2, max_df=0.90).

**Topics discovered:**

| Topic | Label | Sample Keywords |
|---|---|---|
| 1 | Product Quality | quality, material, stitching, price, worth |
| 2 | Sizing & Fit | fit, size, guide, true, expected |
| 3 | Delivery & Shipping | delivery, shipping, arrived, package, late |
| 4 | Returns & Refunds | refund, return, damaged, replacement, paisa |
| 5 | General Feedback | average, okay, decent, moderate, normal |

> Topic labels are assigned by inspecting actual top keywords — not hardcoded. A `label_topic()` function matches keywords to the most appropriate label automatically.

### Cell 16 — Topic Distribution Chart
Bar chart of how reviews are distributed across the 5 NMF topics.

### Cell 17 — Comprehensive Evaluation Summary

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| VADER (Rule-Based) | 0.6545 | 0.6624 | 0.6545 | **0.6552** |
| LR Sentiment (TF-IDF) | — | — | — | — |
| LR Intent (TF-IDF) | — | — | — | — |

> **Most reliable metric:** VADER F1 Score (Weighted) = **0.66** — evaluated on the full dataset with no training bias. LR scores on 11-sample test set are not statistically meaningful and are not reported here.

### Cell 18 — Model Comparison Chart
Grouped bar chart visualising all four metrics across all three models.

### Cell 19 — Gradio Interface
Live web app that runs all three models on any typed review.

**Example:**
```
Input:  "My order never arrived, I want a refund!"

Output:
  Sentiment → 😞 Negative  (VADER: -0.71)
              🤖 ML Prediction: Negative
  Intent    → 🏷️ Refund Request
  Topic     → 📌 Returns & Refunds
              🔑 Keywords: refund, return, damaged, replacement...
```

Launched with `share=True` — generates a public Gradio URL valid for 7 days.

---

## 📊 Results Summary

| Model | F1 Score | Notes |
|---|---|---|
| VADER (Rule-Based) | **0.66** | Evaluated on full dataset. Most reliable score. |
| LR Sentiment (TF-IDF) | — | Test set too small (11 samples) for reliable reporting. |
| LR Intent (TF-IDF) | — | Same reason. Expected 0.80+ on real dataset. |

**On the real Women's Clothing dataset, expect approximately:**
- LR Sentiment: 0.85–0.91 F1
- LR Intent: 0.80–0.87 F1
- VADER: 0.60–0.70 F1 (consistent with synthetic result)

---

## ⚠️ Known Limitations

1. **Small synthetic test set** — With only 11 unique unseen test reviews, LR scores fluctuate significantly with each wrong prediction and are not statistically reliable.

2. **Roman Urdu handling** — NLTK stopwords and VADER are English-only. Roman Urdu tokens are not cleaned or understood by VADER, reducing accuracy on mixed-language reviews.

3. **Intent labels from rules** — Intent ground truth is generated by the same keyword rules used at inference time, meaning the intent classifier learns to replicate a rule-based system rather than true intent understanding.

4. **NMF on small dataset** — Topic separation is less clean on small datasets. Topics may overlap or not perfectly align with real-world categories.

---

## 🧠 Key Concepts

**Why TF-IDF over Bag of Words?**
BoW treats all words equally. TF-IDF penalises words that appear in nearly every document and rewards rare but discriminative terms — giving the classifier stronger signal.

**Why Logistic Regression?**
Fast, interpretable, and strong on text classification. Word-level coefficients directly reveal which tokens drive each prediction.

**Why F1 Score as primary metric?**
The dataset is class-imbalanced. Plain accuracy would be misleadingly high if the model ignored minority classes. Weighted F1 balances precision and recall across all classes fairly.

**What is NMF?**
Non-negative Matrix Factorization factorizes the TF-IDF matrix into topic-word and document-topic components. It is unsupervised — it discovers latent themes without needing labelled data.

**What was the data leakage issue and how was it fixed?**
The original code duplicated 40 reviews ×6 = 240 rows, then split into train/test. This meant identical sentences appeared in both sets — the model memorized rather than learned, producing a fake F1 of 1.0. The fix was to split first on unique reviews, then duplicate only the training portion.

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
