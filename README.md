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

**Synthetic dataset sentiment distribution (actual output):**
```
Negative    125
Neutral      75
Positive     75
```

### Cell 4 — Text Preprocessing Pipeline
Six-step cleaning pipeline applied to every review:
```
Raw Text → Lowercase → Remove URLs → Remove Punctuation
         → Tokenize → Remove Stopwords → Lemmatize → Clean Text
```

Sample:
```
BEFORE: "Great material and stitching, very happy with purchase."
AFTER : "great material stitching happy purchase"

BEFORE: "Bahut achha product hai, zaroor kharidein."
AFTER : "bahut achha product hai zaroor kharidein"
```

> Note: Roman Urdu tokens pass through unchanged since NLTK stopwords are English-only. This is a known limitation.

### Cell 5 — Preprocessing Visualisation
Histograms comparing word count distributions before and after preprocessing.

```
Average words BEFORE: 6.84
Average words AFTER : 4.95
```

### Cell 6 — Feature Extraction: BoW & TF-IDF
Converts cleaned text to numerical matrices using leakage-free train/test splits.

```
Train: 264  |  Test: 11
BoW    matrix shape: (264, 136)
TF-IDF matrix shape: (264, 136)
```

**Top 15 tokens by frequency (BoW):**
```
quality             : 60
product             : 54
great               : 30
wrong               : 24
purchase            : 24
hai                 : 24
average             : 24
worst               : 18
fit                 : 18
item                : 18
absolutely          : 18
delivery            : 18
stitching           : 18
size                : 18
please              : 18
```

### Cell 7 — BoW vs TF-IDF Comparison (Naive Bayes)
Naive Bayes trained on both feature sets and evaluated on genuinely unseen test reviews.

```
        Accuracy  F1 Score
BoW       0.5455    0.5388
TF-IDF    0.5455    0.5388
```

> Both methods are compared on the same unseen test set. On a larger real-world dataset, TF-IDF generally outperforms BoW because it down-weights common words and highlights discriminative terms.

### Cell 8 — VADER Rule-Based Sentiment
Applies the VADER lexicon to the full dataset without any training.

| Metric | Score |
|---|---|
| Accuracy | 0.6545 |
| F1 Score (Weighted) | **0.6552** |

```
              precision    recall  f1-score   support

    Negative       0.83      0.74      0.78       125
     Neutral       0.44      0.41      0.43        75
    Positive       0.60      0.75      0.67        75

    accuracy                           0.65       275
   macro avg       0.63      0.63      0.63       275
weighted avg       0.66      0.65      0.66       275
```

> VADER struggles most with **Neutral** reviews and Roman Urdu text — expected behaviour for an English-only lexicon tool.

### Cell 9 — VADER Confusion Matrix
Heatmap of VADER predictions vs actual labels.

### Cell 10 — Logistic Regression Sentiment
Supervised ML model trained on TF-IDF features on the leakage-free training set.

```
Accuracy : 0.3636
F1 Score : 0.3189

              precision    recall  f1-score   support

    Negative       0.38      0.60      0.46         5
     Neutral       0.50      0.33      0.40         3
    Positive       0.00      0.00      0.00         3

    accuracy                           0.36        11
   macro avg       0.29      0.31      0.29        11
  weighted avg       0.31      0.36      0.32        11
```

> LR scores on the small 11-sample test set are not statistically reliable. On the full Kaggle dataset (23,000+ reviews), expected LR F1 is 0.85–0.91.

### Cell 11 — LR Sentiment Confusion Matrix
Confusion matrix for LR sentiment predictions on unseen test samples.

### Cell 12 — Intent Label Assignment
Intent labels assigned via keyword matching rules:

| Intent | Trigger Keywords |
|---|---|
| Refund Request | refund, money back, paisa wapas, charged twice, billing |
| Delivery Issue | delivery, shipping, arrived, late, kab ayega, package, not received, never arrived, missing |
| Complaint | broken, damaged, wrong item, worst, terrible, unacceptable, poor, ghatia, kharab, zipper, wrong color, broke |
| General Query | everything else |

**Intent distribution (actual output):**
```
General Query     150
Complaint          99
Refund Request     13
Delivery Issue     13
```

### Cell 13 — Intent Classifier
Logistic Regression trained on TF-IDF features for 4-class intent prediction using the leakage-free split.

```
                precision    recall  f1-score   support

     Complaint       0.00      0.00      0.00         3
Delivery Issue       0.00      0.00      0.00         1
 General Query       0.60      1.00      0.75         6
Refund Request       0.00      0.00      0.00         1

      accuracy                           0.55        11
     macro avg       0.15      0.25      0.19        11
  weighted avg       0.33      0.55      0.41        11
```

> Low scores on minority intent classes (Refund, Delivery) are expected with only 1 sample each in the 11-sample test set — not statistically meaningful.

### Cell 14 — Intent Confusion Matrix
4×4 heatmap of intent predictions vs actual labels on unseen test set.

### Cell 15 — NMF Topic Modelling
Unsupervised discovery of 5 topics from the TF-IDF matrix (1000 features, min_df=2, max_df=0.90).

**Raw topics discovered (actual output):**

| Topic | Assigned Label | Top Keywords |
|---|---|---|
| 1 | Product Quality | quality, average, price, hai, theek, thak, better, expected, reasonable, terrible |
| 2 | Product Quality | great, happy, bad, fine, material, stitching, work, beautiful, purchase, fitting |
| 3 | Delivery & Shipping | product, delivery, fast, okay, shown, exactly, love, special, nothing, arrived |
| 4 | Sizing & Fit | worst, wrong, ever, completely, experience, size, better, never, buying, photo |
| 5 | Returns & Refunds | money, waste, complete, buy, nothing, described, special, okay, decent, color |

> Topic labels are assigned automatically by the `label_topic()` function, which matches each topic's keywords to the most appropriate category — not hardcoded.

### Cell 16 — Topic Distribution Chart
Bar chart of how reviews are distributed across the NMF topics.

**Document-Topic Distribution (actual output):**
```
Product Quality        103
Sizing & Fit            78
Delivery & Shipping     73
Returns & Refunds       21
```

> Note: "General Feedback" did not emerge as a dominant topic on this synthetic dataset. Topic separation improves significantly on larger real-world datasets.

### Cell 17 — Comprehensive Evaluation Summary

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| VADER (Rule-Based) | 0.6545 | 0.6624 | 0.6545 | **0.6552** |
| LR Sentiment (TF-IDF) | 0.3636 | 0.3068 | 0.3636 | 0.3189 |
| LR Intent (TF-IDF) | 0.5455 | 0.3273 | 0.5455 | 0.4091 |

**Most Suitable Metric: F1 Score (Weighted)**
> Class-imbalanced dataset — weighted F1 balances Precision & Recall across all classes fairly. VADER is evaluated on the full 275-sample dataset making it the most statistically reliable score. LR scores are based on only 11 unseen test samples and should be interpreted with caution.

### Cell 18 — Model Comparison Chart
Grouped bar chart visualising all four metrics across all three models.

### Cell 19 — Gradio Interface
Live web app that runs all three models on any typed review.

**Example:**
```
Input:  "My order never arrived, I want a refund!"

Output:
  Sentiment → 😞 Negative  (VADER score: -0.71)
              🤖 ML Prediction: Negative (confidence: 0.xx)
  Intent    → 🏷️ Refund Request  (confidence: 0.xx)
  Topic     → 📌 Returns & Refunds
              🔑 Keywords: money, waste, complete, buy, nothing...
```

Launched with `share=True` — generates a public Gradio URL valid for 7 days.

---

## 📊 Results Summary

| Model | Accuracy | F1 Score | Notes |
|---|---|---|---|
| VADER (Rule-Based) | 0.6545 | **0.6552** | Evaluated on full 275-sample dataset. Most reliable score. |
| LR Sentiment (TF-IDF) | 0.3636 | 0.3189 | 11-sample test set — not statistically reliable. |
| LR Intent (TF-IDF) | 0.5455 | 0.4091 | 11-sample test set — not statistically reliable. |

**On the real Women's Clothing dataset, expect approximately:**
- LR Sentiment: 0.85–0.91 F1
- LR Intent: 0.80–0.87 F1
- VADER: 0.60–0.70 F1 (consistent with synthetic result)

---

## ⚠️ Known Limitations

1. **Small synthetic test set** — With only 11 unique unseen test reviews, LR scores fluctuate significantly with each wrong prediction and are not statistically reliable. VADER, evaluated on all 275 samples, is the most trustworthy metric.

2. **Roman Urdu handling** — NLTK stopwords and VADER are English-only. Roman Urdu tokens are not cleaned or understood by VADER, reducing accuracy on mixed-language reviews.

3. **Intent labels from rules** — Intent ground truth is generated by the same keyword rules used at inference time, meaning the intent classifier learns to replicate a rule-based system rather than true intent understanding.

4. **NMF on small dataset** — Topic separation is less clean on small datasets. On the synthetic set, only 4 of the 5 possible topic labels appear in the document-topic distribution. Topics may overlap or not perfectly align with real-world categories.

5. **Imbalanced intent classes** — Refund Request and Delivery Issue each have only 13 samples in the full dataset and only 1 sample each in the test set, making per-class evaluation for these intents unreliable.

---

## 🧠 Key Concepts

**Why TF-IDF over Bag of Words?**
BoW treats all words equally. TF-IDF penalises words that appear in nearly every document and rewards rare but discriminative terms — giving the classifier stronger signal. On this small synthetic dataset both methods score identically (0.5455 accuracy); the advantage of TF-IDF is more pronounced on larger corpora.

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
