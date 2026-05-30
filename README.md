## 📓 View Full Notebook
👉 [Click here to view the complete notebook](https://colab.research.google.com/drive/13Bd7saLEyucSSqECf0lIhfZpDQSyyYR0?usp=sharing)


# 🛍️ Customer Reviews Intelligence System
### NLP & Machine Learning Pipeline for E-Commerce Feedback Analysis

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-NLP-4CAF50?style=for-the-badge)
![Gradio](https://img.shields.io/badge/Gradio-Interface-FF7C00?style=for-the-badge)

> **Course:** AI4001 – Fundamentals of Natural Language Processing  
> **Institution:** NUCES Chiniot-Faisalabad Campus  
> **Department:** Artificial Intelligence and Data Science

---

## 📌 Project Overview

An end-to-end NLP system that automatically analyzes customer reviews from an e-commerce platform. The system handles mixed English and Roman Urdu text, classifies sentiment and customer intent, and discovers hidden topics — all powered by classical NLP techniques and machine learning.

---

## 🎯 Key Features

| Feature | Method Used |
|---|---|
| 🧹 Text Preprocessing | Tokenization, Lemmatization, Stopword Removal |
| 🔢 Feature Extraction | Bag of Words + TF-IDF |
| 😊 Sentiment Analysis | VADER (Rule-Based) + Logistic Regression (ML) |
| 🏷️ Intent Classification | Logistic Regression — 4 intent classes |
| 📚 Topic Modelling | NMF (Non-negative Matrix Factorization) |
| 📊 Model Evaluation | Precision, Recall, F1 Score, Confusion Matrix |
| 🖥️ Live Interface | Gradio Web App |

---

```

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

### 3. (Optional) Download Real Dataset
Place one of the following datasets in the project folder:

- [Women's E-Commerce Clothing Reviews — Kaggle](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)
- [E-Commerce Reviews Sentiment — HuggingFace](https://huggingface.co/datasets/mmehul/ecommerce-reviews-sentiment)
- [Amazon Fine Food Reviews — Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)

> If no dataset file is found, the notebook auto-generates a synthetic dataset including English and Roman Urdu reviews.

### 4. Run the Notebook
```bash
jupyter notebook LT1_24F0001.ipynb
```
Run all cells top to bottom. The last cell launches the Gradio interface.

---

## 📓 Notebook Walkthrough

### Cell 1 — Library Installation
Automatically installs all required packages using `subprocess`.

### Cell 2 — Imports & NLTK Downloads
Loads all libraries and downloads NLTK sub-packages: `punkt`, `stopwords`, `wordnet`.

### Cell 3 — Dataset Loading
Loads the CSV dataset. Falls back to a synthetic dataset with positive, negative, neutral, and Roman Urdu reviews if no file is found. Creates a `sentiment_label` column from ratings.

### Cell 4 — Text Preprocessing Pipeline
Cleans raw text through 6 steps:
```
Raw Text  →  Lowercase  →  Remove URLs  →  Remove Punctuation
         →  Tokenize  →  Remove Stopwords  →  Lemmatize  →  Clean Text
```

### Cell 5 — Preprocessing Visualisation
Histograms comparing review word counts before and after preprocessing.

### Cell 6 — Feature Extraction: BoW & TF-IDF
Converts cleaned text to numerical matrices using both methods. Shows vocabulary size and top frequent tokens.

### Cell 7 — BoW vs TF-IDF Comparison
Trains Naive Bayes on both feature sets and compares Accuracy and F1. TF-IDF consistently outperforms BoW by weighting rare but meaningful words higher.

### Cell 8 — VADER Rule-Based Sentiment
Applies the VADER lexicon to score each review. No training needed — works on compound score thresholds.

### Cell 9 — VADER Confusion Matrix
Heatmap showing per-class prediction accuracy for VADER.

### Cell 10 — Logistic Regression Sentiment
Trains a supervised ML model on TF-IDF features. Learns word-to-sentiment patterns from labelled data.

### Cell 11 — LR Confusion Matrix
Heatmap for the ML model — compare directly with Cell 9 to see improvement.

### Cell 12 — Intent Label Assignment
Labels each review into one of 4 intents using keyword rules:

| Intent | Keywords Detected |
|---|---|
| Refund Request | refund, money back, paisa wapas |
| Delivery Issue | delivery, shipping, late, kab ayega |
| Complaint | broken, damaged, ghatia, worst |
| General Query | everything else |

### Cell 13 — Intent Classifier
Trains Logistic Regression to predict intent. Prints full classification report per class.

### Cell 14 — Intent Confusion Matrix
4×4 heatmap showing intent prediction accuracy.

### Cell 15 — NMF Topic Modelling
Unsupervised discovery of 5 hidden topics across all reviews:

| Topic | Theme | Sample Keywords |
|---|---|---|
| 0 | Product Quality | quality, material, fabric |
| 1 | Sizing & Fit | size, fit, small, large |
| 2 | Delivery & Shipping | deliver, arrive, late |
| 3 | Customer Service | support, service, help |
| 4 | Returns & Refunds | return, refund, exchange |

### Cell 16 — Topic Distribution Chart
Bar chart showing which topics dominate customer feedback.

### Cell 17 — Full Evaluation Table
Side-by-side comparison of all three models across Accuracy, Precision, Recall, and F1 Score.

### Cell 18 — Model Comparison Chart
Grouped bar chart visualising all metrics for all models together.

### Cell 19 — Gradio Interface
Live web app — type any review and get instant analysis:

```
Input:  "My order never arrived, I want a refund!"

Output:
  Sentiment → 😞 Negative  (VADER: -0.71)
              🤖 ML Prediction: Negative (confidence: 0.93)
  Intent    → 🏷️ Refund Request (confidence: 0.88)
  Topic     → 📌 Returns & Refunds
              🔑 Keywords: refund, return, money, order...
```

---

## 📊 Results Summary

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| VADER (Rule-Based) | ~0.72 | ~0.70 | ~0.72 | ~0.71 |
| Logistic Regression (Sentiment) | ~0.89 | ~0.88 | ~0.89 | ~0.88 |
| Logistic Regression (Intent) | ~0.85 | ~0.84 | ~0.85 | ~0.84 |

> **Best Metric:** F1 Score — chosen because the dataset is class-imbalanced (more positive reviews than complaints), making accuracy alone misleading. F1 balances Precision and Recall fairly.

---

## 🖥️ Gradio Interface Demo

The last cell launches an interactive web app:

```bash
# Launches automatically when Cell 19 runs
# Public URL generated with share=True (valid 72 hours)
Running on public URL: https://xxxx.gradio.live
```

**Interface inputs/outputs:**
- 📝 Input: Customer review text (English or Roman Urdu)
- 😊 Output 1: Sentiment (rule-based + ML with confidence)
- 🏷️ Output 2: Intent class with confidence score
- 📌 Output 3: Dominant topic + top keywords

---

## 🧠 Concepts Explained

**Why TF-IDF over Bag of Words?**
BoW treats all words equally. TF-IDF penalises words that appear in every document (like "the", "product") and rewards words unique to specific reviews — giving the model more signal to learn from.

**Why Logistic Regression?**
Fast to train, highly interpretable, and consistently strong on text classification tasks. The coefficients directly tell you which words push a prediction toward each class.

**Why F1 Score as the primary metric?**
Customer review datasets are naturally imbalanced — there are always more positive reviews than complaints. Accuracy would be misleadingly high even if the model ignores the minority class. F1 balances precision and recall, giving a fair evaluation for all classes.

**What is NMF?**
Non-negative Matrix Factorization is an unsupervised technique that factorizes the TF-IDF matrix into topic-word and document-topic components. Unlike LDA, it produces parts-based representations and typically gives cleaner, more interpretable topics.

---

