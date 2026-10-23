# AIML_2_SpamClassifier_byte

## Project Overview
An NLP classification pipeline built to distinguish between legitimate messages (ham) and spam using TF-IDF feature extraction and a Multinomial Naive Bayes classifier.

## Performance Metrics
- **Accuracy:** ~98.2%
- **Precision:** ~97.5%
- **Recall:** ~94.1%
- **F1-Score:** ~95.8%

## Artifacts Included
- `task2_spam_classifier.py` / Jupyter Notebook
- `spam_classifier_model.pkl` & `tfidf_vectorizer.pkl`
- `confusion_matrix_task2.png`

## How to Run Inference
1. Install dependencies: `pip install scikit-learn pandas numpy matplotlib seaborn joblib`
2. Run the script: `python task2_spam_classifier.py`