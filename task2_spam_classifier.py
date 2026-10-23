import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Load dataset (Using UCI SMS Spam Collection dataset publicly available)
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep="\t", header=None, names=["label", "message"])

# Preprocessing: Map labels to binary (ham: 0, spam: 1)
df["label_num"] = df.label.map({"ham": 0, "spam": 1})

X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label_num"], test_size=0.2, random_state=42
)

# Vectorization using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Classifier
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluation
y_pred = model.predict(X_test_vec)
y_prob = model.predict_proba(X_test_vec)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Spam Classifier - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix_task2.png")
plt.close()

# Save Model and Vectorizer
joblib.dump(model, "spam_classifier_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# Generate 10 Sample Predictions with Confidence Scores
sample_indices = np.random.choice(X_test.index, 10, replace=False)
samples = df.loc[sample_indices]
sample_vec = vectorizer.transform(samples["message"])
sample_preds = model.predict(sample_vec)
sample_probs = model.predict_proba(sample_vec)

print("\n--- 10 Sample Predictions ---")
for idx, row in samples.reset_index(drop=True).iterrows():
  pred_label = "spam" if sample_preds[idx] == 1 else "ham"
  confidence = np.max(sample_probs[idx])
  print(
      f"Message: {row['message'][:50]}... | Actual: {row['label']} | Predicted:"
      f" {pred_label} (Conf: {confidence:.2f})"
  )