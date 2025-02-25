import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import os

# Load dataset
data = pd.read_csv("./data/spam.csv")
X = data["sms"]
y = data["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model pipelineimport os
import pandas as pd
import mlflow
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Ensure model directory exists
os.makedirs("./model", exist_ok=True)

# Load dataset
data = pd.read_csv("./data/spam.csv")
print("@@@@@@@@")
X = data["sms"]
y = data["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model pipeline
model_pipeline = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB())
])

# Train model
with mlflow.start_run():
    model_pipeline.fit(X_train, y_train)
    preds = model_pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model_pipeline, "model")

# Save model
model_path = "./model/spam_classifier.pkl"
joblib.dump(model_pipeline, model_path)

print(f"Model saved successfully at {model_path}")

model_pipeline = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB())
])

# Train model
with mlflow.start_run():
    model_pipeline.fit(X_train, y_train)
    preds = model_pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model_pipeline, "model")

# Save model
joblib.dump(model_pipeline, "./model/spam_classifier.pkl")
