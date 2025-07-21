# model_training.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import pickle

# Load the resume dataset
df = pd.read_csv("resume_data.csv")

# Extract features and labels
X = df['text']
y = df['label']

# Create ML pipeline
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model
model.fit(X, y)

# Save the trained model
with open("resume_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as resume_model.pkl")
