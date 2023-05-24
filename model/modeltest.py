import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load the dataset
data = pd.read_csv('Datasets\labeled_data.csv')

# Split the dataset into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(data['tweet'], data['class'], test_size=0.2, random_state=42)

# Preprocess the text data using TF-IDF vectorizer
vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(train_data)
test_vectors = vectorizer.transform(test_data)

# Train a Random Forest Classifier
classifier = RandomForestClassifier()
classifier.fit(train_vectors, train_labels)

# Make predictions on the test set
predictions = classifier.predict(test_vectors)

# Evaluate the model
print(classification_report(test_labels, predictions))

# User input for prediction
user_tweet = input("Enter a tweet: ")
user_tweet_vector = vectorizer.transform([user_tweet])
prediction = classifier.predict(user_tweet_vector)
print(f"Prediction: {'Cyberbullying' if prediction == 1 else 'Non-cyberbullying'}")

joblib.dump(classifier, 'model.pkl')

