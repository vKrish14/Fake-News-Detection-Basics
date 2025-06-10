# Fake-News-Detection-Basics
# 1. Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 2. Upload the dataset
from google.colab import files
uploaded = files.upload()

# 3. Load the dataset (replace 'news.csv' with your file name if different)
df = pd.read_csv('news.csv')

# 4. Display the shape and first few rows
print("Dataset shape:", df.shape)
df.head()

# 5. Extract features and labels
X = df['text']    # News article text
y = df['label']   # 'FAKE' or 'REAL'

# 6. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

# 7. Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# 8. Fit and transform the vectorizer on the train set, transform on the test set
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

# 9. Initialize and train the PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# 10. Predict on the test set
y_pred = pac.predict(tfidf_test)

# 11. Evaluate the model
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(score*100,2)}%')

# 12. Display the confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
print("Confusion Matrix:")
print(cm)
