import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

# Download NLTK data (if not already downloaded)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
# Load the dataset (replace 'spam.csv' with your dataset path if different)
try:
    df = pd.read_csv('spam.csv', encoding='latin-1')

    # Rename columns to match the expected structure
    df.rename(columns={'Category': 'label', 'Message': 'text'}, inplace=True)
    df = df[['label', 'text']] # Keep only relevant columns and ensure their order

    # --- ADD THIS CODE TO HANDLE THE INVALID LABEL ---
    # Find and remove rows where the label is not 'ham' or 'spam'
    invalid_label_count = df[~df['label'].isin(['ham', 'spam'])].shape[0]
    if invalid_label_count > 0:
        print(f"\nWarning: Found {invalid_label_count} rows with invalid labels. These rows will be removed.")
        df = df[df['label'].isin(['ham', 'spam'])].copy() # Filter and create a copy

except FileNotFoundError:
    print("Error: spam.csv not found. Please ensure the dataset is in the same directory or provide the correct path.")
    print("You can download a sample dataset from Kaggle, e.g., 'Spam Mails Dataset'.")
    # Create a dummy DataFrame for demonstration if file is not found
    data = {'v1': ['ham', 'spam', 'ham', 'spam', 'ham'],
            'v2': ['Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat?',
                   'URGENT! You have won a 1 week FREE membership in our $100000 Prize. TEXT FA to 80877 to claim now!',
                   'Nah I don\'t think he goes to usf, he lives around here though',
                   'Had your mobile 11 months or more? U R entitled to Update to the latest Nokia for FREE! Call now on 0800800000 to get on the list.',
                   'I\'m gonna be home soon and i don\'t want to talk about this stuff anymore tonight, k? I\'ve cried enough today.']}
    df = pd.DataFrame(data)
    # Rename columns to match the expected structure
    df.rename(columns={'v1': 'label', 'v2': 'text'}, inplace=True)
    df = df[['label', 'text']] # Keep only relevant columns

print("Dataset Head:")
print(df.head())

print("\nDataset Info:")
df.info()

print("\nLabel Distribution:")
print(df['label'].value_counts())

# Visualize label distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='label', data=df)
plt.title('Distribution of Spam vs. Ham Emails')
plt.xlabel('Email Type')
plt.ylabel('Count')
plt.show()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower() # Lowercasing
    text = text.translate(str.maketrans('', '', string.punctuation)) # Remove punctuation
    words = text.split() # Tokenize (simple split for demonstration)
    words = [stemmer.stem(word) for word in words if word not in stop_words] # Remove stopwords and stem
    return ' '.join(words)

df['processed_text'] = df['text'].apply(preprocess_text)

print("\nOriginal vs. Processed Text (first 5 examples):")
for i in range(5):
    print(f"Original: {df['text'].iloc[i]}")
    print(f"Processed: {df['processed_text'].iloc[i]}\n")
    # Split data into features (X) and target (y)
X = df['processed_text']
y = df['label']

# Convert labels to numerical format (0 for ham, 1 for spam)
y = y.map({'ham': 0, 'spam': 1})

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Limit features to top 5000

# Fit and transform the training data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Transform the test data
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print(f"\nShape of X_train_tfidf: {X_train_tfidf.shape}")
print(f"Shape of X_test_tfidf: {X_test_tfidf.shape}")
# Initialize Multinomial Naive Bayes classifier
model = MultinomialNB()

# Train the model
model.fit(X_train_tfidf, y_train)

print("\nModel training complete.")
# Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nModel Evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Display Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Display Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
def predict_spam(email_text):
    # Preprocess the new email
    processed_email = preprocess_text(email_text)
    # Transform using the fitted TF-IDF vectorizer
    email_tfidf = tfidf_vectorizer.transform([processed_email])
    # Predict
    prediction = model.predict(email_tfidf)
    # Get probability (optional)
    probability = model.predict_proba(email_tfidf)[0]

    if prediction[0] == 1:
        print(f"'{email_text}' is predicted as: SPAM (Probability of Spam: {probability[1]:.2f})")
    else:
        print(f"'{email_text}' is predicted as: HAM (Probability of Ham: {probability[0]:.2f})")

# Test with some example emails
print("\nTesting with new email examples:")
predict_spam("Congratulations! You've won a FREE iPhone! Click this link now to claim your prize.")
predict_spam("Hey, can we meet tomorrow to discuss the project? I've attached the latest report.")
predict_spam("Urgent! Your account has been compromised. Verify your details immediately via the link provided.")
predict_spam("Hi there, how are you doing? I hope everything is fine at your end.")
predict_spam("You are selected for a lucky draw. Send your bank details to claim the reward.")
