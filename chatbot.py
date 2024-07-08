import sys
import json
import random
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Load intents
with open('intents.json') as file:
    intents = json.load(file)

# Prepare training data
sentences = []
labels = []
classes = []
documents = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        sentences.append(' '.join([lemmatizer.lemmatize(w.lower()) for w in word_list]))
        labels.append(intent['tag'])
    if intent['tag'] not in classes:
        classes.append(intent['tag'])

vectorizer = CountVectorizer(token_pattern=r'(?u)\b\w+\b')
X = vectorizer.fit_transform(sentences)
y = labels

# Train model
model = LogisticRegression()
model.fit(X, y)

def predict_class(sentence):
    sentence = ' '.join([lemmatizer.lemmatize(w.lower()) for w in nltk.word_tokenize(sentence)])
    X_test = vectorizer.transform([sentence])
    prediction = model.predict(X_test)
    return prediction[0]

if __name__ == "__main__":
    user_message = sys.argv[1]
    intent = predict_class(user_message)
    print(intent)
