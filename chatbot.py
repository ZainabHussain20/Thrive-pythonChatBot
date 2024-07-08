import json
import nltk
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('wordnet')

# Initialize NLTK lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents from JSON file (intents.json should be structured correctly)
with open('intents.json') as file:
    intents = json.load(file)

# Prepare training data (assuming you've trained your model and have it ready)
sentences = []
labels = []
classes = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        sentences.append(' '.join([lemmatizer.lemmatize(w.lower()) for w in word_list]))
        labels.append(intent['tag'])
    if intent['tag'] not in classes:
        classes.append(intent['tag'])

# Initialize CountVectorizer and train logistic regression model
vectorizer = CountVectorizer(token_pattern=r'(?u)\b\w+\b')
X = vectorizer.fit_transform(sentences)
y = labels
model = LogisticRegression()
model.fit(X, y)

# Function to predict intent from user message
def predict_class(sentence):
    sentence = ' '.join([lemmatizer.lemmatize(w.lower()) for w in nltk.word_tokenize(sentence)])
    X_test = vectorizer.transform([sentence])
    prediction = model.predict(X_test)
    return prediction[0]

class ChatbotHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        json_data = json.loads(post_data.decode('utf-8'))
        message = json_data['message']

        # Predict intent
        intent = predict_class(message)

        # Send response
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'intent': intent}).encode('utf-8'))

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write("<html><body><h1>Welcome to the Chatbot Server</h1></body></html>".encode('utf-8'))

# Function to run HTTP server
def run_server(server_class=HTTPServer, handler_class=ChatbotHandler, port=5000):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting chatbot server on port {port}...')
    httpd.serve_forever()

# Entry point
if __name__ == '__main__':
    run_server()
