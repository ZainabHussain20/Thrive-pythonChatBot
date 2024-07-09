import json
import nltk
import random
import pymongo
from http.server import BaseHTTPRequestHandler, HTTPServer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('wordnet')

# Initialize NLTK lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents from JSON file
with open('intents.json') as file:
    intents = json.load(file)

# Prepare training data
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

# Function to get a random response from the intent
def get_response(intent):
    for intent_data in intents['intents']:
        if intent_data['tag'] == intent:
            response_text = random.choice(intent_data['responses'])
            response_buttons = intent_data.get('buttons', [])
            return response_text, response_buttons
    return "I'm not sure how to respond to that.", []

# MongoDB client setup using environment variable
mongo_uri = os.getenv('MONGO_URI')
client = pymongo.MongoClient(mongo_uri)
db = client.get_default_database()
programs_collection = db['programs']

selected_program = None  # Variable to store the selected program

class ChatbotHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        global selected_program

        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        json_data = json.loads(post_data.decode('utf-8'))
        message = json_data.get('message', None)

        if not message:
            response = {"text": "Invalid input. Please provide a message.", "buttons": []}
        else:
            # Predict intent
            intent = predict_class(message)
            print(f"Predicted intent: {intent}")  # Debug statement
            response_text, response_buttons = get_response(intent)

            # Handle specific intents and include buttons
            if intent == "program_list":
                print("Retrieving programs from database...")  # Debug statement
                programs = programs_collection.find({}, {"_id": 0, "name": 1})
                program_names = [program["name"] for program in programs]
                print(f"Programs retrieved: {program_names}")  # Debug statement
                response = {
                    "text": "Here are our programs:",
                    "buttons": program_names
                }
            elif intent == "program_details" or message in [program["name"] for program in programs_collection.find({}, {"_id": 0, "name": 1})]:
                if selected_program is None:
                    selected_program = message
                response = {
                    "text": "What do you want to know about this program?",
                    "buttons": ["Time", "Start date", "End date", "Description", "Gender", "Location", "Reviews"]
                }
            elif message in ["Time", "Start date", "End date", "Description", "Gender", "Location", "Reviews"] and selected_program:
                program = programs_collection.find_one({"name": selected_program})
                if message == "Time":
                    response_text = f"The program time is {', '.join(program['time'])}."
                elif message == "Start date":
                    response_text = f"The program starts on {program['start']}."
                elif message == "End date":
                    response_text = f"The program ends on {program['end']}."
                elif message == "Description":
                    response_text = f"Description: {program['description']}"
                elif message == "Gender":
                    response_text = f"Gender: {program['gender']}"
                elif message == "Location":
                    response_text = f"Location: {program['location']}"
                elif message == "Reviews":
                    response_text = f"Reviews: {program['reviews']}"
                
                response = {
                    "text": response_text,
                    "buttons": ["Time", "Start date", "End date", "Description", "Gender", "Location", "Reviews"]
                }
            elif intent == "yes_response" or message.lower() in ["yes", "sure", "okay"]:
                selected_program = None
                response = {
                    "text": "Going back to the main menu.",
                    "buttons": ["About us", "Program list", "Contact"]
                }
            else:
                response = {
                    "text": response_text,
                    "buttons": response_buttons
                }

        # Send response
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'response': response}).encode('utf-8'))

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
