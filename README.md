# Document-GPT
I would like to create a AI powered repository from my hundreds of CSV and Excel files containing questionnaire data. I want to develop a chatbot that, based on user inputs, can retrieve relevant questions from these files. Additionally, I want the chatbot's retrieved results to be written to a my database or Google Sheets.

I previously used Dante AI for this purpose, but I’m currently exploring alternatives.
===================
To create an AI-powered repository for handling your CSV and Excel questionnaire data, and to build a chatbot that retrieves relevant questions based on user inputs, you can follow a multi-step approach. This involves:

    Extracting data from CSV and Excel files.
    Processing and indexing the data for efficient retrieval.
    Building a chatbot interface to interact with users.
    Storing the chatbot's responses in a database or Google Sheets.

Here’s a breakdown of how you can approach this problem using Python, with libraries such as Pandas, OpenAI, Google Sheets API, and Flask for web development.
Steps for Implementation:

    Install Necessary Libraries: You’ll need libraries like Pandas (for reading CSV/Excel), Flask (for building a chatbot interface), Google API client (to access Google Sheets), and OpenAI (for NLP-based chatbot functionality).

pip install pandas openai flask google-api-python-client google-auth-httplib2 google-auth-oauthlib

    Reading CSV/Excel Files: We will use Pandas to read CSV and Excel files and prepare the data.

import pandas as pd

def load_data(file_path):
    # Read CSV or Excel file depending on the file extension
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    else:
        raise ValueError("File must be CSV or Excel format")

    Preprocessing and Indexing the Data: You’ll want to preprocess and index the questions in a way that the chatbot can efficiently retrieve relevant data based on user queries. One simple method is to use TF-IDF (Term Frequency-Inverse Document Frequency) to vectorize the questions.

from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_and_index_data(data_frame):
    questions = data_frame['Question'].tolist()  # Assuming 'Question' is a column
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(questions)
    return tfidf_matrix, vectorizer

    Building a Simple Chatbot using OpenAI: You can use OpenAI’s GPT (or any other LLM) to handle the chatbot interface. The idea is to use the user’s input to retrieve relevant questions from the dataframe based on similarity scores and then generate meaningful responses.

import openai

openai.api_key = 'your_openai_api_key'

def generate_response(user_query, tfidf_matrix, vectorizer, data_frame):
    # Vectorize the user query
    user_query_tfidf = vectorizer.transform([user_query])
    
    # Calculate cosine similarity between the user query and the indexed questions
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_scores = cosine_similarity(user_query_tfidf, tfidf_matrix)
    
    # Get the index of the most relevant question
    most_similar_index = similarity_scores.argmax()
    question = data_frame.iloc[most_similar_index]['Question']
    
    # Send the question and response to OpenAI for generating a chatbot reply
    prompt = f"User asked: {user_query}\nChatbot, please respond to the user based on this question: {question}"
    response = openai.Completion.create(
        model="text-davinci-003", 
        prompt=prompt, 
        max_tokens=150
    )
    
    return response.choices[0].text.strip()

    Storing the Data in Google Sheets: To store the chatbot responses in Google Sheets, you can use the Google Sheets API. Below is an example of how to authenticate and write to Google Sheets.

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

# Set up Google Sheets API credentials
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
SERVICE_ACCOUNT_FILE = 'path/to/your/service_account.json'

def authenticate_google_sheets():
    creds = None
    creds = Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    service = build('sheets', 'v4', credentials=creds)
    return service

def append_to_google_sheet(sheet_id, values):
    service = authenticate_google_sheets()
    sheet = service.spreadsheets()
    request = sheet.values().append(
        spreadsheetId=sheet_id, range="Sheet1!A1", valueInputOption="RAW", body={"values": values})
    response = request.execute()
    return response

    Building the Flask API for Chatbot: The Flask web framework can be used to expose the chatbot as a web service. The following code sets up a simple API to interact with the chatbot.

from flask import Flask, request, jsonify

app = Flask(__name__)

# Load your data (CSV/Excel files)
file_path = 'path/to/your/csv_or_excel_file.csv'
data_frame = load_data(file_path)
tfidf_matrix, vectorizer = preprocess_and_index_data(data_frame)

@app.route('/chat', methods=['POST'])
def chat():
    user_query = request.json.get('user_query')
    if not user_query:
        return jsonify({"error": "No query provided"}), 400
    
    # Get the chatbot response
    response = generate_response(user_query, tfidf_matrix, vectorizer, data_frame)
    
    # Store response in Google Sheets (optional)
    sheet_id = 'your_google_sheet_id'
    values = [[user_query, response]]
    append_to_google_sheet(sheet_id, values)
    
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)

Example Request to the Chatbot API:

You can interact with the chatbot via a simple HTTP POST request.

curl -X POST -H "Content-Type: application/json" -d '{"user_query": "What is your return policy?"}' http://127.0.0.1:5000/chat

Conclusion:

This Python code snippet sets up a simple AI-powered repository that allows you to analyze and interact with your questionnaire data stored in CSV or Excel format. The chatbot can retrieve relevant questions based on user inputs, and you can easily store these interactions in a database or Google Sheets for future analysis.

By combining data preprocessing, machine learning techniques like TF-IDF, and NLP (via OpenAI's GPT model), you can create a highly responsive and intelligent chatbot that streamlines your questionnaire data analysis.
