import pandas as pd
import numpy as np
from datasets import load_dataset
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import requests
import time
import sys
import google.generativeai as genai
# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Google Gemini API configuration
GEMINI_API_URL = "https://api.google.com/gemini/v1/generateText"
API_KEY = "AIzaSyDDW21Sdx0yc_1-bgRAmZ3Vjbi35PzskMY"  # Replace with your actual API key

# Load the Google Generative AI library and configure it
genai.configure(api_key=API_KEY)

# Load the dataset from Hugging Face
try:
    dataset = load_dataset("opennyaiorg/InJudgements_dataset")
    data = pd.DataFrame(dataset['train'])
except Exception as e:
    print("Error loading dataset:", e)
    data = pd.DataFrame()  # Initialize an empty DataFrame if there's an issue

# Check if data is loaded
if data.empty:
    print("No cases found in the dataset.")
else:
    print("Dataset loaded successfully.\n")

    # Step 1: Preprocess Text Data with Loading Indicator
    stop_words = set(stopwords.words('english'))

    def preprocess_text(text):
        # Clean the text by removing non-word characters and stopwords
        text = re.sub(r'\W', ' ', text)  # Keep only alphanumeric characters
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        tokens = word_tokenize(text.lower())  # Tokenize text and convert to lowercase
        tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
        return ' '.join(tokens)

    # Simulate loading progress for preprocessing
    print("Step 1: Preprocessing text data...")
    for i in range(1, 101):
        time.sleep(0.01)  # Simulate work being done
        sys.stdout.write(f"\rPreprocessing text data... {i}% completed")
        sys.stdout.flush()

    # Apply preprocessing to the entire text (case details and judgment content)
    data['Processed_Text'] = data['Text'].apply(preprocess_text)
    print("\nPreprocessing completed.\n")

    # Step 2: Load Pretrained Summarization Model (T5)
    print("Step 2: Loading pretrained summarization model...")
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
    print("Pretrained summarization model loaded.\n")

    # Step 3: Generate a refined judgment using the T5 model and Gemini API
    def generate_custom_judgment_with_ai(text):
        # Preprocess and vectorize the input text
        processed_text = preprocess_text(text)
        # print(f"Case Details:\n{text}")  # Print the original case details

        # Format input for the T5 model with a prompt
        input_text = f"Generate custom judgment for the following case description (in 100 words): {text}"

        # Tokenize input and generate prediction
        # inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        # summary_ids = model.generate(inputs.input_ids, max_length=150, num_beams=5, early_stopping=True)

        # Decode and refine with Gemini API using the SDK
        # generated_judgment = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        try:
            geminimodel = genai.GenerativeModel("gemini-1.5-flash")
            refined_judgment = geminimodel.generate_content(input_text)   
            refined_judgment = refined_judgment.text
        except Exception as e:
            print(f"Error during Gemini API call: {e}")
            refined_judgment = text  # Fallback if API call fails

        return refined_judgment

    # Interactive Prediction Function for User Input
    def user_input_prediction():
        while True:
            user_text = "The case involves a dispute between two parties over the ownership of a plot of land in the city. The plaintiff claims that they inherited the land from their late father and has documentation to prove it. The defendant, however, argues that they purchased the land from a third party who claimed ownership. The plaintiff alleges that the sale was fraudulent, as the third party had no legal right to sell the land in the first place. The defendant insists they were unaware of any fraud and acted in good faith during the purchase. Both parties seek a ruling to clarify the rightful ownership of the land."
            if user_text.lower() == 'exit':
                print("Exiting...")
                break
            result = generate_custom_judgment_with_ai(user_text)
            print("\nGenerated Judgment:\n", result)

    # Run interactive prediction for user input
    user_input_prediction()
