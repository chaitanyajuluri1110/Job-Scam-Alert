import streamlit as st
import joblib
import os
import pandas as pd
import spacy
import string
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the models
rfc = joblib.load('random_forest_model.pkl')
svm_model = joblib.load('svm_model.pkl')

# Initialize Spacy
nlp = spacy.load("en_core_web_sm")
stop_words = spacy.lang.en.stop_words.STOP_WORDS
punctuations = string.punctuation

# Define the tokenizer and text cleaner
def spacy_tokenizer(sentence):
    parser = spacy.lang.en.English()
    mytokens = parser(sentence)
    mytokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens]
    mytokens = [word for word in mytokens if word not in stop_words and word not in punctuations]
    return mytokens

def clean_text(text):
    return text.strip().lower()

# Preprocessing function
def preprocess_input(data):
    data = data.apply(clean_text)
    vectorizer = TfidfVectorizer(max_features=100)
    vectorized_data = vectorizer.fit_transform(data)
    return vectorized_data

# Define the main function for Streamlit
def main():
    st.title("Fake Job Posting Prediction")

    # Input fields
    title = st.text_input("Job Title")
    description = st.text_area("Job Description")
    company_profile = st.text_area("Company Profile")
    requirements = st.text_area("Requirements")
    benefits = st.text_area("Benefits")

    # Select model
    model_choice = st.selectbox("Choose Model", ["Random Forest"])

    # Predict button
    if st.button("Predict"):
        input_data = {
            'title': title,
            'description': description,
            'company_profile': company_profile,
            'requirements': requirements,
            'benefits': benefits
        }
        input_df = pd.DataFrame([input_data])
        input_df['text'] = input_df['title'] + ' ' + input_df['company_profile'] + ' ' + input_df['description'] + ' ' + input_df['requirements'] + ' ' + input_df['benefits']
        processed_input = preprocess_input(input_df['text'])

        # Make a prediction
        if model_choice == "Random Forest":
            prediction = rfc.predict(processed_input)
        else:
            prediction = svm_model.predict(processed_input)

        # Display the result
        if prediction[0] == 1:
            st.write("This job posting is predicted to be **Fake**.")
        else:
            st.write("This job posting is predicted to be **Real**.")

if __name__ == '__main__':
    main()
