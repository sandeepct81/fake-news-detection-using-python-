import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set Kaggle credentials directly
KAGGLE_USERNAME = "put your username "
KAGGLE_KEY = " put your api key here"

# Set environment variables
os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
os.environ["KAGGLE_KEY"] = KAGGLE_KEY

# Download and prepare dataset
@st.cache_data
def download_dataset():
    dataset_path = "news.csv"

    if not os.path.exists(dataset_path):
        # Download dataset
        os.system("kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset --unzip")

        # Load Fake and Real CSVs
        fake_df = pd.read_csv("Fake.csv")
        true_df = pd.read_csv("True.csv")

        # Add labels
        fake_df["label"] = 1  # Fake = 1
        true_df["label"] = 0  # Real = 0

        # Combine both
        df = pd.concat([fake_df, true_df], ignore_index=True)
        df.to_csv(dataset_path, index=False)
    else:
        df = pd.read_csv(dataset_path)

    return df

# Train the model
@st.cache_resource
def train_model(df):
    X = df['text']
    y = df['label']

    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, vectorizer, accuracy

# Extract news content from a given URL
def fetch_news_content(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        content = ' '.join([para.get_text() for para in paragraphs])
        return content
    except:
        return None

# Predict whether the news is real or fake
def predict_news(url, model, vectorizer):
    content = fetch_news_content(url)
    if content:
        transformed_content = vectorizer.transform([content])
        prediction = model.predict(transformed_content)
        return prediction[0], content
    return None, None

# Streamlit GUI
st.title("📰 Fake News Detection App")
st.markdown("Enter a news link to check whether it is **Fake** or **Real**.")

news_url = st.text_input("Paste News URL here:")

if news_url:
    df = download_dataset()
    model, vectorizer, accuracy = train_model(df)

    prediction, content = predict_news(news_url, model, vectorizer)

    if content:
        st.subheader("News Content:")
        st.write(content[:1000] + "..." if len(content) > 1000 else content)

        st.subheader("Prediction:")
        if prediction == 1:
            st.error("🚨 The news is likely **Fake**!")
        else:
            st.success("✅ The news is likely **Real**!")

        st.write(f"**Model Accuracy:** {accuracy * 100:.2f}%")
    else:
        st.warning("Unable to fetch news content from the given URL. Please check the link and try again.")
