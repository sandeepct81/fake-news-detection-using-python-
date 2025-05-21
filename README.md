# fake-news-detection-using-python-
Detect fake news using Python and machine learning models trained on Kaggle datasets. Includes data preprocessing, model training, and evaluation.


# 📰 Fake News Detection App with Python, Streamlit & Kaggle API

A web-based application that detects whether a news article is **fake** or **real** using natural language processing and machine learning. The app uses data from Kaggle and provides a user-friendly interface using Streamlit.

---

## 🚀 Demo

![App Screenshot](https://user-images.githubusercontent.com/your-screenshot.png)  
👉 Try the app by pasting a news article URL into the input field.

---

## 📦 Features

- ✅ Download dataset from Kaggle via API
- 🧹 Preprocess news data (TF-IDF)
- 🤖 Train Logistic Regression model
- 🌐 Extract content from news URLs (using BeautifulSoup)
- 🧠 Predict if the news is real or fake
- 📊 Display model accuracy
- 🎨 Built with Streamlit for interactivity

---

## 🗂️ Project Structure
there is no need of any dataset to run the project just use kaggle api key. and it will automatically fetch the dataset.

fake-news-detector/
│
├── Fake.csv # (Kaggle dataset)
├── True.csv # (Kaggle dataset)
├── news.csv # Merged dataset used for training
├── app.py # Main Streamlit app
├── README.md # Project documentation
└── requirements.txt # Dependencies

