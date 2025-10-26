🧠 Sentiment Analysis App (Roman Urdu + English)

This project is a Streamlit-based Sentiment Analysis Web App that can analyze text sentiments in English and Roman Urdu.
If the user types a comment in Roman Urdu, it is automatically translated to English before predicting the sentiment using a trained ML model.

🚀 Features

✅ Detects text language (English / Roman Urdu / Urdu script)
✅ Automatically translates Roman Urdu / Urdu to English
✅ Predicts Positive, Negative, or Neutral sentiment
✅ Handles timeouts, handshake errors, and translation failures
✅ User-friendly Streamlit interface for real-time testing
✅ Supports retraining with new datasets (IMDB, custom CSVs, etc.)

🧩 Tech Stack
Component	Technology
Frontend	Streamlit
Language Detection	langdetect
Translation	deep-translator (GoogleTranslator)
Model	Logistic Regression / Naive Bayes (Scikit-learn)
Vectorization	TF-IDF
Backend	Python
Dataset (example)	IMDB Reviews Dataset
🛠️ Installation

Clone the Repository

git clone https://github.com/yourusername/sentiment-analysis-app.git
cd sentiment-analysis-app


Create a Virtual Environment (Optional but Recommended)

python -m venv env
env\Scripts\activate     # For Windows
source env/bin/activate  # For Linux/Mac


Install Dependencies

pip install -r requirements.txt


Or manually install key libraries:

pip install streamlit scikit-learn pandas numpy deep-translator langdetect joblib


(Optional) Train Your Model
If you want to retrain:

python train_model.py


This will generate:

sentiment_model.pkl
vectorizer.pkl
label_encoder.pkl

💻 Run the App

Start the Streamlit app:

streamlit run app.py


Then open the URL shown in your terminal (usually http://localhost:8501).
