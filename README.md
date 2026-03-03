# Twitter Sentiment Analysis using BiLSTM + FastAPI + Streamlit

## 📌 Project Overview
This project implements a production-style Twitter Sentiment Analysis system using a Bidirectional LSTM deep learning model.

The system follows a client-server architecture where:

Streamlit (Frontend UI)  
→ communicates with  
FastAPI (Backend Inference API)  
→ which loads and runs  
BiLSTM Model  

The backend must be active for the frontend to function.

---

## 🏗️ System Architecture

User Input  
→ Streamlit Interface  
→ HTTP Request to FastAPI  
→ Model Prediction  
→ JSON Response  
→ Displayed on UI  

The API can also be tested independently using FastAPI Swagger documentation.

---

## 🧠 Model Architecture

Embedding Layer  
→ Bidirectional LSTM  
→ Dense Layer  
→ Softmax Output Layer  

The Bidirectional LSTM processes text in both forward and backward directions to capture contextual dependencies.

---

## ⚙️ Tech Stack

### Machine Learning
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy

### Backend
- FastAPI
- Uvicorn

### Frontend
- Streamlit

---

## 📂 Project Structure

```
mini_project_3/
│
├── app.py              # FastAPI backend (API endpoints)
├── streamlit_app.py    # Streamlit frontend
├── model.py            # BiLSTM architecture
├── train.py            # Model training pipeline
├── preprocessing.py    # Text cleaning
├── tokenizer.py        # Tokenization utilities
├── config.py           # Configuration parameters
└── save_load.py        # Model persistence utilities
```

---

## 🚀 How to Run

### 1️⃣ Install dependencies
```
pip install -r requirements.txt
```

### 2️⃣ Start FastAPI backend
```
uvicorn app:app --reload
```

Access Swagger API documentation at:
```
http://127.0.0.1:8000/docs
```

### 3️⃣ Start Streamlit frontend
```
streamlit run streamlit_app.py
```

Note: FastAPI must be running before launching the Streamlit app.

---

## 🎯 Key Highlights
- Built an API-based ML inference system
- Designed modular client-server architecture
- Integrated FastAPI backend with Streamlit frontend
- Enabled independent API testing via Swagger UI
- Structured project using production-style separation of concerns

---

## 📈 Future Improvements
- Docker containerization
- Cloud deployment
- Load balancing for API scalability
- Authentication layer for secure API access

---

## 👩‍💻 Author
Neha Nazreena Anwer  
Electronics and Communication Engineering  
Aspiring AI/ML Engineer