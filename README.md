# Twitter Sentiment Analysis using Bidirectional LSTM

## 📌 Project Overview
This project implements a multi-class Twitter sentiment classification system using Natural Language Processing (NLP) techniques and a Bidirectional LSTM deep learning model.

The system performs end-to-end processing:
- Text preprocessing and cleaning
- Tokenization and sequence padding
- Model training and evaluation
- Deployment using Streamlit for real-time prediction

This project demonstrates practical implementation of sequence modeling for NLP tasks.

---

## 🧠 Model Architecture

Embedding Layer  
→ Bidirectional LSTM  
→ Dense Layer  
→ Softmax Output Layer  

The Bidirectional LSTM processes text sequences in both forward and backward directions, allowing better contextual understanding of sentence structure.

---

## 📊 Dataset
- Twitter sentiment dataset  
- Multi-class classification problem  
- Text data preprocessed using tokenization and padding  

(Note: Dataset files are not included in this repository.)

---

## ⚙️ Tech Stack
- Python  
- TensorFlow / Keras  
- Scikit-learn  
- Pandas  
- NumPy  
- Streamlit  

---

## 📂 Project Structure

```
mini_project_3/
│
├── preprocessing.py      # Text cleaning and preprocessing
├── model.py              # BiLSTM model architecture
├── train.py              # Model training script
├── streamlit_app.py      # Streamlit web interface
├── config.py             # Configuration parameters
├── data_loader.py        # Dataset loading utilities
├── tokenizer.py          # Tokenization logic
└── save_load.py          # Model save/load utilities
```

---

## 🚀 How to Run

### 1️⃣ Install dependencies
```
pip install -r requirements.txt
```

### 2️⃣ Train the model
```
python train.py
```

### 3️⃣ Launch the web application
```
streamlit run streamlit_app.py
```

---

## 🎯 Key Highlights
- Implemented sequence modeling using Bidirectional LSTM
- Designed modular and maintainable project structure
- Applied NLP preprocessing pipeline for text normalization
- Built an interactive web interface for real-time predictions
- Structured project following production-style organization

---

## 📈 Future Improvements
- Hyperparameter tuning
- Model optimization and regularization
- Deployment using cloud platforms
- Integration with live Twitter API

---

## 👩‍💻 Author
Neha Nazreena Anwer  
Electronics and Communication Engineering  
Aspiring AI/ML Engineer