# Next Word Prediction using LSTM and GRU

This project is a simple NLP-based text generation tool that predicts the **next word** in a sentence using deep learning. It was trained on Shakespeare's *Hamlet*, providing context-aware predictions in classical English style.

## 🚀 Project Overview

The model takes an input sentence fragment and returns the most likely next word based on training data. It uses:

- **LSTM (Long Short-Term Memory)** layers for sequence modeling
- **GRU (Gated Recurrent Unit)** as an alternative architecture
- **Adam Optimizer** for efficient training
- **Shakespeare’s Hamlet** as the dataset for training

## 🧠 Techniques Used

- Tokenization of input text
- Sequence padding and preparation
- RNN-based model architectures: **LSTM** and **GRU**
- One-hot encoding for next word prediction
- Model serialization using `.h5` and tokenizer `.pickle`

## 📦 Files in This Repo

- `app.py` – Streamlit app to run the model in the browser
- `next_word_lstm.h5` – Trained LSTM model weights
- `tokenizer.pickle` – Fitted tokenizer to process input
- `utils.py` – Contains the prediction function
- `README.md` – You’re reading it!

## 🧪 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/JainHarsh00/nextWordPrediction.git
   cd nextWordPrediction
