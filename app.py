import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set page config
st.set_page_config(page_title="Next Word Predictor", page_icon="🔮")

# Load the LSTM Model
model = load_model("next_word_lstm.h5")

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return "❓ (Not found in vocabulary)"

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>🔮 Next Word Predictor 🔮</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; font-size: 18px;'>Type a short phrase and let the LSTM model guess what comes next!</p>",
    unsafe_allow_html=True
)

# Input box
st.markdown("### 📝 Input:")
input_text = st.text_input("Enter a short sequence", placeholder="e.g., To be or not to")

# Predict button
if st.button("✨ Predict Next Word"):
    if input_text.strip() == "":
        st.warning("⚠️ Please enter a valid text input.")
    else:
        max_sequence_len = model.input_shape[1] + 1
        next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
        st.success(f"🔤 **Predicted Next Word:** `{next_word}`")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: grey;'>Built with 💡 using Streamlit & TensorFlow</div>",
    unsafe_allow_html=True
)
