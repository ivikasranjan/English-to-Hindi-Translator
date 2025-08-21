# 📘 English → Hindi Translator using Hugging Face NLP
# 📌 Project Overview

This project implements an English-to-Hindi Neural Machine Translation (NMT) system using Hugging Face Transformers. It leverages pre-trained models such as Helsinki-NLP/opus-mt-en-hi to translate English text into natural, fluent Hindi.

# 🎯 Objectives

🌍 Build an English → Hindi Translator using state-of-the-art NLP models

⚡ Provide fast and accurate translations

🎛️ Make the system ready for integration with Streamlit UI or Flask API

🛠️ Technologies & Tools Used

🐍 Python

🤗 Hugging Face Transformers

🔤 Pre-trained model: Helsinki-NLP/opus-mt-en-hi

📊 Pandas, Numpy (optional for dataset handling)

🚀 Streamlit (for deployment UI)

# ⚙️ Installation & Setup
# 1️⃣ Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # for Linux/Mac
venv\Scripts\activate     # for Windows

# 2️⃣ Install required libraries
pip install transformers torch sentencepiece streamlit

💻 Code Implementation
from transformers import MarianMTModel, MarianTokenizer

# Load pre-trained English → Hindi model
model_name = "Helsinki-NLP/opus-mt-en-hi"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate_en_to_hi(text):
    # Encode text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    # Generate translation
    translated = model.generate(**inputs)
    # Decode output
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# 🔎 Example
english_text = "Hello, how are you?"
hindi_translation = translate_en_to_hi(english_text)

print(f"English: {english_text}")
print(f"Hindi: {hindi_translation}")


✅ Output:

English: Hello, how are you?
Hindi: नमस्ते, आप कैसे हैं?

🚀 Run as Web App (Optional with Streamlit)
# save as app.py
import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

# Load model
model_name = "Helsinki-NLP/opus-mt-en-hi"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate_en_to_hi(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

st.title("🌍 English → Hindi Translator")
user_input = st.text_area("✍️ Enter English text:")

if st.button("Translate"):
    if user_input.strip() != "":
        hindi_text = translate_en_to_hi(user_input)
        st.success(f"✅ Hindi Translation: {hindi_text}")
    else:
        st.warning("⚠️ Please enter some text!")


Run the app:

streamlit run app.py

✅ Results

🔥 Accurate Hindi translations for general sentences

⚡ Works in real-time with Streamlit UI

🛠️ Easy to extend with Flask/Django APIs

🚀 Future Enhancements

📡 Deploy as an API for mobile/web apps

🧠 Add custom fine-tuning on domain-specific datasets

🎙️ Add speech-to-text and text-to-speech features

👨‍💻 Author

Vikas Ranjan
📧 ivikasranjan@gmail.com

🔗 LinkedIn | 🔗 GitHub

📌 Tags

#NLP #MachineTranslation #HuggingFace #EnglishToHindi #Streamlit #Python
