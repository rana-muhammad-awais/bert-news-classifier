import streamlit as st
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification

# AG News classes
class_names = ["World", "Sports", "Business", "Sci/Tech"]

st.set_page_config(page_title="News Topic Classifier", page_icon="📰", layout="centered")

@st.cache_resource
def load_model_and_tokenizer():
 try:
 model_path = "./saved_bert_model"
 tokenizer = BertTokenizer.from_pretrained(model_path)
 model = BertForSequenceClassification.from_pretrained(model_path)
 model.eval()
 return model, tokenizer
 except Exception as e:
 st.error("Model not found. Please run the `task1_bert_classifier.py` script first to fine-tune and save the model.")
 return None, None

st.title("📰 News Topic Classifier")
st.markdown("This app uses a fine-tuned BERT model to classify news headlines into four categories: **World, Sports, Business, Sci/Tech**.")

model, tokenizer = load_model_and_tokenizer()

user_input = st.text_area("Enter a news headline or short article snippet here:", height=150, placeholder="e.g., Apple announces new AI-powered iPhone features at WWDC...")

if st.button("Classify Text"):
 if user_input.strip() == "":
 st.warning("Please enter some text to classify.")
 elif model is not None and tokenizer is not None:
 with st.spinner("Analyzing text..."):
 # Preprocess the input
 inputs = tokenizer(user_input, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
 
 # Perform inference
 with torch.no_grad():
 outputs = model(**inputs)
 logits = outputs.logits
 probabilities = torch.nn.functional.softmax(logits, dim=-1).squeeze().tolist()
 predicted_class_id = np.argmax(probabilities)
 
 # Display results
 predicted_label = class_names[predicted_class_id]
 st.success(f"**Predicted Category:** {predicted_label}")
 
 st.markdown("### Confidence Scores")
 for i, score in enumerate(probabilities):
 st.progress(score, text=f"{class_names[i]}: {score:.2%}")
 else:
 st.error("Model could not be loaded. Ensure the model directory exists.")

st.markdown("---")
st.markdown("Built with ❤️ using Hugging Face Transformers and Streamlit. Task 1 for internship.")
