import streamlit as st
from transformers import pipeline

st.title("🧠 Hugging Face Text Summarizer")

# Load model only once
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# Input area
user_input = st.text_area("Paste the text you want to summarize:", 
                          "Artificial intelligence (AI) is a rapidly evolving field of computer science focused on creating systems that can perform tasks that typically require human intelligence. These tasks include understanding natural language, recognizing patterns, learning from experience, and making decisions. AI is being used in various applications such as healthcare, finance, transportation, and more...")

max_len = st.slider("Max Summary Length", min_value=30, max_value=200, value=100)

if st.button("Summarize"):
    with st.spinner("Summarizing..."):
        summary = summarizer(user_input, max_length=max_len, min_length=30, do_sample=False)[0]['summary_text']
        st.subheader("📝 Summary:")
        st.write(summary)
