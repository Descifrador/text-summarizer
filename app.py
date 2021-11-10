# import streamlit and ntlk
import streamlit as st
import nltk

from src.t5_api import t5_summary
from src.tf_idf import run_summarization as tf_idf_summarization
from src.word_freq import run_summarization as run_summarization_wf

if __name__ == "__main__":
    nltk.download('punkt')
    nltk.download('stopwords')
    st.title("Text Summarization")
    st.markdown("""
    #### Summary
    This app is a text summarization tool using the T5-Transformer model.
    """)

    text = st.text_area("Paste your text here")
    summarization_method = st.selectbox(
        "Select summarization method",
        ["T5-Transformer", "TF-IDF", "Word Frequency"])

    if st.button("Get Summary"):
        if summarization_method == "T5-Transformer":
            summary = t5_summary(text)
        elif summarization_method == "TF-IDF":
            summary = tf_idf_summarization(text)
        elif summarization_method == "Word Frequency":
            summary = run_summarization_wf(text)
        st.success(summary)
        st.markdown("---")
