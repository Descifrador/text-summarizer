# import streamlit and ntlk
import streamlit as st
import nltk

from t5_api import t5_summary
from tf_idf import run_summarization as tf_idf_summarization
from word_freq import run_summarization as run_summarization_wf

if __name__ == "__main__":
    st.title("Text Summarization")
    st.markdown("""
    #### Summary
    This app is a text summarization tool. It can summarize a text using the T5 model and the TF-IDF algorithm.
    """)
    st.markdown("""
    #### Word Frequency
    This app is a word frequency tool. It can calculate the frequency of words in a text.
    """)
    st.markdown("""
    #### T5
    This app is a text summarization tool. It can summarize a text using the T5 model.
    """)
    nltk.download('punkt')
    nltk.download('stopwords')

    text = st.text_area("Enter your text here")
    if st.button("Summarize"):
        if text:
            summary = t5_summary(text)
            st.subheader("Using T5")
            st.success(summary)

    if st.button("Word Frequency"):
        if text:
            freq = run_summarization_wf(text)
            st.subheader("Word Frequency")
            st.success(freq)

    if st.button("TF-IDF"):
        if text:
            summary = tf_idf_summarization(text)
            st.subheader("Using TF-IDF")
            st.success(summary)

    else:
        st.success("Please enter some text to summarize.")
