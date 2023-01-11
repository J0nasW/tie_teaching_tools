import pdfplumber
import pandas as pd
import nltk
import re, string
from nltk.tokenize import RegexpTokenizer
nltk.download('punkt')

import streamlit as st

### STREAMLIT Init:

# SETTING PAGE CONFIG TO WIDE MODE AND ADDING A TITLE AND FAVICON
st.set_page_config(layout="centered", page_title="PDF Analyzer", page_icon="🏢")

# Start of the page

st.title("PDF Analyzer")

st.write("Analyze PDF files")

def extract_data(feed):
    text = ''
    with pdfplumber.open(feed) as pdf:
        for i, page in enumerate(pdf.pages):
            text = text+'\n'+str(page.extract_text())
    return text # build more code to return a dataframe 

def clean_text(text):
    # remove numbers
    text_nonum = re.sub(r'\d+', '', text)
    # remove punctuations and convert characters to lower case
    text_nopunct = "".join([char.lower() for char in text_nonum if char not in string.punctuation]) 
    # substitute multiple whitespace with single whitespace
    # Also, removes leading and trailing whitespaces
    text_no_doublespace = re.sub('\s+', ' ', text_nopunct).strip()
    return text_no_doublespace

df = pd.DataFrame(columns=["Filename", "Character Count", "Word Count", "Potential Error"])
tokenizer = RegexpTokenizer(r'\w+') # remove punctuation

uploaded_files = st.file_uploader("Choose a PDF file", accept_multiple_files=True, type="pdf")
for uploaded_file in uploaded_files:
    if uploaded_file is not None:
        text = extract_data(uploaded_file)
        #df.append([uploaded_file.name, len(text), len(text.split())])
        words = tokenizer.tokenize(clean_text(text))
        if len(text.split()) / len(text) < 0.05:
            p_error = True
        df.loc[len(df)] = [uploaded_file.name, len(text), len(text.split()), p_error]
        p_error = False

st.write(df)