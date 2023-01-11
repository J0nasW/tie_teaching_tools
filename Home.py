import pdfplumber
import pandas as pd

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

df = pd.DataFrame(columns=["Filename", "Character Count", "Word Count"])

uploaded_files = st.file_uploader("Choose a PDF file", accept_multiple_files=True, type="pdf")
for uploaded_file in uploaded_files:
    if uploaded_file is not None:
        text = extract_data(uploaded_file)
        #df.append([uploaded_file.name, len(text), len(text.split())])
        df.loc[len(df)] = [uploaded_file.name, len(text), len(text.split())]

st.write(df)