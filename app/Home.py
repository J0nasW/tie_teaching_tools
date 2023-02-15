####################################################################################
# Text Extractor
# by JW
#
# A simple python tool to extract text from websites, PDFs, and other file formats.
# 
# pages / Text_Extractor.py
####################################################################################

# IMPORT STATEMENTS ----------------------------------------------------------------
import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
import json
from text_extraction import (
    clean_text,
    read_text_from_file,
)

# Own Functions
from first_init import *

from model import GPT2PPL


### STREAMLIT Init:


init_application()
nltk.download('punkt')
df = pd.DataFrame(columns=["Filename", "Filesize", "Filetype", "Character Count", "Word Count","AI Generated (GPT)", "Perplexity", "PPL", "Potential Error"])
tokenizer = RegexpTokenizer(r'\w+') # remove punctuation
p_error = False
GPT_generated = False
model = GPT2PPL()

# SETTING PAGE CONFIG TO WIDE MODE AND ADDING A TITLE AND FAVICON
st.set_page_config(layout="wide", page_title="Text Extractor", page_icon=":paperclip:")

# Start of the page

st.title("📝 Text Extractor & Analyzer")

st.write("A simple python tool to extract and analyzte text from PDFs, MS Office Documents and other file formats.")
# st.info("Still in development")

text_extracted = st.session_state["text_extracted"]
url_text = st.session_state["url_text"]
cleaned_raw_text = st.session_state["cleaned_raw_text"]
uploaded_files_infos = st.session_state["uploaded_files_infos"]
url_infos = st.session_state["url_infos"]

with st.sidebar:
    st.markdown("## 🛠️ Settings")
    st.info("Here you can set the threshold for AI Text detection.", icon="ℹ️")
    st.write("A rule of thumb: PPL < 60 = Almost certainly AI generated, PPL < 80 = could be AI generated, PPL > 100 = Human generated")
    AI_treshold = st.slider("Threshold for Perplexity per Line", min_value=0, max_value=100, value=70, step=1, key="threshold")

with st.form("text_extractor_form"):
    text_input = st.text_area("Enter text here", height=100)
    st.markdown(
        "<h3 style='text-align: center; color: green;'>OR</h3>",
        unsafe_allow_html=True,
    )
    
    uploaded_file = st.file_uploader(
        "Upload a .txt, .pdf, .docx or .pptx file for summarization.", type=["txt", "pdf", "docx", "pptx", "md"], accept_multiple_files=True
    )
    check_gpt = st.checkbox("Include GPT-3 and ChatGPT Recognition (experimental)", value=st.session_state.check_gpt, key="gpt3")
    submit_text_extract = st.form_submit_button(label="Go!")    

if submit_text_extract:
    text_extracted = False
    st.session_state.text_extracted = text_extracted
    st.session_state.check_gpt = check_gpt
    with st.spinner("Extracting text..."):
        try:
            if uploaded_file:
                if uploaded_file is not None and uploaded_file:
                    uploaded_files_infos = []
                    st.session_state["uploaded_files_infos"] = []
                    for i in range(len(uploaded_file)):
                        file_text = read_text_from_file(uploaded_file[i])
                        cleaned_file_text = clean_text(file_text)
                        file_dict = {'file_name': uploaded_file[i].name, 'file_type': uploaded_file[i].type, 'file_size': uploaded_file[i].size, 'file_text': file_text, 'cleaned_file_text': cleaned_file_text}
                        uploaded_files_infos.append(file_dict)
                text_extracted = True
                st.session_state.text_extracted = text_extracted
                st.session_state["uploaded_files_infos"] = uploaded_files_infos
                st.success("Text extracted successfully")

            elif text_input:
                cleaned_raw_text = ""
                st.session_state["cleaned_raw_text"] = ""
                cleaned_raw_text = clean_text(text_input)
                text_extracted = True
                st.session_state.text_extracted = text_extracted
                st.session_state["cleaned_raw_text"] = cleaned_raw_text
                st.success("Text extracted successfully")

            else:
                st.error("No Text, URL or file uploaded")
                text_extracted = False
                st.session_state.text_extracted = text_extracted

        except Exception as e:
            st.error("Invalid URL or file")
            st.write(e)
 
if st.session_state.text_extracted:
    st.markdown("# Results")
    uploaded_file = st.session_state["uploaded_files_infos"]
    text_input = st.session_state["cleaned_raw_text"]
    check_gpt = st.session_state["check_gpt"]
    if uploaded_file:
        # st.write("### File Text")
        # st.write(uploaded_files_infos)
        pbar = st.progress(0)
        json_result = json.dumps(st.session_state["uploaded_files_infos"])
        for i in range(len(uploaded_files_infos)):
            if len(uploaded_files_infos[i]["cleaned_file_text"].split()) / len(uploaded_files_infos[i]["cleaned_file_text"]) < 0.05:
                p_error = True
            if check_gpt:
                isTextAI = model(uploaded_files_infos[i]["cleaned_file_text"])
                if isTextAI["Perplexity per Line"] < AI_treshold:
                    GPT_generated = True
                df.loc[len(df)] = [uploaded_files_infos[i]["file_name"], str(round(uploaded_files_infos[i]["file_size"]/1000000, 2)) + " MB", uploaded_files_infos[i]["file_type"], len(uploaded_files_infos[i]["cleaned_file_text"]), len(uploaded_files_infos[i]["cleaned_file_text"].split()), GPT_generated, isTextAI["Perplexity"], round(isTextAI["Perplexity per Line"],2), p_error]
            else:
                df.loc[len(df)] = [uploaded_files_infos[i]["file_name"], str(round(uploaded_files_infos[i]["file_size"]/1000000, 2)) + " MB", uploaded_files_infos[i]["file_type"], len(uploaded_files_infos[i]["cleaned_file_text"]), len(uploaded_files_infos[i]["cleaned_file_text"].split()), False, "-", "-", p_error]
            p_error = False
            pbar.progress(len(df) / len(uploaded_files_infos))
        st.success("Text analyzed successfully")
        st.write(df)
        st.markdown("### Download Text")
        st.download_button("Download Text Files as JSON", data=json_result, file_name="textfiles.json", mime="application/json")
        
    elif text_input:
        st.spinner("Analyzing text...")
        
        p_error = False
        if check_gpt:
            isTextAI = model(cleaned_raw_text)
            if isTextAI["Perplexity per Line"] < AI_treshold:
                GPT_generated = True
            df.loc[len(df)] = ["Raw Text", "-", "txt", len(cleaned_raw_text), len(cleaned_raw_text.split()), GPT_generated, isTextAI["Perplexity"], round(isTextAI["Perplexity per Line"], 2), p_error]
        else:
            df.loc[len(df)] = ["Raw Text", "-", "txt", len(cleaned_raw_text), len(cleaned_raw_text.split()), False, "-", "-", p_error]
        st.success("Text analyzed successfully")
        st.write(df)
        st.markdown("### Download Text")
        st.download_button("Download Raw Text", data=cleaned_raw_text, file_name="raw_text.txt", mime="text/plain")
    else:
        st.error("No file uploaded and no text extracted")

    if st.button("🚨 Clear Results"):
        st.session_state.text_extracted = False
        st.session_state.uploaded_files_infos = []
        st.session_state.cleaned_raw_text = ""
        st.session_state.check_gpt = False
        st.experimental_rerun()