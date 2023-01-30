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
    # st.markdown(
    #     "<h3 style='text-align: center; color: green;'>OR</h3>",
    #     unsafe_allow_html=True,
    # )
    # uploaded_list_of_urls = st.file_uploader(
    #     "Upload a list of URLs in .txt format for summarization.", type="txt", accept_multiple_files=False
    # )
    submit_text_extract = st.form_submit_button(label="Go!")    

if submit_text_extract:
    text_extracted = False
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
            st.session_state["uploaded_files_infos"] = uploaded_files_infos
            st.success("Text extracted successfully")
        # elif validators.url(text_input):
        #     url_text = []
        #     st.session_state["url_text"] = []
        #     url_raw_text, url_sentences = fetch_article_text(url=text_input)
        #     cleaned_url_text = clean_text(url_raw_text)
        #     url_dict = {'url': text_input, 'url_text': url_raw_text, 'cleaned_url_text': cleaned_url_text, 'url_text_chunks': url_sentences}
        #     url_text.append(url_dict)
        #     text_extracted = True
        #     st.session_state["url_text"] = url_text
        #     st.success("Text extracted successfully")
        elif text_input:
            cleaned_raw_text = ""
            st.session_state["cleaned_raw_text"] = ""
            cleaned_raw_text = clean_text(text_input)
            text_extracted = True
            st.session_state["cleaned_raw_text"] = cleaned_raw_text
            st.success("Text extracted successfully")
        # elif uploaded_list_of_urls:
        #     if uploaded_list_of_urls is not None:
        #             url_infos = []
        #             st.session_state["url_infos"] = []
        #             st.write("Extracting text from URLs...")
        #             progress_bar = st.progress(0)
        #             list_of_urls = uploaded_list_of_urls.read().decode('utf-8')
        #             list_of_urls = list_of_urls.splitlines()
        #             total_url_count = len(list_of_urls)
        #             for url in list_of_urls:
        #                 try:
        #                     url = format_url(url)
        #                     if validators.url(url):
        #                         article_text, article_text_chunks = fetch_article_text(url=url)
        #                         cleaned_article_text = clean_text(article_text)
        #                         file_dict = {'url': url, 'article_text': article_text, 'cleaned_article_text': cleaned_article_text, 'article_text_chunks': article_text_chunks}
        #                         url_infos.append(file_dict)
        #                     else:
        #                         st.error(f"Invalid URL: {url}")
        #                 except Exception as e:
        #                     st.error("Invalid URL" + url)
        #                 progress_bar.progress(len(url_infos) / total_url_count)
        #     text_extracted = True
        #     st.session_state["url_infos"] = url_infos
        #     st.success("Text extracted successfully")
        else:
            st.error("No Text, URL or file uploaded")
            text_extracted = False

    except Exception as e:
        st.error("Invalid URL or file")
        st.write(e)
 
if text_extracted:
    st.markdown("# Results")
    if uploaded_file:
        # st.write("### File Text")
        # st.write(uploaded_files_infos)
        pbar = st.progress(0)
        json_result = json.dumps(st.session_state["uploaded_files_infos"])
        for i in range(len(uploaded_files_infos)):
            if len(uploaded_files_infos[i]["cleaned_file_text"].split()) / len(uploaded_files_infos[i]["cleaned_file_text"]) < 0.05:
                p_error = True
            isTextAI = model(uploaded_files_infos[i]["cleaned_file_text"])
            if isTextAI["Perplexity per Line"] < AI_treshold:
                GPT_generated = True
            df.loc[len(df)] = [uploaded_files_infos[i]["file_name"], str(round(uploaded_files_infos[i]["file_size"]/1000000, 2)) + " MB", uploaded_files_infos[i]["file_type"], len(uploaded_files_infos[i]["cleaned_file_text"]), len(uploaded_files_infos[i]["cleaned_file_text"].split()), GPT_generated, isTextAI["Perplexity"], round(isTextAI["Perplexity per Line"],2), p_error]
            p_error = False
            pbar.progress(len(df) / len(uploaded_files_infos))
        st.success("Text analyzed successfully")
        st.write(df)
        st.markdown("### Download Text")
        st.download_button("Download Text Files as JSON", data=json_result, file_name="textfiles.json", mime="application/json")
        
    # elif validators.url(text_input):
    #     # st.write("### Article Text")
    #     # st.write(url_text)
    #     # st.write("### Cleaned Article Text")
    #     # st.write(cleaned_url_text)
    #     # st.write("### Article Text Chunks")
    #     # st.write(url_sentences)
    #     json_result = json.dumps(st.session_state["url_text"])
    #     st.download_button("Download URL Text", data=json_result, file_name="url_text.json", mime="application/json")
    # elif uploaded_list_of_urls:
    #     # st.write("### URLs Text")
    #     # st.write(url_infos)
    #     json_result = json.dumps(st.session_state["url_infos"])
    #     st.download_button("Download URLs Text", data=json_result, file_name="urls_text.json", mime="application/json")
    elif text_input:
        st.spinner("Analyzing text...")
        # st.write("### Cleaned Raw Text")
        # st.write(cleaned_raw_text
        
        p_error = False
        isTextAI = model(cleaned_raw_text)
        if isTextAI["Perplexity per Line"] < AI_treshold:
            GPT_generated = True
        df.loc[len(df)] = ["Raw Text", "-", "txt", len(cleaned_raw_text), len(cleaned_raw_text.split()), GPT_generated, isTextAI["Perplexity"], round(isTextAI["Perplexity per Line"], 2), p_error]
        st.success("Text analyzed successfully")
        st.write(df)
        st.markdown("### Download Text")
        st.download_button("Download Raw Text", data=cleaned_raw_text, file_name="raw_text.txt", mime="text/plain")
    else:
        st.error("No file uploaded and no text extracted")

