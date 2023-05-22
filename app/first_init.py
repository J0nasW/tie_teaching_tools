####################################################################################
# Initialization
# by JW
#
# A helper script to initialize the DB and other things at first boot of the
# container cluster.
# 
# helpers / first_init.py
####################################################################################

# IMPORT STATEMENTS ----------------------------------------------------------------
import streamlit as st
from random import randint



def init_application():
    try:
        # Text Extraction
        if "text_extracted" not in st.session_state:
            st.session_state["text_extracted"] = False

        if "url_text" not in st.session_state:
            st.session_state["url_text"] = []

        if "cleaned_raw_text" not in st.session_state:
            st.session_state["cleaned_raw_text"] = ""

        if "uploaded_files_infos" not in st.session_state:
            st.session_state["uploaded_files_infos"] = []

        if "url_infos" not in st.session_state:
            st.session_state["url_infos"] = []

        if "check_gpt" not in st.session_state:
            st.session_state["check_gpt"] = False
            
        if "create_embedding" not in st.session_state:
            st.session_state["create_embedding"] = False
            
        if "file_uploader_key" not in st.session_state:
            st.session_state["file_uploader_key"] = str(randint(1000, 100000000))

         # Initialization done --------------------------------------------------------------
        st.session_state.app_init = True
    
    except Exception as e:
        st.warning("There was a problem initializing the app", icon="⚠️")
        st.write(e)