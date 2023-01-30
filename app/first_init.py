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

         # Initialization done --------------------------------------------------------------
        st.session_state.app_init = True
    
    except Exception as e:
        st.warning("There was a problem initializing the app", icon="⚠️")
        st.write(e)