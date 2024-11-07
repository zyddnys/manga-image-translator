import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
import logging
import tempfile
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

uploaded: list[UploadedFile] = st.file_uploader("pick a file", type=['png', 'jpg'], accept_multiple_files=True)

logger.debug('uploaded: %s', uploaded)

def start_save():
    with tempfile.TemporaryDirectory() as tmpdirname:
        for file in uploaded:
            with open(f'{tmpdirname}/{file.name}', 'wb') as f:
                f.write(file.getvalue())
                logger.info("saved %/%s", tmpdirname, file.name)

st.button("save", on_click=start_save)