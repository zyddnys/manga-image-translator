import streamlit as st
import streamlit_pydantic as sp

from manga_translator.streamlit import start_translate_task, TranslateTaskDef

task_def = sp.pydantic_form(key="single_input_file", model=TranslateTaskDef)

if task_def:
    sp.pydantic_output(task_def)