"""
Gradio app: everything in tabs
"""

import logging
import gradio as gr
from moeflow_companion import setup_logging
from moeflow_companion.gradio.mit import mit_workflow_block
from moeflow_companion.gradio.multimodal import (
    multimodal_workflow_block,
)

if gr.NO_RELOAD:
    setup_logging()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    demo = gr.TabbedInterface(
        [mit_workflow_block, multimodal_workflow_block],
        tab_names=["manga-image-translator", "multimodal LLM"],
        title="moeflow companion | 萌翻助手",
    )
    demo.queue(api_open=True, max_size=100).launch(
        share=False,
        debug=True,
        server_name="0.0.0.0",
        max_file_size=10 * gr.FileSize.MB,
    )
