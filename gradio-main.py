"""
Gradio app: everything in tabs
"""

import logging
import gradio as gr
from moeflow_companion.mit_workflow.gradio_component import mit_workflow_block
from moeflow_companion.multimodal_workflow.gradio_component import (
    multimodal_workflow_block,
)

if gr.NO_RELOAD:
    logging.basicConfig(
        level=logging.WARN,
        format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for name in ["httpx"]:
        logging.getLogger(name).setLevel(logging.WARN)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    demo = gr.TabbedInterface(
        [mit_workflow_block, multimodal_workflow_block],
        tab_names=["manga-image-translator", "multimodal LLM"],
        title="moeflow pre-translate companion | 萌翻助手：漫画预翻译",
    )
    demo.queue(api_open=True, max_size=100).launch(
        share=False,
        debug=True,
        server_name="0.0.0.0",
        max_file_size=10 * gr.FileSize.MB,
    )
