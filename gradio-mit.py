"""
Gradio app: manga-image-translator only
"""

import logging
import gradio as gr
from moeflow_companion.gradio.mit import mit_workflow_block

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
    mit_workflow_block.queue(api_open=True, max_size=100).launch(
        share=False,
        debug=True,
        server_name="0.0.0.0",
        max_file_size=10 * gr.FileSize.MB,
    )
