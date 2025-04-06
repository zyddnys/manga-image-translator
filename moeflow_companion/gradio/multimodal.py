from pathlib import Path
import gradio as gr

from ..llm_clients.gemini_bare import GcpGeminiBare
from moeflow_companion.multimodal_workflow import process_images, export_moeflow_project
from moeflow_companion.utils import create_unique_dir
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

with gr.Blocks() as multimodal_workflow_block:
    gr.Markdown("# multimodal workflow")
    file_input = gr.File(
        label="upload file",
        file_count="multiple",
        type="filepath",
    )

    target_language_input = gr.Radio(
        (
            "English",
            "Chinese Simplified",
            "Chinese Traditional",
        ),
        label="translate into language",
        value="Chinese Traditional",
    )

    model_input = gr.Radio(
        choices=[
            GcpGeminiBare.gemini25_pro_exp,
            GcpGeminiBare.gemini20_flash,
            GcpGeminiBare.gemini20_flash_lite,
        ],
        label="LLM",
        value=GcpGeminiBare.gemini20_flash_lite,
    )
    export_moeflow_project_name_input = gr.Text(
        None,
        label="moeflow project name",
        placeholder="when empty, project name will be set to first image filename",
    )
    run_button = gr.Button("run")

    file_output = gr.File(label="moeflow project zip", type="filepath")

    ocr_output = gr.JSON(
        label="process result",
    )

    @run_button.click(
        inputs=[
            file_input,
            model_input,
            target_language_input,
            export_moeflow_project_name_input,
        ],
        outputs=[ocr_output, file_output],
    )
    async def on_run_button(
        gradio_temp_files: list[str],
        model: str,
        target_language: str | None,
        export_moeflow_project_name: str | None,
    ) -> tuple[str, str | None]:
        processed = process_images(
            image_files=[Path(f) for f in gradio_temp_files],
            target_lang=target_language,
            model=model,
        )
        res_obj = [p.model_dump(mode="json") for p in processed]

        moeflow_zip = export_moeflow_project(
            image_files=[Path(f) for f in gradio_temp_files],
            process_result=processed,
            project_name=export_moeflow_project_name,
            dest_dir=create_unique_dir("export"),
        )

        return res_obj, str(moeflow_zip)
