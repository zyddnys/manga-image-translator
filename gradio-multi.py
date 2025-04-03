import logging
import gradio as gr
from pathlib import Path
from moeflow_companion.mit_workflow import (
    process_files,
    is_cuda_avaiable,
    create_unique_dir,
)
from moeflow_companion.exporter import export_moeflow_project

if gr.NO_RELOAD:
    logging.basicConfig(
        level=logging.WARN,
        format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    for name in ["httpx"]:
        logging.getLogger(name).setLevel(logging.WARN)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


with gr.Blocks() as demo:
    file_input = gr.File(
        label="upload file",
        file_count="multiple",
        type="filepath",
    )

    target_language_input = gr.Radio(
        ("ENG", "CHS", "CHT", None), label="translate into language", value="CHS"
    )

    device_input = gr.Radio(
        choices=["cpu", "cuda"],
        label="device",
        value="cuda" if is_cuda_avaiable() else "cpu",
    )
    detector_key_input = gr.Radio(
        choices=[
            "default",
            # maybe broken: manga_translator.utils.inference.InvalidModelMappingException: [DBConvNextDetector->model] Invalid _MODEL_MAPPING - Malformed url property
            #  "dbconvnext",
            "ctd",
            "craft",
            "none",
        ],
        value="default",
        label="detector",
    )

    export_moeflow_project_name = gr.Text(None, label="moeflow project name")

    ocr_key_input = gr.Radio(
        choices=["48px", "48px_ctc", "mocr"], label="ocr", value="48px"
    )
    run_button = gr.Button("run")

    file_output = gr.File(label="moeflow project zip", type="filepath")

    ocr_output = gr.JSON(
        label="process result",
    )

    @run_button.click(
        inputs=[
            file_input,
            detector_key_input,
            ocr_key_input,
            device_input,
            target_language_input,
        ],
        outputs=[ocr_output, file_output],
    )
    async def on_run_button(
        gradio_temp_files: list[str],
        detector_key: str,
        ocr_key: str,
        device: str,
        target_language: str | None,
        export_moeflow_project_name: str | None,
    ) -> tuple[str, bytes]:
        res = await process_files(
            gradio_temp_files,
            detector_key=detector_key,
            ocr_key=ocr_key,
            device=device,
            # translator_key="gpt4",
            target_language=target_language,
        )
        if res:
            moeflow_zip = str(
                export_moeflow_project(
                    res,
                    export_moeflow_project_name,
                    output_dir=Path(create_unique_dir()),
                )
            )
        else:
            moeflow_zip = None

        output_json = {
            "project_name": "unnamed",
            "files": [f.model_dump() for f in res.files],
        }

        return output_json, moeflow_zip


if __name__ == "__main__":
    demo.queue(api_open=True, max_size=100).launch(
        share=False,
        debug=True,
        server_name="0.0.0.0",
        max_file_size=10 * gr.FileSize.MB,
    )
