import gradio as gr
from moeflow_companion.mit_workflow import (
    process_files,
    is_cuda_avaiable,
    export_moeflow_project,
)
from moeflow_companion.utils import create_unique_dir


with gr.Blocks() as mit_workflow_block:
    # inputs
    gr.Markdown("# manga-image-translator workflow: text detection > ocr > translation")
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

    export_moeflow_project_name_input = gr.Text(
        None,
        label="moeflow project name",
        placeholder="when empty, project name will be set to first image filename",
    )

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
            export_moeflow_project_name_input,
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
    ) -> tuple[str, str | None]:
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
                    dest_dir=create_unique_dir("export"),
                )
            )
        else:
            moeflow_zip = None

        output_json = {
            "project_name": export_moeflow_project_name or "unnamed",
            "files": [f.model_dump(mode="json") for f in res.files],
        }

        return output_json, moeflow_zip
