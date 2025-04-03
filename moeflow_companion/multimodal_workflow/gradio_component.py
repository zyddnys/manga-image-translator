import gradio as gr

with gr.Blocks() as multimodal_workflow_block:
    gr.Markdown("# multimodal workflow: let LLM do ")
    file_input = gr.File(
        label="upload file",
        file_count="multiple",
        type="filepath",
    )

    model_input = gr.Radio(
        choices=["gemini"],
        label="LLM",
    )
