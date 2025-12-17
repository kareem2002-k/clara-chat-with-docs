"""
Gradio web application
"""

import gradio as gr
from .models import load_models
from .utils import parse_docs, format_output
from .rag import run_normal_rag
from .clara import run_clara


def process_query(mode: str, docs_text: str, question: str, top_k: int):
    """Main processing function called by Gradio"""
    docs = parse_docs(docs_text)
    
    if not docs:
        return "Error: No documents found. Please separate documents with '---'"
    
    if mode == "CLaRa":
        answer, evidence = run_clara(docs, question)
    else:  # Normal RAG
        answer, evidence = run_normal_rag(docs, question, top_k)
    
    return format_output(answer, evidence)


def create_app():
    """Create and return Gradio app"""
    # Load models at startup
    print("Initializing...")
    load_models()
    
    # Create Gradio UI
    with gr.Blocks(title="CLaRa vs Normal RAG Demo") as demo:
        gr.Markdown("""
        # CLaRa vs Normal RAG Comparison Demo
        
        This demo compares two approaches to document-based question answering:
        - **CLaRa**: Latent document compression (no prompt stuffing)
        - **Normal RAG**: Retrieve top-K docs → stuff into prompt → LLM answers
        """)
        
        with gr.Row():
            with gr.Column():
                mode = gr.Radio(
                    choices=["Normal RAG", "CLaRa"],
                    value="Normal RAG",
                    label="Mode"
                )
                top_k = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=3,
                    step=1,
                    label="Top K (for RAG mode)"
                )
                docs_input = gr.Textbox(
                    label="Documents",
                    placeholder="Paste your documents here, separated by '---'\n\nExample:\nDoc A text...\n---\nDoc B text...\n---\nDoc C text...",
                    lines=10
                )
                question_input = gr.Textbox(
                    label="Question",
                    placeholder="Enter your question here..."
                )
                run_btn = gr.Button("Run", variant="primary")
            
            with gr.Column():
                output = gr.Markdown(label="Result")
        
        run_btn.click(
            fn=process_query,
            inputs=[mode, docs_input, question_input, top_k],
            outputs=output
        )
        
        gr.Markdown("""
        ### How to use:
        1. Paste your documents in the textarea, separated by `---`
        2. Enter your question
        3. Choose the mode (Normal RAG or CLaRa)
        4. For RAG mode, adjust Top-K slider
        5. Click "Run"
        
        ### Example Documents:
        ```
        The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower from 1887 to 1889.
        ---
        The Statue of Liberty is a neoclassical sculpture on Liberty Island in New York Harbor. It was a gift from France to the United States and was dedicated in 1886.
        ---
        The Great Wall of China is a series of fortifications made of stone, brick, and other materials, generally built along an east-to-west line across the historical northern borders of China.
        ```
        """)
    
    return demo


if __name__ == "__main__":
    demo = create_app()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)

