import gradio as gr
import os
from dotenv import load_dotenv
from ingest import ingest_pdf
from rag import answer_question

load_dotenv()

def process_pdf_and_ask(pdf_file, question):
    if pdf_file is None:
        return "Please upload a PDF file first."
    if not question or question.strip() == "":
        return "Please enter a question."
    
    try:
        ingest_pdf(pdf_file.name)
        answer = answer_question(question)
        return answer
    except Exception as e:
        return f"Error: {str(e)}"

demo = gr.Interface(
    fn=process_pdf_and_ask,
    inputs=[
        gr.File(
            label="Upload PDF",
            file_types=[".pdf"]
        ),
        gr.Textbox(
            label="Ask a question about your PDF",
            placeholder="e.g. What are this person's technical skills?"
        )
    ],
    outputs=gr.Textbox(
        label="Answer",
        lines=5
    ),
    title="RAG Powered Q&A System",
    description="Upload any PDF document and ask questions about it using LLMs powered by LangChain, FAISS, and Groq.",
    examples=[
        [None, "What are the main topics covered in this document?"],
        [None, "Summarize the key points of this document."]
    ]
)

if __name__ == "__main__":
    demo.launch()