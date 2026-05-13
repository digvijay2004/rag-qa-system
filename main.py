import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
from dotenv import load_dotenv
from ingest import ingest_pdf
from rag import answer_question

load_dotenv()

app = FastAPI(
    title="RAG Q&A System",
    description="Upload a PDF and ask questions about it using LLM",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "RAG Q&A System is running!"}

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    # Save uploaded PDF
    pdf_path = f"uploaded_{file.filename}"
    with open(pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Ingest the PDF
    result = ingest_pdf(pdf_path)
    
    return {"message": result, "filename": file.filename}

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    answer = answer_question(request.question)
    return {
        "question": request.question,
        "answer": answer
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}