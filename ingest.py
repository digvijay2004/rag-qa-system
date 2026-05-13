from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def ingest_pdf(pdf_path: str):
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    
    # Create embeddings using Hugging Face
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Store in FAISS vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Save locally
    vectorstore.save_local("faiss_index")
    
    return f"Successfully ingested {len(chunks)} chunks from PDF"

if __name__ == "__main__":
    result = ingest_pdf("sample.pdf")
    print(result)