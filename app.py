import os
import tempfile
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings

app = FastAPI()

# Allow frontend (adjust origin if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Groq API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@app.post("/upload")
async def upload_file(file: UploadFile):
    """Upload file and build temporary vector DB"""
    # Save temp file
    suffix = file.filename.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Load document
    if suffix.lower() == "pdf":
        loader = PyPDFLoader(tmp_path)
    else:
        loader = TextLoader(tmp_path)
    documents = loader.load()

    # Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)

    # Create vector DB in memory
    vectordb = Chroma.from_documents(docs, embedding_model)

    # Save reference
    app.state.vectordb = vectordb
    return {"message": f"File {file.filename} processed successfully."}

@app.post("/ask")
async def ask_question(question: str = Form(...)):
    """Ask a question based on uploaded document"""
    if not hasattr(app.state, "vectordb"):
        return {"error": "Please upload a file first."}

    retriever = app.state.vectordb.as_retriever()

    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0,
        api_key=GROQ_API_KEY,
    )

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    answer = qa.run(question)

    return {"answer": answer}
