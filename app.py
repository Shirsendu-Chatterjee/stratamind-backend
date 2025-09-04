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

print("Backend starting...")  # helps Render detect server start

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Small embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Store in-memory DB per session
app.state.vectordb = None

@app.post("/upload")
async def upload_file(file: UploadFile):
    suffix = file.filename.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Load document
    loader = PyPDFLoader(tmp_path) if suffix.lower() == "pdf" else TextLoader(tmp_path)
    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)

    # Create in-memory vector DB
    vectordb = Chroma.from_documents(docs, embedding_model)
    app.state.vectordb = vectordb

    return {"message": f"File {file.filename} processed successfully."}

@app.post("/ask")
async def ask_question(question: str = Form(...)):
    if not app.state.vectordb:
        return {"error": "Upload a file first."}

    retriever = app.state.vectordb.as_retriever()

    # Small Groq model for free tier
    llm = ChatGroq(
        model_name="llama-3.3-3b-versatile",  # smaller model
        temperature=0,
        api_key=GROQ_API_KEY,
    )

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    answer = qa.run(question)

    return {"answer": answer}

# For Render free tier detection
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)


