from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import os

# Init FastAPI
app = FastAPI()

# Allow CORS (so frontend can call this backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq API key (set this in HF Spaces secrets!)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Create embeddings & LLM
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=GROQ_API_KEY)

# Temporary in-memory store (one per session)
user_vectorstores = {}

class QueryRequest(BaseModel):
    query: str
    session_id: str

@app.post("/ingest/{session_id}")
async def ingest_text(session_id: str, request: Request):
    data = await request.json()
    text = data.get("text", "")

    if not text.strip():
        return {"error": "No text provided"}

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = [Document(page_content=chunk) for chunk in splitter.split_text(text)]

    # Create a Chroma vectorstore for this session
    vectorstore = Chroma.from_documents(docs, embedding=embeddings, persist_directory=f"db_{session_id}")
    user_vectorstores[session_id] = vectorstore

    return {"status": "Text ingested"}

@app.post("/query")
async def query_rag(req: QueryRequest):
    session_id = req.session_id
    query = req.query

    if session_id not in user_vectorstores:
        return {"error": "No knowledge base found for this session"}

    retriever = user_vectorstores[session_id].as_retriever()
    docs = retriever.get_relevant_documents(query)

    # Build context
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"Answer based on the following context:\n{context}\n\nQuestion: {query}\nAnswer:"

    result = llm.invoke(prompt)

    return {"answer": result.content}
