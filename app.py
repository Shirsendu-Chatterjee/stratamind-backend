from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import os

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Lightweight embeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Store vector DB per session
user_vectordbs = {}

# Upload text
@app.post("/upload")
async def upload_text(session_id: str = Form(...), text: str = Form(...)):
    documents = [Document(page_content=text)]
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    vectordb = Chroma.from_documents(docs, embedding_model)
    user_vectordbs[session_id] = vectordb
    return {"message": "Text uploaded and knowledge base created."}

# Ask question
@app.post("/ask")
async def ask_question(session_id: str = Form(...), question: str = Form(...)):
    vectordb = user_vectordbs.get(session_id)
    if not vectordb:
        return {"error": "Upload text first."}

    retriever = vectordb.as_retriever()

    llm = ChatGroq(
        model_name="llama-3.3-3b",  # Must match your free-tier key
        temperature=0,
        api_key=GROQ_API_KEY
    )

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    answer = qa.run(question)
    return {"answer": answer}

# End session
@app.post("/end_session")
async def end_session(session_id: str = Form(...)):
    if session_id in user_vectordbs:
        del user_vectordbs[session_id]
    return {"message": "Session ended and knowledge base deleted."}

# Render entrypoint
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
