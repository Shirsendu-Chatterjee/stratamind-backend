from fastapi import FastAPI, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import secrets
import os

app = FastAPI()
security = HTTPBasic()

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Your Groq API key as environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Embeddings (lightweight for Render free tier)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# In-memory user-specific vector DBs
user_vectordbs = {}

# Simple demo users
users = {
    "alice": "password1",
    "bob": "password2"
}

# Basic auth
def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    correct_password = users.get(credentials.username)
    if not correct_password or not secrets.compare_digest(credentials.password, correct_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return credentials.username

# Login endpoint
@app.get("/login")
def login(username: str = Depends(get_current_username)):
    return {"message": f"Welcome {username}"}

# Upload text for RAG
@app.post("/upload")
async def upload_text(text: str = Form(...), username: str = Depends(get_current_username)):
    # Convert text to document
    documents = [Document(page_content=text)]

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    # Create in-memory vector DB for this user
    vectordb = Chroma.from_documents(docs, embedding_model)
    user_vectordbs[username] = vectordb

    return {"message": "Text uploaded and knowledge base created."}

# Ask question
@app.post("/ask")
async def ask_question(question: str = Form(...), username: str = Depends(get_current_username)):
    vectordb = user_vectordbs.get(username)
    if not vectordb:
        raise HTTPException(status_code=400, detail="Upload text first.")

    retriever = vectordb.as_retriever()

    llm = ChatGroq(
        model_name="llama-3.3-3b",  # Must be available for your free-tier key
        temperature=0,
        api_key=GROQ_API_KEY
    )

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    answer = qa.run(question)
    return {"answer": answer}

# End session / cleanup
@app.post("/end_session")
def end_session(username: str = Depends(get_current_username)):
    if username in user_vectordbs:
        del user_vectordbs[username]
    return {"message": "Session ended and knowledge base deleted."}

# Render entrypoint
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)


