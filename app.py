from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_groq import ChatGroq
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import uuid
import shutil
import tempfile
import time
import threading

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Store user sessions: {user_id: {"qa": RetrievalQA, "db_path": str, "last_active": float}}
user_sessions = {}

chatbot_prompt = """
You are the official website assistant for this business.
Always respond as if you are the company itself.
Be confident, friendly, and concise.
Never mention 'context', 'retrieved documents', or 'knowledge base'.
Never say "I think" or "it seems". Always state answers as facts.
If the user asks something unrelated, politely steer back to the business.

Question: {question}
Relevant Info: {context}

Answer:
"""

CHATBOT_PROMPT = PromptTemplate(
    template=chatbot_prompt,
    input_variables=["context", "question"]
)

# Session cleanup thread
SESSION_TIMEOUT = 1800  # 1 hour in seconds

def cleanup_sessions():
    while True:
        now = time.time()
        expired = []
        for user_id, session in list(user_sessions.items()):
            if now - session["last_active"] > SESSION_TIMEOUT:
                expired.append(user_id)
        for user_id in expired:
            db_path = user_sessions[user_id]["db_path"]
            shutil.rmtree(db_path, ignore_errors=True)
            del user_sessions[user_id]
            print(f"🗑️ Session {user_id} expired and removed.")
        time.sleep(300)  # check every 5 minutes

threading.Thread(target=cleanup_sessions, daemon=True).start()


@app.post("/upload_text/")
async def upload_text(file: UploadFile):
    user_id = str(uuid.uuid4())

    contents = await file.read()
    try:
        text = contents.decode("utf-8")
    except Exception:
        return JSONResponse(status_code=400, content={"error": "File must be UTF-8 text."})

    splitter = CharacterTextSplitter(chunk_size=20, chunk_overlap=4)
    docs = splitter.split_text(text)

    db_path = tempfile.mkdtemp()

    vectordb = Chroma.from_texts(docs, embedding=embeddings, persist_directory=db_path)
    retriever = vectordb.as_retriever()

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="allam-2-7b",
        temperature=0.1,
    )
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": CHATBOT_PROMPT},
    )

    user_sessions[user_id] = {"qa": qa, "db_path": db_path, "last_active": time.time()}

    return {"message": "File uploaded successfully!", "filename": file.filename, "user_id": user_id}


@app.post("/ask/")
async def ask_question(user_id: str = Form(...), question: str = Form(...)):
    if user_id not in user_sessions:
        return {"answer": "❌ Invalid or expired session."}

    session = user_sessions[user_id]
    session["last_active"] = time.time()  # refresh activity timestamp

    qa = session["qa"]
    result = await qa.arun(question)
    return {"answer": result}


@app.post("/end_session/")
async def end_session(user_id: str = Form(...)):
    if user_id in user_sessions:
        db_path = user_sessions[user_id]["db_path"]
        shutil.rmtree(db_path, ignore_errors=True)
        del user_sessions[user_id]
        return {"message": f"Session {user_id} ended and DB deleted."}
    return {"message": "❌ Session not found."}
