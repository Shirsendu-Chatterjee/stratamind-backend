from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_groq import ChatGroq
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
import os

app = FastAPI()

# frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# @app.get("/")
# def serve_frontend():
#     return FileResponse(os.path.join(frontend_dir, "index.html"))

@app.get("/health")
def health():
    return {"status": "ok"}

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

embeddings = HuggingFaceEmbeddings(model_name="paraphrase-MiniLM-L3-v2")
qa = None

@app.post("/upload_text/")
async def upload_text(file: UploadFile):
    global qa
    contents = await file.read()
    text = contents.decode("utf-8")
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_text(text)
    vectordb = Chroma.from_texts(docs, embedding=embeddings)
    retriever = vectordb.as_retriever()
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="allam-2-7b",
        temperature=0.2,
    )
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return {"message": "File uploaded successfully!", "filename": file.filename}

@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    global qa
    if qa is None:
        return {"answer": "❌ No knowledge uploaded yet."}
    result = await qa.arun(question)
    return {"answer": result}
