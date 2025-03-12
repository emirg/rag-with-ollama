import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM

load_dotenv()

CHROMA_DB_DIR =  os.getenv("CHROMA_DB_DIR") 
LLM_MODEL =  os.getenv("LLM_MODEL")

def initialize_rag():
    embedding_function = OllamaEmbeddings(model=LLM_MODEL)
    vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embedding_function)
    retriever = vectorstore.as_retriever()
    # Configure document retriever
    qa_chain = RetrievalQA.from_chain_type(OllamaLLM(model=LLM_MODEL), retriever=retriever)
    return qa_chain

# Initialize FastAPI
app = FastAPI(title="RAG API", description="API for querying documents with RAG", version="1.0")

qa_chain = initialize_rag()

# Define query schema
class QueryRequest(BaseModel):
    question: str

@app.post("/query/")
async def ask_question(request: QueryRequest):
    response = qa_chain.invoke(request.question)
    return {"question": request.question, "answer": response['result']}

@app.get("/")
async def root():
    return {"message": "Welcome to a RAG API. Use /query/."}



