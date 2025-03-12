from fastapi import FastAPI
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM

CHROMA_DB_DIR = "db_chroma"

def initialize_rag():
    embedding_function = OllamaEmbeddings(model="mistral")
    vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embedding_function)
    retriever = vectorstore.as_retriever()
    # Configurar la cadena de QA
    qa_chain = RetrievalQA.from_chain_type(OllamaLLM(model="mistral"), retriever=retriever)
    return qa_chain

# Initialize FastAPI
app = FastAPI(title="RAG API", description="API para consultar documentos con RAG", version="1.0")

qa_chain = initialize_rag()

# Define query schema
class QueryRequest(BaseModel):
    question: str

@app.post("/query/")
async def ask_question(request: QueryRequest):
    """Recibe una pregunta y devuelve la respuesta basada en los documentos indexados."""
    response = qa_chain.invoke(request.question)
    return {"question": request.question, "answer": response['result']}

@app.get("/")
async def root():
    return {"message": "Bienvenido a la API de RAG. Usa /query/ para hacer preguntas."}



