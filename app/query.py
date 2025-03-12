import os
import argparse
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM

load_dotenv()

CHROMA_DB_DIR =  os.getenv("CHROMA_DB_DIR") 
LLM_MODEL =  os.getenv("LLM_MODEL")

def query(query):
    # 1️⃣ Load DB already indexed
    embedding_function = OllamaEmbeddings(model=LLM_MODEL)
    vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embedding_function)

    # 2️⃣ Configure document retriever
    retriever = vectorstore.as_retriever()

    # 3️⃣ Create the Question-Answer chain (RAG)
    qa_chain = RetrievalQA.from_chain_type(OllamaLLM(model=LLM_MODEL), retriever=retriever)

    # 4️⃣ Query
    response = qa_chain.invoke(query)

    print(response['result'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, help="Query to search in the documents", required=True)
    args = parser.parse_args()
    query(args.query)
