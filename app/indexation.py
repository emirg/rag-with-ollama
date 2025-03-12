import os
import argparse
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings

load_dotenv()

DOCUMENTS_FOLDER = os.getenv("DOCUMENTS_FOLDER")  # Folder with the documents to index
NEW_DOCUMENTS_FOLDER = os.getenv("NEW_DOCUMENTS_FOLDER") 
CHROMA_DB_DIR =  os.getenv("CHROMA_DB_DIR") 
LLM_MODEL =  os.getenv("LLM_MODEL")
 
def load_documents(path=DOCUMENTS_FOLDER):
    documents = []
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            print(f"⚠️ File ignored (format not compatible): {filename}")
            continue
        documents.extend(loader.load())

    return documents

def index_documents():
    print("🔄 Indexing first documents")
    print("📂 Loading documents...")
    documents = load_documents()    

    if not documents:
        print("❌ No documents found.")
        return

    print(f"✅ {len(documents)} documents loaded.")

    # Split into fragments
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    print(f"🔹 Fragments generated: {len(docs)}")
    
    print("🔄 Indexing database.")
    embedding_function = OllamaEmbeddings(model=LLM_MODEL)
    Chroma.from_documents(docs, embedding_function, persist_directory="db_chroma")

    print("✅ Database created based on the documentation.")

def reindex_documents():
    print("🔄 Reindexing new documents")
    print("📂 Loading documents...")
    documents = load_documents(NEW_DOCUMENTS_FOLDER)

    if not documents:
        print("❌ No new documents found.")
        return

    print(f"✅ {len(documents)} documents loaded.")

    # Split into fragments
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(documents)
    
    print(f"🔹 Fragments generated: {len(docs)}")

    # Connect with ChromaDB
    embedding_function = OllamaEmbeddings(model=LLM_MODEL)
    vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embedding_function)

    # Add new documents to the index
    vectorstore.add_documents(docs)

    print("✅ Database updated with new documentation.")    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reindex",action="store_true",help="Whether you want to reindex new documents", required=False)
    args = parser.parse_args()
    if(args.reindex):
        reindex_documents()
    else:
        index_documents()
