import os
from dotenv import load_dotenv
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loaders.markdown_loader import MarkdownLoader
from db_connectors.chroma_connector import ChromaConnector
from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings

load_dotenv()

DOCUMENTS_FOLDER = os.getenv("DOCUMENTS_FOLDER")  # Folder with the documents to index
NEW_DOCUMENTS_FOLDER = os.getenv("NEW_DOCUMENTS_FOLDER") 
CHROMA_DB_DIR =  os.getenv("CHROMA_DB_DIR") 
LLM_MODEL =  os.getenv("LLM_MODEL")
OBSIDIAN_VAULT_PATH = os.getenv("OBSIDIAN_VAULT_PATH") # Path to the Obsidian vault

print(OBSIDIAN_VAULT_PATH)

def load_notes(directory_path: str) -> List[Dict[str, str]]:
    return  MarkdownLoader(directory_path).load()

def store_documents(documents: List[Dict[str, str]], persist_path: str = None):
    connector = ChromaConnector(persist_path)
    connector.add_documents("default_collection", documents)
    return connector


print("📝 Loading Obsidian notes...")
notes = load_notes(OBSIDIAN_VAULT_PATH)
print(f"Loaded {len(notes)} documents")

# Store documents in vector database
connector = store_documents(notes, './db_chroma')
print("Documents stored in vector database")
print("✅ Notas de Obsidian indexadas en ChromaDB")
