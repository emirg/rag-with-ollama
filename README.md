# 🚀 RAG API with Ollama, ChromaDB y FastAPI

This project implements a **RAG (Retrieval-Augmented Generation)** system using **Ollama** with **Mistral** as the language model, **ChromaDB** as the vector database, and **FastAPI** to expose a query API.

## 📌 **Features**
✅ **Loads and splits documents** in `.pdf` formats.  
✅ **Indexes and reindexes documents in ChromaDB** without deleting previous data.  
✅ **Exposes a FastAPI-based API** for querying documentation.  
✅ **Uses embeddings and response generation with Ollama**.

## 🔧 **Requirements**:
```bash
pip install -r requirements.txt
```

Additionally, you need [Ollama](https://ollama.com/) client installed

## 🏃‍♂️‍➡️ **Quick Start**:
### Indexing your data
1. Create your **.env** file
    The requiered variables are the following:  
    ```bash
    DOCUMENTS_FOLDER = "app/data"  # Folder with the base documents to index
    NEW_DOCUMENTS_FOLDER = "app/data/new-data"  # Folder with new files to index
    CHROMA_DB_DIR = "db_chroma"  # Folder where the DB is stored
    LLM_MODEL = "mistral"  # LLM model to use
    ```
2. Put all your PDF files in the **data** folder
3. Run the indexation script so that your documents get indexed in ChromaDB
    ```bash
    python3 app/indexation.py
    ```
### Querying your LLM  
You have 2 ways of querying your LLM:
#### Directly running Python
```bash
python3 app/query.py
```
#### Running the API locally
```bash
uvicorn app.api.api:app --reload
```

Once the server is up, by default it will be running _localhost:8000_

You can access the Swagger UI http://localhost:8000/docs and test directly there

Additionaly, you can run a cURL request or an HTTP client like Postman:
```bash
curl -X 'POST' 'http://127.0.0.1:8000/query/' \
     -H 'Content-Type: application/json' \
     -d '{"question": "What is greedy algorithm?"}'
```

## 🆕**Adding more documents**:
Once you index your documents, you don't have to reindex your entire knowledge base when adding more information/docuemnts. You can use the reindex feature to add documents to the already existing knowledge
```bash
python3 app/indexation.py --reindex
```

This will run the indexation only of the documents found in the data/new-data, and will not overwrite the current ChromeDB, it will append more data.