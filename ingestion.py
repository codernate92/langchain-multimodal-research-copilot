from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders.image import UnstructuredImageLoader
from langchain.document_loaders.audio import UnstructuredAudioLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

import os

CHROMA_DIR = "chroma_store"
EMBEDDING_MODEL = OpenAIEmbeddings()

def load_documents(file_paths):
    docs = []
    for path in file_paths:
        ext = os.path.splitext(path)[1].lower()
        if ext.endswith('.pdf'):
            loader = UnstructuredPDFLoader(path)
        elif ext.endswith(('.jpg', '.jpeg', '.png')):
            loader = UnstructuredImageLoader(path)
        elif ext.endswith(('.mp3', '.mp4', '.wav')):
            loader = UnstructuredAudioLoader(path)
        else:
            print(f"Unsupported file type: {path}")
            continue
        docs.extend(loader.load())
    return docs

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

def embed_documents(documents):
    vectorstore = Chroma.from_documents(documents, EMBEDDING_MODEL, persist_directory=CHROMA_DIR)
    vectorstore.persist()
    return vectorstore

def ingest(file_paths):
    print("📥 Loading documents...")
    docs = load_documents(file_paths)
    print(f"✅ Loaded {len(docs)} documents. Now chunking...")
    splits = split_documents(docs)
    print(f"🧠 {len(splits)} chunks created. Embedding...")
    embed_documents(splits)
    print("✅ Ingestion complete. Vectorstore saved in chroma_store/")

if __name__ == "__main__":
    files = ["sample.pdf", "image.png", "interview.mp3"]  # Replace with real paths
    ingest(files)
