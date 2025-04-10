import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA

load_dotenv()

#Load embedding and LLM models securely
EMBEDDING_MODEL = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
LLM_MODEL = OpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))

VECTORSTORE_DIR = "faiss_store"

def get_vectorstore():
    """Load the FAISS vectorstore."""
    print("🔄 Loading FAISS vectorstore...")
    vectordb = FAISS.load_local(VECTORSTORE_DIR, EMBEDDING_MODEL)
    return vectordb

def get_retriever():
    """"Returns the retriever from the FAISS vectorstore."""
    vectordb = get_vectorstore()
    return vectordb.as_retriever()

def get_qa_chain():
    """Creates and returns the QA chain."""
    retriever = get_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=LLM_MODEL,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
   
def pretty_print_answer(result):
    """Pretty prints the answer and sources."""
    print("✅ Answer:")
    print(result["result"])
    print("\n📚 Sources:")
    for doc in result["source_documents"]:
        print(f"- {doc.metadata.get('source', 'Unknown')}")

    return qa_chain
