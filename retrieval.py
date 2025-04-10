# retrieval.py

import os
from dotenv import load_dotenv

load_dotenv()

from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA

EMBEDDING_MODEL = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
LLM_MODEL = OpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))

def load_retriever():
    print("🔄 Loading FAISS vectorstore...")
    vectordb = FAISS.load_local("faiss_store", EMBEDDING_MODEL)
    return vectordb.as_retriever()

def get_qa_chain():
    retriever = load_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=LLM_MODEL,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

def ask_question(question):
    qa = get_qa_chain()
    print(f"❓ {question}")
    result = qa({"query": question})
    print("✅ Answer:")
    print(result["result"])
    print("\n📚 Sources:")
    for doc in result["source_documents"]:
        print(f"- {doc.metadata.get('source', 'Unknown')}")

if __name__ == "__main__":
    ask_question("What are the key points in the document?")
