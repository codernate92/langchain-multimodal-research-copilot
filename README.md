# LangChain Multimodal Ingestion

This project ingests PDFs, images, and audio files using LangChain and embeds them into a ChromaDB vector store. 

## Features
- Supports .pdf, .png/.jpg, .mp3/.wav
- Uses LangChain loaders and embeddings
- Stores in local vector database for later retrieval

## Setup
```bash
pip install -r requirements.txt
python ingestion.py
