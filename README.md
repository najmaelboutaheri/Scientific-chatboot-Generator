# **Flask-based AI Knowledge Retrieval System with S3 Integration**
This project demonstrates a Flask-based web application for knowledge retrieval using a Retrieval-Augmented Generation (RAG) approach. It integrates with Amazon S3 to download pre-trained models and documents, leverages FAISS for similarity search, and uses the HuggingFace Transformers and LangChain libraries to enable efficient large language model (LLM) responses. Key features include:

- **Document Embedding & Retrieval:** Uses FAISS for vector-based document similarity search.
- **Language Model Integration:** Incorporates a quantized language model and reranker model for enhanced query responses.
- **S3 Integration:** Downloads models and processed documents from an S3 bucket.
- **Interactive API:** Accepts user questions via a ```/ask``` API endpoint, retrieves relevant documents, and generates answers using a pre-defined prompt.
