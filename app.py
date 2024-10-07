from flask import Flask, request, jsonify
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragatouille import RAGPretrainedModel
from langchain_community.vectorstores.utils import DistanceStrategy
import logging
from langchain_community.vectorstores import FAISS
import time  # Optional: for simulating time taken for indexing
import faiss
import boto3
import os

logging.basicConfig(level=logging.DEBUG)

# Initialize Flask app
app = Flask(__name__)

# S3 Configuration (replace with your S3 bucket and directory)
S3_BUCKET_NAME = 'chatboot-ressources'
S3_BASE_DIR = 'quantized_model/'


# Initialize S3 client
s3 = boto3.client('s3')

def download_file_from_s3(s3_key, local_path):
    """Download a file from S3 to the local path on the EC2 instance."""
    try:
        s3.download_file(S3_BUCKET_NAME, s3_key, local_path)
        logging.info(f"Downloaded {s3_key} from S3 to {local_path}")
    except Exception as e:
        logging.error(f"Error downloading {s3_key}: {str(e)}")
# Check if the model files already exist locally before downloading from S3
def download_file_from_s3_if_not_exists(s3_key, local_path):
    if not os.path.exists(local_path):
        download_file_from_s3(s3_key, local_path)
        
def load_resources():
    global KNOWLEDGE_VECTOR_DATABASE, embedding_model, READER_LLM, RERANKER, RAG_PROMPT_TEMPLATE
    logging.debug("Loading resources...")

    # Ensure directories for storing models exist
    local_model_dir = "quantized_model/"
    os.makedirs(local_model_dir, exist_ok=True)

    # Define local paths for model files and docs
    local_docs_processed = "docs_processed.pkl"

    # Download necessary files from S3
    download_file_from_s3_if_not_exists(S3_BASE_DIR + 'docs_processed.pkl', local_docs_processed)
    model_files = [
        'config.json', 'generation_config.json', 'model.safetensors', 'special_tokens_map.json',
        'tokenizer_config.json', 'tokenizer.json', 'tokenizer.model'
    ]

    for file in model_files:
        download_file_from_s3_if_not_exists(S3_BASE_DIR + file, os.path.join(local_model_dir, file))

    # Load processed documents
    with open(local_docs_processed, "rb") as f:
        docs_processed = pickle.load(f)
    logging.debug("Documents processed loaded")

    # Load embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name="thenlper/gte-small",  
        multi_process=True,
        model_kwargs={"device": "cuda"},  # Switch to GPU
        encode_kwargs={"normalize_embeddings": True}
    )
    logging.debug("Embedding model loaded")

    # Indexing docs
    total_docs = len(docs_processed)  
    logging.debug("Indexing docs...")

    KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
        docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
    )

    for i in range(total_docs):
        time.sleep(0.1)  # Simulate processing time, adjust/remove in production
        percent_complete = (i + 1) / total_docs * 100
        logging.debug(f"Indexing progress: {percent_complete:.2f}%")

    logging.debug("Knowledge vector database initialized")

    # Load reranker model
    RERANKER = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
    logging.debug("Reranker model loaded")

    # Load quantized language model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(local_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
    logging.debug("Language model and tokenizer loaded")

    # Initialize text generation pipeline
    READER_LLM = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        do_sample=True,
        temperature=0.2,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens=500
    )
    logging.debug("Reader LLM initialized")

    # Define the RAG prompt template
    prompt_in_chat_format = [
        {
            "role": "system",
            "content": """Using the information contained in the context, give a comprehensive detailed answer to the question."""
        },
        {
            "role": "user",
            "content": """Context:\n{context}\n---\nNow here is the question you need to answer.\nQuestion: {question}"""
        }
    ]
    RAG_PROMPT_TEMPLATE = tokenizer.apply_chat_template(
        prompt_in_chat_format, tokenize=False, add_generation_prompt=True
    )
    logging.debug("RAG prompt template defined")

@app.route('/ask', methods=['POST'])
def ask():
    # Parse user query
    data = request.get_json()
    user_query = data.get("question", "")
    
    # Retrieve relevant documents
    retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=user_query, k=5)
    retrieved_docs_text = [doc.page_content for doc in retrieved_docs]
    
    # Rerank relevant documents
    relevant_docs = RERANKER.rerank(user_query, retrieved_docs_text, k=5)
    relevant_docs = [doc["content"] for doc in relevant_docs]

    # Construct context from retrieved documents
    context = "\nExtracted documents:\n"
    context += "".join([f"Document {str(i)}::\n" + doc for i, doc in enumerate(relevant_docs)])

    # Format the final prompt
    final_prompt = RAG_PROMPT_TEMPLATE.format(question=user_query, context=context)

    # Generate the answer from the LLM
    answer = READER_LLM(final_prompt)[0]["generated_text"]
    return jsonify({"answer": answer, "relevant_docs": relevant_docs})

# Main Flask app entry point
if __name__ == '__main__':
    load_resources()
    logging.debug("Starting Flask app...")
    app.run(host='0.0.0.0', port=5000)
