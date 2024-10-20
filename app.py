# Streamlit App
import streamlit as st
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from typing import List, Tuple
import pandas as pd
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragatouille import RAGPretrainedModel

# Configure pandas
pd.set_option("display.max_colwidth", None)  

# Caching the RAG model
@st.cache_resource
def load_rag_model():
    return RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

# Caching the embedding model
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="thenlper/gte-small",
        multi_process=True,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

# Function to load FAISS database
@st.cache_resource
def load_faiss_database(_embedding_model):
    return FAISS.load_local(
        "/IndexedData", _embedding_model, allow_dangerous_deserialization=True
    )

# Load all models at the beginning
embedding_model = load_embedding_model()
KNOWLEDGE_VECTOR_DATABASE = load_faiss_database(embedding_model)
RERANKER = load_rag_model()

# Caching the LLM
@st.cache_resource
def load_llm():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceH4/zephyr-7b-alpha",
        quantization_config=bnb_config
    )
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-alpha")
    return pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        do_sample=True,
        temperature=0.2,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens=500
    )

READER_LLM = load_llm()

# Function to answer queries with RAG
def answer_with_rag(
    question: str,
    llm: pipeline,
    knowledge_index,
    reranker=None,
    num_retrieved_docs: int = 30,
    num_docs_final: int = 5,
) -> Tuple[str, List[str]]:
    relevant_docs = knowledge_index.similarity_search(query=question, k=num_retrieved_docs)
    relevant_docs = [doc.page_content for doc in relevant_docs]

    if reranker:
        relevant_docs = reranker.rerank(question, relevant_docs, k=num_docs_final)
        relevant_docs = [doc["content"] for doc in relevant_docs]

    relevant_docs = relevant_docs[:num_docs_final]

    context = "\nExtracted documents:\n" + "".join([f"Document {i}:::\n{doc}" for i, doc in enumerate(relevant_docs)])
    final_prompt = f"Using the information contained in the context,\ngive a detailed answer to the question.\nRespond only to the question asked.\nIf the answer cannot be deduced from the context, do not give an answer.\nContext:\n{context}\n\n---\n\nQuestion: {question}"

    answer = llm(final_prompt)[0]["generated_text"]
    return answer, relevant_docs

# User Profiles Configuration
USER_PROFILES = {
    "student": {
    "additional_context": """
    Assume the student has a basic understanding of the subject but is unfamiliar with more advanced concepts.
    Start by introducing the topic with simple terms and gradually move towards the more complex ideas.
    Use analogies and examples to explain difficult concepts, and break down any technical terms or jargon.
    At the end of your explanation, provide a real-world example or application to solidify the student’s understanding.
    If relevant, include a step-by-step guide on how to approach solving a related problem.
    """,
    "response_style": "clear, detailed, and engaging with simple language",
    "max_tokens": 700  # Increased for more detailed explanations
},
    "researcher": {
        "additional_context": "Provide concise, precise, and technical responses, focusing on the relevant research findings.",
        "response_style": "concise and technical",
        "max_tokens": 300
    },
    "domain_expert": {
        "additional_context": "Give a highly technical explanation assuming the user has deep expertise in the subject matter.",
        "response_style": "highly technical",
        "max_tokens": 1000
    }
}

# Function to get prompt template based on user profile
def get_prompt_template(user_profile: str, context: str, question: str) -> str:
    if user_profile not in USER_PROFILES:
        raise ValueError(f"User profile '{user_profile}' not found. Available profiles: {list(USER_PROFILES.keys())}")

    profile_data = USER_PROFILES[user_profile]

    base_prompt = [
        {
            "role": "system",
            "content": f"""Using the information contained in the context,
give a {profile_data['response_style']} answer to the question. 
Respond only to the question asked. Provide the names of the authors, do not give document numbers.
If the answer cannot be deduced from the context, do not give an answer.
{profile_data['additional_context']}"""
        },
        {
            "role": "user",
            "content": f"""Context:
{context}

---

Question: {question}"""
        }
    ]

    return base_prompt  # Adjust as needed for your tokenizer
# Streamlit interface
st.title("RAG-based Scientific Chatbot")

# Input: user profile selection
user_profile = st.selectbox("Select your profile:", options=list(USER_PROFILES.keys()))

# Input: user query
user_query = st.chat_input("Enter your query:")

if user_query:
    # Display user input as a chat message
    with st.chat_message("user"):
        st.write(user_query)

    # Perform the retrieval first to get the relevant documents
    with st.spinner('Retrieving relevant documents...'):
        relevant_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=user_query, k=5)  # Adjust the number of documents as needed
        #relevant_docs = [doc.page_content for doc in relevant_docs]

    # Construct the context from the retrieved documents
    #context = "\nExtracted documents:\n" + "".join([f"Document {i}:::\n{doc}" for i, doc in enumerate(relevant_docs)])

    # Construct the context from the retrieved documents with metadata
    context = "\nExtracted documents:\n" + "".join(
    [f"**Title:** {doc.metadata['title']}\n**Authors:** {doc.metadata['authors']}\n**Submitter:** {doc.metadata['submitter']}\n**Categories:** {doc.metadata['categories']}\n**Journal Reference:** {doc.metadata['journal reference']}\n\n**Content:**\n{doc.page_content}\n\n" for doc in relevant_docs])

    # Generate the prompt based on user profile
    prompt = get_prompt_template(user_profile, context, user_query)

    # Perform the generation with the LLM
    with st.spinner('Generating an answer...'):
        answer, _ = answer_with_rag(user_query, READER_LLM, KNOWLEDGE_VECTOR_DATABASE, reranker=None)

    # Display the generated answer as a chat message
    with st.chat_message("assistant"):
        st.write(answer)

    # Display retrieved documents with metadata
    with st.expander("Retrieved Documents"):
        for i, doc in enumerate(relevant_docs):
            st.markdown(f"#### Document {i + 1}")
            st.markdown(f"**Title:** {doc.metadata['title']}")  # Assuming there's a title
            st.markdown(f"**Authors:** {doc.metadata['authors']}")  # Assuming there's an authors field
            st.markdown(f"**Submitter:** {doc.metadata['submitter']}")  # Display submitter info
            st.markdown(f"**Categories:** {doc.metadata['categories']}")  # Display categories info
            st.markdown(f"**Journal Reference:** {doc.metadata['journal reference']}")  # Display journal reference
            st.write(doc.page_content)  # Display each document content
