import os
import logging
from typing import Optional
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_google_genai import ChatGoogleGenerativeAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__name__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(BASE_DIR, "db")

EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gemini-1.5-flash-latest"

def _get_embeddings() -> HuggingFaceEmbeddings:
    """Returns the configured embeddings model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)

def ingest_data(data_dir: str = DATA_DIR, db_dir: str = DB_DIR) -> bool:
    """
    Ingests documents from the data directory and builds/refreshes the Chroma database.
    """
    try:
        logger.info(f"Loading documents from {data_dir}...")
        loader = DirectoryLoader(data_dir, glob="**/*.txt")
        docs = loader.load()

        if not docs:
            logger.warning("No text documents found in the data directory.")
            return False

        logger.info(f"Loaded {len(docs)} document(s). Splitting into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        logger.info(f"Creating embeddings and saving to Chroma at {db_dir}...")
        embeddings = _get_embeddings()
        
        # We process from scratch to ensure we don't have stale embeddings
        _ = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=db_dir
        )
        logger.info("Ingestion completed successfully.")
        return True
    except Exception as e:
        logger.error(f"Error during data ingestion: {e}")
        return False

def get_vectorstore(db_dir: str = DB_DIR) -> Optional[Chroma]:
    """
    Loads the persistent Chroma vector store. Assumes it has been ingested.
    """
    if not os.path.exists(db_dir) or not os.listdir(db_dir):
        logger.error(f"Vector database directory '{db_dir}' does not exist or is empty.")
        return None
    
    try:
        embeddings = _get_embeddings()
        vectorstore = Chroma(
            persist_directory=db_dir,
            embedding_function=embeddings
        )
        return vectorstore
    except Exception as e:
        logger.error(f"Failed to load Chroma vector store: {e}")
        return None

def create_rag_chain():
    """
    Creates and returns the RAG chain connected to the Gemini LLM and Local Vector Store.
    If the vector store does not exist, it runs the ingestion process first.
    """
    vectorstore = get_vectorstore(DB_DIR)
    
    if vectorstore is None:
        logger.info("Vector store not found. Triggering initial ingestion...")
        success = ingest_data(DATA_DIR, DB_DIR)
        if not success:
            raise RuntimeError("Initial data ingestion failed. Please check your data directory.")
        vectorstore = get_vectorstore(DB_DIR)

    if vectorstore is None:
        raise RuntimeError("Could not load vector store even after ingestion step.")

    # Create the LLM instance
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL)

    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context.
    Provide a detailed and concise answer in markdown format. 
    Ensure a polite and professional tone.
    If you don't know the answer from the context, state that the information is not available in the provided documents.

    <context>
    {context}
    </context>

    Question: {input}
    """)

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain