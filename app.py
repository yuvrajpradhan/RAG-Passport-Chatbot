import streamlit as st
import os
import logging
from dotenv import load_dotenv

from rag_pipeline import create_rag_chain, ingest_data, DATA_DIR, DB_DIR

# Setup simple logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Load Environment ---
load_dotenv()

# --- Checks ---
if not os.getenv("GOOGLE_API_KEY"):
    st.error("Google API key not found. Please set `GOOGLE_API_KEY` in the `.env` file.")
    st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title="Govt. FAQ Chatbot 🇮🇳",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("Passport FAQ Chatbot")
st.caption("A production-grade RAG-powered assistant for answering Passport services questions.")

# --- Initialization & Caching ---
@st.cache_resource(show_spinner="Initializing RAG Chain...")
def get_chain():
    logger.info("Initializing RAG chain...")
    try:
        return create_rag_chain()
    except Exception as e:
        logger.error(f"Error initializing chain: {e}")
        st.error(f"Failed to initialize the chatbot service: {e}")
        st.stop()

rag_chain = get_chain()

# --- Sidebar / Management UI ---
with st.sidebar:
    st.header("Admin Controls")
    st.write("Manage the underlying vector database.")

    if st.button("Reload Knowledge Base", help="Reads data folder and updates vector store"):
        with st.spinner("Ingesting documents and building vector store..."):
            success = ingest_data(DATA_DIR, DB_DIR)
            if success:
                st.success("Successfully updated documents.")
                # Clearing cache to force recreating the chain next run
                get_chain.clear()
                st.rerun()
            else:
                st.error("Document ingestion failed. Check terminal logs.")

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am ready to help. How can I assist you with your Passport queries today?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question related to Passports..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching for answers..."):
            try:
                current_chain = get_chain()
                response = current_chain.invoke({"input": prompt})
                answer = response.get("answer", "I could not generate an answer.")
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                logger.error(f"Error during chain invocation: {e}")
                err_msg = "An error occurred while fetching the answer. Please check backend logs."
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})