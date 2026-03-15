# Govt. FAQ Chatbot (Passport Services) 🇮🇳🤖

A production-grade Retrieval-Augmented Generation (RAG) chatbot designed to answer questions related to Indian Passport services. Built with Streamlit and LangChain, it utilizes Google's Gemini 1.5 Flash for natural language understanding and generation, powered by localized text documents.

## 🚀 Features

*   **RAG Powered Answers**: Uses local text files (`/data`) as a knowledge base to provide accurate, context-aware answers.
*   **Production-Grade Architecture**: Designed to be resilient and efficient. Document ingestion and the RAG model are decoupled to ensure fast server startups and optimal performance.
*   **Dynamic Knowledge Base Reloading**: Features an admin control panel directly in the UI to seamlessly refresh the vector database (Chroma DB) whenever new documents are added or modified, without needing to restart the application.
*   **Robust Error Handling**: Incorporates comprehensive try-catch logic and logging. Prevents application crashes and displays graceful error messages for missing API keys, corrupted databases, or backend LLM issues.
*   **Streamlit UI**: A clean, accessible chat interface for users to comfortably ask their passport-related questions.

## 🛠️ Technology Stack

*   **Frontend & Web Framework**: [Streamlit](https://streamlit.io/)
*   **LLM Orchestration**: [LangChain](https://python.langchain.com/)
*   **Generative AI Model**: Google Gemini 1.5 Flash (`langchain-google-genai`)
*   **Embeddings Model**: HuggingFace `all-MiniLM-L6-v2` (`langchain-community`)
*   **Vector Database**: [Chroma](https://www.trychroma.com/) (Stored locally)
*   **Python**: Version 3.8+ Recommended

## ⚙️ Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/yuvrajpradhan/RAG-Passport-Chatbot.git
    cd RAG-Passport-Chatbot
    ```

2.  **Set Up Virtual Environment (Recommended)**
    ```bash
    python -m venv myenv
    # On Windows:
    myenv\Scripts\activate
    # On macOS/Linux:
    source myenv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables**
    *   Create a `.env` file in the root directory.
    *   Add your Google Gemini API key:
        ```env
        GOOGLE_API_KEY=your_google_gemini_api_key_here
        ```

5.  **Add Knowledge Data**
    *   Place any `.txt` files containing your passport FAQ data or context inside the `data/` folder.

6.  **Run the Application**
    ```bash
    streamlit run app.py
    ```

## 🎮 Usage

*   Open the provided local URL (usually `http://localhost:8501`) in your web browser.
*   Type your passport-related questions into the chat input.
*   **Admin Controls**: Expand the sidebar to find the "Reload Knowledge Base" button. Use this whenever you modify the documents in the `data/` folder to update the chatbot's knowledge instantly.

## 📁 Project Structure

```text
├── app.py                 # Main Streamlit application entry point
├── rag_pipeline.py        # Core RAG logic, embeddings, and vector store management
├── data/                  # Directory containing text files (.txt) for the knowledge base
├── db/                    # Persistent directory for the Chroma vector database
├── requirements.txt       # Project dependencies
├── .env                  # Environment variables file (not tracked in Git)
└── .gitignore             # Git ignore file
```
