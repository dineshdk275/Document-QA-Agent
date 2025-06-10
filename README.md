# 🧠 Chat with Multiple PDFs using LLMs (Groq + LangChain + FAISS + Streamlit)

This Streamlit app allows you to chat with the content of multiple PDF documents using a Groq-hosted LLM (like LLaMA 3). It uses LangChain for pipeline orchestration, FAISS for semantic search, and HuggingFace sentence transformers for embeddings.

## 🚀 Features
- 📄 Upload multiple PDF documents
- 🔍 Ask questions based on their content
- 💬 Get detailed, context-aware answers using Groq LLaMA-3
- 🧠 Uses FAISS for fast similarity search
- 🔒 Embeddings handled locally using HuggingFace



## 📦 Tech Stack
| Component        | Tech Used                                     |
|------------------|-----------------------------------------------|
| UI               | Streamlit                                     |
| LLM API          | Groq (OpenAI-compatible endpoint for LLaMA)   |
| Embeddings       | `sentence-transformers/all-MiniLM-L6-v2`      |
| Vector Store     | FAISS (Local)                                 |
| Orchestration    | LangChain                                     |
| File Parsing     | PyPDF2                                        |
| Env Variables    | python-dotenv                                 |



### 📂 Folder Structure

```bash
📦 Document QA AI Agent/
├── main.py                  # Streamlit app main file
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (not pushed to GitHub)
├── faiss_index/             # Auto-created FAISS vector store
└── README.md                # Project documentation
```



## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/pdf-chat-groq.git
cd pdf-chat-groq
```
### 2. Install Dependencies
#### Make sure you're using Python 3.9+
``` bash
pip install -r requirements.txt
```
### 3. Set Up Your .env File
#### Create a .env file in the root directory with:
```bash
GROQ_API_KEY=your_groq_api_key
```
#### 🔑 You can get your free Groq API key here: https://console.groq.com
### 4. Run the App
```bash
streamlit run main.py
```

### 🧪 Example Use Case
#### Upload your academic PDFs or company reports.
#### Ask questions like:

- “What were the findings on page 12?”
- “Summarize the conclusion section.”
- “What is the definition of cloud computing?”



### 🛡️ License
#### This project is licensed under the MIT License. Feel free to use and modify it.

### 🙌 Acknowledgements
- Groq API
- LangChain
- FAISS by Facebook
- Sentence Transformers
- Streamlit


