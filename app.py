import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Set Groq API environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"
os.environ["OPENAI_API_TYPE"] = "open_ai"  # Optional


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}  # ✅ Force CPU to avoid meta device errors
    )
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local("faiss_index")


def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}  # ✅ Force CPU
    )
    vector_store = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vector_store


def get_conversionation_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details.
    If the answer is not in the context, just say that you don't know, don't provide an answer.
    Context: \n{context}?\n
    Question: \n{question}\n
    """
    model = ChatOpenAI(model_name="llama3-8b-8192", temperature=0.7)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}  # ✅ Force CPU
    )

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversionation_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    print(response)
    st.write("Reply: ", response['output_text'])


def main():
    st.set_page_config("Chat with Multiple PDFs")
    st.header("Chat with Multiple PDFs")

    user_question = st.text_input("Ask a question about the PDFs")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload PDF files and Click on Submit & Process", type="pdf", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("PDFs processed successfully!")


if __name__ == "__main__":
    main()
