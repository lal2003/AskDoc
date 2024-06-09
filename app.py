import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
# Assuming you have set your Google API key in the environment variable 'google_api_key'
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("google_api_key")

# Reading the PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

# PDF to vector format
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain 

# User input
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True
    )
    return response["output_text"]

def main():
    st.set_page_config(page_title="EduMinds Multiple PDF Reader")
    st.header("Chat with Multiple PDF")

    if 'conversation' not in st.session_state:
        st.session_state['conversation'] = []

    if 'input' not in st.session_state:
        st.session_state['input'] = ""

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF files and click on the Submit", accept_multiple_files=True, type=["pdf"])
        if st.button("Submit & Process"):
            with st.spinner("Processing"):
                if pdf_docs:
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")
                else:
                    st.error("Please upload PDF files.")

    # Display conversation history
    for chat in st.session_state['conversation']:
        st.write("**You:**", chat['question'])
        st.write("**🤖**", chat['answer'])
        st.write("---")

    def clear_input():
        st.session_state['input'] = ""

    # Input field with a key and a callback for clearing input
    user_question = st.text_input("Ask a question from the PDF files", key="input")
    
    def send_question():
        if user_question:
            response = user_input(user_question)
            st.session_state['conversation'].append({"question": user_question, "answer": response})
            clear_input()  # Clear the input field after interaction

    if st.button("Send", on_click=send_question):
        pass

if __name__ == "__main__":
    main()
