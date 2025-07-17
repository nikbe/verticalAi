import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

# Set your OpenAI API key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Header
st.markdown("<h2 style='color:#e20074;'>T-bot</h2>", unsafe_allow_html=True)

# Upload interface
with st.sidebar:
    st.title("T-Docs")
    file = st.file_uploader("Upload 1 page PDF document & start asking questions", type="pdf")

# Extract text
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    # Break text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Generate embeddings using OpenAI
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Create Vector Store
    vector_store = FAISS.from_texts(chunks, embeddings)

    # Input from user
    user_question = st.text_input("Type your Question")

    # Perform similarity search and display results
    if user_question:
        match = vector_store.similarity_search(user_question)
        # st.write(match)

        #define llm
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            temperature=0,
            max_tokens=1000,
            model = "gpt-3.5-turbo"
        )

        chain = load_qa_chain(llm,chain_type="stuff")
        response = chain.run(input_documents=match,question=user_question)
        st.write(response)


