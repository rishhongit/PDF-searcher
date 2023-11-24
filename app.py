import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain import vectorstores
from langchain import chains
# from goose3 import Goose
import streamlit as st
from langchain.llms import AI21 
from langchain import embeddings
llm = AI21(ai21_api_key='diNNQzvL40ZnBnEQkIBwNESWjtj792NG')

def main():
    st.set_page_config(page_title="Upload PDF")
    st.header("PDF QA")

    names = ['PDF']
    page = st.radio('Format', names)

    if page == 'PDF':
        pdf = st.file_uploader("Upload your PDF", type="pdf")
        if pdf is not None:
            pdf_reader = PdfReader(pdf)
            texts = ""
            for page in pdf_reader.pages:
                texts += page.extract_text()
            text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size = 1000,
            chunk_overlap = 0
        )
            chunks = text_splitter.split_text(texts)
            embeddings = embeddings.HuggingFaceEmbeddings()
            db = vectorstores.Chroma.from_texts(chunks, embeddings)
            retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":10})
            qa = chains.ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)
            chat_history = []
            query = st.text_input("Ask a question in PDF")
            if query:
                result = qa({"question": query, "chat_history": chat_history})
                st.write(result["answer"])
    
if __name__ == "__main__":
    main()
