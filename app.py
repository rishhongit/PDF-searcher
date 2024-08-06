import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain import vectorstores
from langchain import chains
from langchain import llms 
from langchain.embeddings import HuggingFaceEmbeddings
import gradio as gr

load_dotenv()

llm = llms.AI21(ai21_api_key=os.getenv('AI21_API_KEY'))

def process_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    texts = ""
    for page in pdf_reader.pages:
        texts += page.extract_text()
    
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=0
    )
    chunks = text_splitter.split_text(texts)
    embeddings = HuggingFaceEmbeddings()
    db = vectorstores.Chroma.from_texts(chunks, embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":10})
    qa = chains.ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)
    return qa

def answer_question(pdf_file, question, chat_history):
    if not pdf_file:
        return "Please upload a PDF file first."
    
    qa = process_pdf(pdf_file)
    result = qa({"question": question, "chat_history": chat_history})
    chat_history.append((question, result["answer"]))
    return result["answer"]

def main():
    with gr.Blocks() as demo:
        gr.Markdown("# PDF QA")
        with gr.Row():
            pdf_file = gr.File(label="Upload your PDF", file_types=[".pdf"])
            question = gr.Textbox(label="Ask a question about the PDF")
        output = gr.Textbox(label="Answer")
        chat_history = gr.State([])
        submit_btn = gr.Button("Submit")
        submit_btn.click(answer_question, inputs=[pdf_file, question, chat_history], outputs=output)
    
    demo.launch()

if __name__ == "__main__":
    main()
