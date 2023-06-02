from dotenv import load_dotenv #python-dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI




def main():
    load_dotenv()
    
    #Testing .env access
    #print(os.getenv("OPENAI_API_KEY"))
    
    st.set_page_config(page_title="Talk to your PDF")
    st.header("Talk to it! 🚀")

    #get file from user
    pdf = st.file_uploader(label="Drop your file here:", type="pdf")

    #get text from document
    if pdf is not None:
        reader = PdfReader(pdf)
        text=""
        for page in reader.pages:
            text += page.extract_text()
        
        #chunk the text
        splitter = CharacterTextSplitter(separator="\n", chunk_size=3000, chunk_overlap=200, length_function=len, keep_separator=False)
        chunks = splitter.split_text(text)

        #create embeddings from chunks
        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        knowledge_base = FAISS.from_texts(texts=chunks, embedding=embeddings)

        user_question = st.text_input("What would you like to know?")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)

            st.write(response)



if __name__ == '__main__' :
    main()