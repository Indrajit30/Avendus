import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from PIL import Image
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader, CSVLoader
import os
import pinecone

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

logo_path = "./avendus.png"
logo_url = "https://www.avendus.com/dist/img/logo.png"

if os.path.exists(logo_path):
    logo = Image.open(logo_path)
else:
    logo = logo_url

def load_files_from_folder(folder_path):
    data = []
    # Load CSV files first
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            print(f'Loading CSV: {file_path}')
            loader = CSVLoader(file_path=file_path)
            data.extend(loader.load())
    # Load PDF files after CSV files
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            file_path = os.path.join(folder_path, filename)
            print(f'Loading PDF: {file_path}')
            loader = PyPDFLoader(file_path)
            data.extend(loader.load())
    return data

def chunk_data(data,chunk_size=1000):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = 100
    )
    chunks = text_splitter.split_documents(data)
    return chunks

def create_embedding_pinecone(chunks, index_name="avendus"):
    import pinecone
    from langchain_community.vectorstores import Pinecone
    from langchain_openai import OpenAIEmbeddings
    from pinecone import PodSpec
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)
    
    # Initialize Pinecone
    pc = pinecone.Pinecone()
    
    # Create Pinecone index
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=PodSpec(
                environment='gcp-starter'
            )
        )

    vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
    return vector_store

def load_embeddings_pinecone(index_name="avendus"):
    from langchain.vectorstores import Pinecone
    from langchain_openai import OpenAIEmbeddings
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)
    
    # Initialize Pinecone 
    vector_store = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)
    return vector_store

def ask_question(q, chain):
    result = chain.invoke({'question': q})
    return result['answer']

if __name__ == "__main__":
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
    system_template = r'''
    You will be provided with a csv of all the deals made in last 3 years but the word 'deal' wont be mentioned in the csv.
    AND the pdf files of these deals are also provided.
    Use the following piece of context to answer the questions.
    If you don't find the answer in the given context, just respond "Not found in the context".
    DO NOT SHOW THE FILE NAME IN YOUR RESPONSE.
    ---------------
    Context: ```{context}```
    '''
    user_template = '''
    Question: ```{question}```
    '''
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(user_template)
    ]   
    qa_prompt = ChatPromptTemplate.from_messages(messages)

    if 'vs' not in st.session_state:
        st.session_state['vs'] = None

    st.image(logo, width=150)
    st.subheader("Transaction GPT")

    # with st.sidebar:
    #     uploaded_folder = st.text_input('Enter the path of the folder containing your files:')
        
    #     if uploaded_folder:
    #         data = load_files_from_folder(uploaded_folder)
    #         chunks = chunk_data(data)
    #         vector_store = create_embedding_pinecone(chunks)
    #         st.session_state['vs'] = vector_store
    # st.text_area('Suggestions:', value="Try: Give me 5 latest horror scripts or Scripts similar to Fast and furious\n or\n Annabelle", height=100)
    q = st.text_input("Enter the question")
    submit = st.button("Submit")
    if submit:
        if q:
            with st.spinner("Running..."):
                if st.session_state['vs'] is not None:
                    vector_store = st.session_state['vs']
                else:
                    vector_store = load_embeddings_pinecone()
                    st.session_state['vs'] = vector_store

                retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 20})
                memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                crc = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=retriever,
                    memory=memory,
                    chain_type="stuff",
                    combine_docs_chain_kwargs={'prompt': qa_prompt},
                    verbose=False
                )
                answer = ask_question(q, crc)
            st.text_area('Answer:', value=answer, height=300)

            st.divider()
            if 'history' not in st.session_state:
                st.session_state.history = ''
            value = f'Q: {q} \nA: {answer}'
            st.session_state.history = f'{value} \n {"-"*100}\n {st.session_state.history}'
            h = st.session_state.history
            st.text_area(label="Chat history",value=h, key='history',height = 300)