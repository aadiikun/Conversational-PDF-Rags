import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


# 🎯 Load embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 🌈 Streamlit page setup
st.set_page_config(page_title="Conversational PDF Chat", page_icon="📘")
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
    }
    h1, h2, h3, h4 {
        text-align: center;
        color: #2C3E50;
    }
    </style>
""", unsafe_allow_html=True)

# 🧠 App header
st.title("🤖 Conversational RAG with PDF Uploads and Chat History")
st.write("Chat with the content of your PDF using Groq + LangChain 🔍")

# 🔑 Groq API input
api_key = st.text_input("🔐 Enter your Groq API key:", type="password")

if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")
    session_id = st.text_input("💬 Session ID:", value="default_session")

    if "store" not in st.session_state:
        st.session_state.store = {}

    # 📂 PDF upload section
    uploaded_files = st.file_uploader("📁 Choose your PDF file:", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        documents = []
        st.info("📄 Processing your uploaded PDF(s)... please wait ⏳")

        for uploaded_file in uploaded_files:
            temp_pdf = f"./temp_{uploaded_file.name}"
            with open(temp_pdf, "wb") as file:
                file.write(uploaded_file.getvalue())
            loader = PyPDFLoader(temp_pdf)
            docs = loader.load()
            documents.extend(docs)

        # ✂️ Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)

        # 🧬 Create vector store
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()
        st.success("✅ PDF successfully embedded into the vector store!")

        # 🧩 Contextualization
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            Given the chat history and the latest user question,
            rewrite it into a standalone question if necessary.
            """),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # 💬 System prompt for Q&A
        system_prompt = """
        You are a helpful assistant that answers questions using the provided context.
        If the answer is not in the context, say you don't know. Be clear and concise.
        """ + "\n\n{context}"

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        # ⚙️ Combine into a chain
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # 💾 Maintain chat history
        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # ✏️ Chat input
        st.divider()
        st.subheader("💭 Ask something about your document:")
        user_input = st.text_input("Type your question here 👇")

        if user_input:
            session_history = get_session_history(session_id)
            with st.spinner("🧠 Thinking..."):
                response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}},
                )
            st.success("🤖 Assistant:")
            st.write(response["answer"])

            st.divider()
            st.caption("🕒 Chat History:")
            st.write(session_history.messages)

else:
    st.warning("⚠️ Please enter your Groq API Key to start chatting!")
