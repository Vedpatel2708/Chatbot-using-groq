import streamlit as st
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Set the GROQ API key
os.environ['GROQ_API_KEY'] = 'INSERT YOUR GROQ API KEY'

# Set up Streamlit title and description
st.title("Ved GPT ")
st.write("This interface allows you to interact with LangChain models.")

# Set up language models and components
llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Load PDF document and create vector store
loader = PyPDFLoader("/Users/vedpatel/Downloads/project/BS.pdf")
docs = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

# Define conversation chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Define Streamlit components and functions
conversation_history = []
user_input = st.text_input("You:", "")

if st.button("Send"):
    if user_input:
        system_response = rag_chain.invoke(user_input)
        conversation_history.append(("You:", user_input))
        if hasattr(system_response, "content"):
            conversation_history.append(("System:", system_response.content))
        else:
            conversation_history.append(("System (Full Response):", system_response))

# Display conversation history
st.sidebar.title("Conversation History")
for role, message in conversation_history:
    st.sidebar.write(f"{role} {message}")
