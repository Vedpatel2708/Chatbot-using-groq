


from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

os.environ['GROQ_API_KEY']='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'

llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")

system = "You are a helpful assistant."
human = "{text}"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

chain = prompt | llm

res = chain.invoke({"text": "Explain the type of LLM Agents in langchain."})

res.content

from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

sentences = ["This is an example sentence", "Each sentence is converted"]
sent_embeddings = embeddings.embed_documents(sentences)
print(sent_embeddings)
print(f"emb size: {len(sent_embeddings[0])}")



# Load, chunk and index the contents of the blog.
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("/content/BS.pdf")
docs = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke("Do you think management has the characteristics of a fullfledged profession?")
