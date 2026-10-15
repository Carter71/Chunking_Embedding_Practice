# Import Libraries
from rich import print
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


# Import our local model
local_llm = ChatOllama(model="mistral")

# RAG Function
def rag(chunks, collection_name):
    vectorstore = Chroma.from_documents(
        documents=documents,
        collection_name=collection_name,
        embedding=OllamaEmbeddings(model="nomic-embed-text")
        )
    retriever = vectorstore.as_retriever()

    prompt_template = """Answer the question based only on the following context: 
    {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(prompt_template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | local_llm
        | StrOutputParser()
    )
    results = chain.invoke("What is the use of Text Splitting?")
    print(results)

# 1. Charecter Text Splitting
print("--- Charecter Text Splitting ---")
text = "Text splitting in LangChain is a critical feature that facilitates the division of large texts into smaller, manageable segments."

# Manual Splitting
chunks = []
chunk_size = 35 # This is by number of charecters
for i in range(0, len(text), chunk_size):
    chunk = text[i:i + chunk_size]
    chunks.append(chunk)
documents = [Document(page_content=chunk, metadata={"source": "local"}) for chunk in chunks]
print(documents)