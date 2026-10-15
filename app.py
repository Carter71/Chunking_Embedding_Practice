# Import Libraries
from rich import print
from langchain.docstore,document import Document
from landchain_community.chat_models import ChatOllama
from landchain_community.vectorstores import Chroma
from landchain_community import embeddings
from landchain_core.runnables import RunnablePassthrough
from landchain_core.output_parsers import StringOutputParser
from landchain_core.prompts import ChatPromptTemplate


# Import our local model
local_llm = ChatOllama(model="mistral")

# RAG Function
def rag(chunks, collection_name):
    vectorstore = Chroma.from_documents(
        documents=documents,
        collection_name=collection_name,
        embedding=embeddings.ollama.OllamaEmbeddings(model="nomic-embed-text")
        )
    retriver = vectorstore.as_retriever()

    prompt_template = """Answer the question based onlt on the following context: 
    {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(prompt_template)

    chain = (
        {"context": retriver, "question": RunnablePassthrough()}
        | prompt
        | local_llm
        | StringOutputParser()
    )
    results = chain.invoke("What is the use of Text Splitting?")
    print(results)

