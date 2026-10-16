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

# 1. Character Text Splitting
print("--- Character Text Splitting ---")
text = "Text splitting in LangChain is a critical feature that facilitates the division of large texts into smaller, manageable segments."

# Manual Splitting
chunks = []
chunk_size = 35 # This is by number of charecters
for i in range(0, len(text), chunk_size):
    chunk = text[i:i + chunk_size]
    chunks.append(chunk)
documents = [Document(page_content=chunk, metadata={"source": "local"}) for chunk in chunks]
# print(documents)

# Automatic Text Splitting using LangChain's TextSplitter
from langchain_text_splitters import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=35, chunk_overlap=0, separator="", strip_whitespace=False)
documents = text_splitter.create_documents([text])
# print(documents)

# 2. Recursive Text Splitting
print("--- Recursive Text Splitting ---")
from langchain_text_splitters import RecursiveCharacterTextSplitter
with open('sample.txt', 'r', encoding='utf-8') as file:
    text = file.read()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0, separators=["\n\n", "\n", " ", ""])
documents = text_splitter.create_documents([text])
# print(documents)


# 3. Document Text Splitting

# Document Specific Splitting - Markdown
print("--- Document Text Splitting ---")
from langchain_text_splitters import MarkdownTextSplitter
splitter = MarkdownTextSplitter(chunk_size=40, chunk_overlap=0)

markdown_text = """
# Fun in california

## Driving

Try dribving on the 1 down to san diego

### Food

Make sure to eat a burrito while you're there

## Hiking

Go to Yosemite
"""
print("--- Document Specific Splitting - Markdown ---")
print(splitter.create_documents([markdown_text]))



# Docuemnt Specific Splitting - Python
from langchain_text_splitters import PythonCodeTextSplitter
python_text = """
class Python:
    def __init__(self):
        self.name = name
        self.age = age

p1 = Person("John", 36)

for i in range(10):
    print(i)

"""

python_splitter = PythonCodeTextSplitter(chunk_size=100, chunk_overlap=0)
print("--- Document Specific Splitting - Python ---")
print(python_splitter.create_documents([python_text]))




# Document Specific Splitting - JavaScript
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
javascript_text = """
// Function is calledm the return value will end up in x
let x = myFunction(4, 3);

function myFunction(a, b) {
// Function returns the product of a and b
    return a * b;
}
"""
js_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.JS, chunk_size=65, chunk_overlap=0
)
print("--- Document Specific Splitting - JavaScript ---")
print(js_splitter.create_documents([javascript_text]))

