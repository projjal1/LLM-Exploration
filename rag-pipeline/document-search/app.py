from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOllama
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def process_index_docs(embeddings):
    # File path for sample document
    file_path = "nke-10k-2023.pdf"
    # Load the document
    loader = PyPDFLoader(file_path)

    # Load the data in memory
    docs = loader.load()

    # Data is loaded per page of pdf
    # print(f"{docs[0].page_content[:200]}\n")
    # print(docs[0].metadata)

    # Split the text with overlap so as to not break important sentence
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)

    # Show the length of splits
    # print(len(all_splits))

    # Show vector embeddings
    vector_1 = embeddings.embed_query(all_splits[0].page_content)
    # print(f"Generated vectors of length {len(vector_1)}\n")
    # print(vector_1[:10])

    # Connect to Chroma vector store
    vector_store = Chroma(
        collection_name="financial-document-rag",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
    )
    print("Chroma object loaded")

    # Index all documents
    vector_store.add_documents(documents=all_splits)
    print("All documents indexed")


import os


def main():
    embeddings = OllamaEmbeddings(model="llama3:8b")

    vector_store = Chroma(
        collection_name="financial-document-rag",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",
    )

    if not os.path.isdir("chroma_langchain_db"):
        process_index_docs(embeddings)

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1},
    )

    llm = ChatOllama(model="llama3:8b")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    queries = [
        "How many distribution centers does Nike have in the US?",
        "When was Nike founded?",
        "How were Nike's margins impacted in 2023?",
        "Give a short summary of the document"
    ]

    for query in queries:
        answer = qa_chain.run(query)
        print(f"Q: {query}\nA: {answer}\n")


if __name__ == "__main__":
    main()