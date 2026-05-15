import os
import warnings

from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

warnings.filterwarnings("ignore", category=DeprecationWarning)

PDF_PATH = "nke-10k-2023.pdf"
COLLECTION_NAME = "financial-document-rag"
PERSIST_DIR = "./chroma_langchain_db"
MODEL_NAME = "llama3:8b"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def process_index_docs(embeddings):
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )
    vector_store.add_documents(documents=all_splits)
    print(f"Indexed {len(all_splits)} chunks into '{COLLECTION_NAME}'")


def main():
    embeddings = OllamaEmbeddings(model=MODEL_NAME)

    if not os.path.isdir(PERSIST_DIR):
        process_index_docs(embeddings)

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1},
    )

    llm = ChatOllama(model=MODEL_NAME)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    queries = [
        "How many distribution centers does Nike have in the US?",
        "When was Nike founded?",
        "How were Nike's margins impacted in 2023?",
        "Give a short summary of the document",
    ]

    for query in queries:
        answer = qa_chain.run(query)
        print(f"Q: {query}\nA: {answer}\n")


if __name__ == "__main__":
    main()
