from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

DOCUMENTS_DIR = Path("./docs")
CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"
documents = []

# PDF DOCUMENT ID MAPPING
DOCUMENT_IDS = {
    "payment-service-runbook.pdf":
        "payment-service-runbook",

    "payment-service-architecture.pdf":
        "payment-service-architecture",

    "payment-incident-0814.pdf":
        "payment-incident-0814",

    "payment-incident-0821.pdf":
        "payment-incident-0821",

    "postgres-troubleshooting.pdf":
        "postgres-troubleshooting",

    "redis-troubleshooting.pdf":
        "redis-troubleshooting",

    "kubernetes-production-guide.pdf":
        "kubernetes-production-guide",

    "kubernetes-payment-deployment.pdf":
        "kubernetes-payment-deployment",
}

for pdf_path in DOCUMENTS_DIR.glob("*.pdf"):

    print(f"Loading PDF: {pdf_path.name}")

    loader = PyPDFLoader(str(pdf_path))

    pages = loader.load()

    document_id = DOCUMENT_IDS.get(
        pdf_path.name,
        pdf_path.stem,
    )

    for page in pages:
        page.metadata["documentId"] = document_id
        page.metadata["sourceFile"] = pdf_path.name

    documents.extend(pages)

if not documents:
    raise RuntimeError(
        f"No PDFs found in {DOCUMENTS_DIR.resolve()}"
    )

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
)

chunks = splitter.split_documents(documents)

# Stable IDs help prevent duplicate chunks during re-ingestion.
chunk_ids = [
    (
        f"{doc.metadata['documentId']}"
        f"-page-{doc.metadata.get('page', 0)}"
        f"-chunk-{i}"
    )
    for i, doc in enumerate(chunks)
]

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
)

vectorstore = Chroma(
    collection_name="devops_graphrag",
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)

vectorstore.add_documents(
    documents=chunks,
    ids=chunk_ids,
)

print(f"Ingested {len(chunks)} chunks into ChromaDB.")
